import json
import math
import os
import subprocess
import platform
import threading
import logging
from datetime import datetime
import ezdxf
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import tkinter as tk
from tkinter import filedialog, Tk, Toplevel, Text, Scrollbar, Frame, Label, Button
from simple_viewer import EngineeringViewer

# ==========================================
# 日誌設定 (Logging Configuration)
# ==========================================

def setup_logging():
    """
    設定日誌系統
    日誌檔案儲存在 logs/yyyymmdd.log
    """
    # 建立 logs 目錄
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成日誌檔案名稱 (yyyymmdd.log)
    today = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"{today}.log")

    # 設定日誌格式
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 建立 logger
    logger = logging.getLogger("AutoDrafter")
    logger.setLevel(logging.DEBUG)

    # 清除現有的 handlers（避免重複）
    if logger.handlers:
        logger.handlers.clear()

    # 檔案 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 初始化 logger
logger = setup_logging()

def log_print(message: str, level: str = "info", console: bool = False):
    """
    輸出到日誌檔案（可選擇是否同時輸出到終端機）
    Args:
        message: 要輸出的訊息
        level: 日誌級別 (debug, info, warning, error)
        console: 是否同時輸出到終端機，預設 False
    """
    if console:
        print(message)
    if level == "debug":
        logger.debug(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)

# 延遲導入 cadquery，避免啟動時的導入錯誤
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError as e:
    CADQUERY_AVAILABLE = False
    logger.warning(f"CadQuery not available: {e}")
    logger.warning("3D file loading will be disabled. Using mock data only.")

# ==========================================
# 1. 核心資料結構 (模擬 3D CAD 實體)
# ==========================================

@dataclass
class GeometricFeature:
    """代表 3D 模型中的一個特徵 (如孔、邊、面)"""
    id: str
    type: str  # 'circle', 'line', 'rect'
    params: Dict[str, float]  # 例如 {'radius': 5.0, 'x': 0, 'y': 0}
    description: str # 用於 AI 語意匹配，例如 "center_mounting_hole"

class MockCADEngine:
    """
    模擬真實的 CAD 核心 (如 FreeCAD/SolidWorks API)
    負責實際的幾何運算與 3D->2D 投影
    支援從 STEP/STL 檔案讀取 3D 模型
    """
    def __init__(self, model_file: Optional[str] = None):
        """
        初始化 CAD 引擎
        Args:
            model_file: 可選的 3D 模型檔案路徑 (.step, .stp, .stl)
                        如果為 None，則使用模擬資料
        """
        self.model_file = model_file
        self.cad_model = None  # 儲存 CadQuery 物件
        self._xcaf_doc = None  # XCAF 文檔（用於組件結構）
        self._xcaf_shape_tool = None  # XCAF ShapeTool（用於遍歷組件樹）

        if model_file and os.path.exists(model_file):
            self.load_3d_file(model_file)
        else:
            # 模擬載入了一個帶有一個孔的方塊
            self.features = [
                GeometricFeature("F01", "rect", {'w': 100, 'h': 100, 'x': 0, 'y': 0}, "base_plate"),
                GeometricFeature("F02", "circle", {'radius': 10, 'x': 0, 'y': 0}, "center_hole")
            ]
            log_print("[CAD Kernel] Using mock data (no 3D file loaded)")
    
    def load_3d_file(self, filename: str):
        """
        從檔案讀取 3D 模型 (支援 STEP/STL)
        並提取幾何特徵
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"找不到檔案: {filename}")
        
        # 檢查 cadquery 是否可用
        if not CADQUERY_AVAILABLE:
            log_print("[CAD Kernel] Warning: CadQuery not available, cannot load 3D file", "warning")
            log_print("[CAD Kernel] Falling back to mock data", "warning")
            self.features = [
                GeometricFeature("F01", "rect", {'w': 100, 'h': 100, 'x': 0, 'y': 0}, "base_plate"),
                GeometricFeature("F02", "circle", {'radius': 10, 'x': 0, 'y': 0}, "center_hole")
            ]
            return
        
        file_ext = os.path.splitext(filename)[1].lower()
        log_print(f"[CAD Kernel] Loading 3D file: {filename} ({file_ext})")

        try:
            if file_ext in ['.step', '.stp']:
                # 嘗試使用 XCAF 讀取 STEP 檔案（保留完整組件結構）
                xcaf_success = self._load_step_with_xcaf(filename)
                if not xcaf_success:
                    # 回退到 CadQuery 的 importStep
                    self.cad_model = cq.importers.importStep(filename)
                    log_print("[CAD Kernel] STEP file loaded with CadQuery")
            elif file_ext == '.stl':
                # STL 是網格格式，使用 OCP (OpenCASCADE) 直接讀取
                log_print("[CAD Kernel] Loading STL file (mesh-based)...")

                # 先提取 STL 元資料（即使形狀載入失敗也可以顯示基本資訊）
                self._extract_stl_metadata(filename)

                stl_shape = self._load_stl_with_ocp(filename)
                if stl_shape is not None:
                    self.cad_model = cq.Workplane("XY").newObject([cq.Shape(stl_shape)])
                    log_print("[CAD Kernel] STL file loaded successfully")
                else:
                    # 即使形狀載入失敗，也建立一個基本的模型佔位符
                    log_print("[CAD Kernel] STL shape loading failed, using metadata only", "warning")
                    # 建立一個空的 workplane 作為佔位符
                    self.cad_model = None
                    # 仍然設定基本特徵（從元資料）
                    if hasattr(self, '_stl_metadata') and self._stl_metadata:
                        tri_count = self._stl_metadata.get('triangle_count', 0)
                        solid_name = self._stl_metadata.get('solid_name', os.path.basename(filename))
                        self.features = [
                            GeometricFeature("F01", "stl_mesh",
                                           {'triangles': tri_count, 'name': solid_name},
                                           f"STL Mesh: {solid_name}")
                        ]
                        log_print(f"[CAD Kernel] STL metadata extracted: {tri_count} triangles")
                        return  # 跳過後續的 _extract_features
            else:
                raise ValueError(f"不支援的檔案格式: {file_ext}")
            
            # 從 3D 模型中提取特徵
            self.features = self._extract_features()
            log_print(f"[CAD Kernel] Extracted {len(self.features)} features from 3D model")

        except Exception as e:
            log_print(f"[CAD Kernel] Error loading file: {e}", "error")
            log_print("[CAD Kernel] Falling back to mock data", "warning")
            self.features = [
                GeometricFeature("F01", "rect", {'w': 100, 'h': 100, 'x': 0, 'y': 0}, "base_plate"),
                GeometricFeature("F02", "circle", {'radius': 10, 'x': 0, 'y': 0}, "center_hole")
            ]

    def _load_step_with_xcaf(self, filename: str) -> bool:
        """
        使用 XCAF (XDE) 讀取 STEP 檔案，保留完整的組件結構和變換矩陣
        這是處理組件實例（如多個相同螺栓放置在不同位置）的正確方法

        Returns:
            bool: 是否成功使用 XCAF 載入
        """
        try:
            from OCP.STEPCAFControl import STEPCAFControl_Reader
            from OCP.XCAFDoc import XCAFDoc_DocumentTool
            from OCP.TDocStd import TDocStd_Document
            from OCP.TCollection import TCollection_ExtendedString
            from OCP.XCAFDoc import XCAFDoc_ShapeTool
            from OCP.TDF import TDF_LabelSequence
            from OCP.TopoDS import TopoDS_Compound
            from OCP.BRep import BRep_Builder
            from OCP.IFSelect import IFSelect_RetDone

            log_print("[CAD Kernel] Attempting XCAF-based STEP loading...")

            # 創建 XCAF 文檔
            doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))

            # 創建 STEP 讀取器
            reader = STEPCAFControl_Reader()
            reader.SetColorMode(True)
            reader.SetNameMode(True)
            reader.SetLayerMode(True)

            # 讀取 STEP 檔案
            status = reader.ReadFile(filename)
            if status != IFSelect_RetDone:
                log_print("[CAD Kernel] XCAF reader failed to read file, falling back to CadQuery", "warning")
                return False

            # 傳輸到文檔
            if not reader.Transfer(doc):
                log_print("[CAD Kernel] XCAF transfer failed, falling back to CadQuery", "warning")
                return False

            # 獲取 ShapeTool
            shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

            # 存儲 XCAF 數據供後續使用
            self._xcaf_doc = doc
            self._xcaf_shape_tool = shape_tool

            # 獲取所有自由形狀（根級組件）
            free_shapes = TDF_LabelSequence()
            shape_tool.GetFreeShapes(free_shapes)

            if free_shapes.Length() == 0:
                log_print("[CAD Kernel] No shapes found in XCAF document, falling back to CadQuery", "warning")
                return False

            # 構建複合形狀（用於 CadQuery 兼容性）
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)

            for i in range(1, free_shapes.Length() + 1):
                label = free_shapes.Value(i)
                shape = shape_tool.GetShape(label)
                if not shape.IsNull():
                    builder.Add(compound, shape)

            # 創建 CadQuery Workplane 以保持兼容性
            self.cad_model = cq.Workplane("XY").newObject([cq.Shape(compound)])

            log_print(f"[CAD Kernel] STEP file loaded with XCAF ({free_shapes.Length()} root shapes)")
            return True

        except ImportError as e:
            log_print(f"[CAD Kernel] XCAF modules not available: {e}", "warning")
            return False
        except Exception as e:
            log_print(f"[CAD Kernel] XCAF loading failed: {e}", "warning")
            return False

    def _load_stl_with_ocp(self, filename: str):
        """
        使用 OCP (OpenCASCADE) 直接讀取 STL 檔案

        STL 是網格格式，OpenCASCADE 可以將其讀取為 TopoDS_Shape

        Args:
            filename: STL 檔案路徑

        Returns:
            TopoDS_Shape 或 None（如果讀取失敗）
        """
        # 確保路徑格式正確（Windows 兼容）
        import os
        filename = os.path.abspath(filename)
        log_print(f"[CAD Kernel] Loading STL: {filename}")

        # 方法 1: 嘗試使用 StlAPI_Reader
        try:
            from OCP.StlAPI import StlAPI_Reader
            from OCP.TopoDS import TopoDS_Shape

            log_print("[CAD Kernel] Trying StlAPI_Reader...")
            reader = StlAPI_Reader()
            shape = TopoDS_Shape()

            # StlAPI_Reader.Read 返回布林值
            if reader.Read(shape, filename):
                if not shape.IsNull():
                    log_print("[CAD Kernel] STL loaded with StlAPI_Reader")
                    return shape
                else:
                    log_print("[CAD Kernel] StlAPI_Reader returned null shape", "warning")
            else:
                log_print("[CAD Kernel] StlAPI_Reader.Read() returned False", "warning")

        except Exception as e:
            log_print(f"[CAD Kernel] StlAPI_Reader failed: {e}", "warning")

        # 方法 2: 使用 RWStl 讀取三角網格
        try:
            from OCP.RWStl import RWStl
            from OCP.TopoDS import TopoDS_Face, TopoDS_Compound
            from OCP.BRep import BRep_Builder
            from OCP.Message import Message_ProgressRange

            log_print("[CAD Kernel] Trying RWStl.ReadFile_s...")
            progress = Message_ProgressRange()
            triangulation = RWStl.ReadFile_s(filename, progress)

            if triangulation is not None and triangulation.NbTriangles() > 0:
                log_print(f"[CAD Kernel] RWStl loaded {triangulation.NbTriangles()} triangles")

                # 建立帶有 triangulation 的 face
                builder = BRep_Builder()
                face = TopoDS_Face()
                builder.MakeFace(face, triangulation)

                # 包裝成 compound
                compound = TopoDS_Compound()
                builder.MakeCompound(compound)
                builder.Add(compound, face)

                log_print("[CAD Kernel] STL converted to TopoDS_Compound")
                return compound
            else:
                log_print("[CAD Kernel] RWStl returned empty triangulation", "warning")

        except Exception as e:
            log_print(f"[CAD Kernel] RWStl failed: {e}", "warning")

        # 方法 3: 使用 numpy-stl 讀取並轉換
        try:
            from stl import mesh as stl_mesh
            import numpy as np
            from OCP.gp import gp_Pnt
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
            from OCP.TopoDS import TopoDS_Compound
            from OCP.BRep import BRep_Builder

            log_print("[CAD Kernel] Trying numpy-stl...")
            stl_data = stl_mesh.Mesh.from_file(filename)

            if len(stl_data.vectors) > 0:
                log_print(f"[CAD Kernel] numpy-stl loaded {len(stl_data.vectors)} triangles")

                builder = BRep_Builder()
                compound = TopoDS_Compound()
                builder.MakeCompound(compound)

                # 將每個三角形轉換為 face
                for triangle in stl_data.vectors[:1000]:  # 限制數量避免過慢
                    try:
                        p1 = gp_Pnt(float(triangle[0][0]), float(triangle[0][1]), float(triangle[0][2]))
                        p2 = gp_Pnt(float(triangle[1][0]), float(triangle[1][1]), float(triangle[1][2]))
                        p3 = gp_Pnt(float(triangle[2][0]), float(triangle[2][1]), float(triangle[2][2]))

                        polygon = BRepBuilderAPI_MakePolygon(p1, p2, p3, True)
                        if polygon.IsDone():
                            wire = polygon.Wire()
                            face_maker = BRepBuilderAPI_MakeFace(wire, True)
                            if face_maker.IsDone():
                                builder.Add(compound, face_maker.Face())
                    except:
                        continue

                log_print("[CAD Kernel] STL converted via numpy-stl")
                return compound

        except ImportError:
            log_print("[CAD Kernel] numpy-stl not available", "warning")
        except Exception as e:
            log_print(f"[CAD Kernel] numpy-stl failed: {e}", "warning")

        log_print("[CAD Kernel] All STL loading methods failed", "error")
        return None

    def _extract_solids_from_xcaf(self, features: List, seen_solids: set, start_feature_id: int) -> bool:
        """
        使用 XCAF 標籤樹提取實體，正確累積組件實例的變換矩陣

        這是處理 STEP 組件的最準確方法，因為 XCAF 保留了完整的組件層次結構
        和每個實例的變換矩陣（XCAFDoc_ShapeTool.GetLocation）

        Args:
            features: 特徵列表（會被修改）
            seen_solids: 已處理的實體集合（用於去重）
            start_feature_id: 起始特徵 ID

        Returns:
            bool: 是否成功提取
        """
        try:
            from OCP.TDF import TDF_LabelSequence
            from OCP.TopAbs import TopAbs_SOLID
            from OCP.GProp import GProp_GProps
            from OCP.BRepGProp import BRepGProp
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            from OCP.gp import gp_Trsf
            from OCP.TopLoc import TopLoc_Location
            from OCP.Bnd import Bnd_Box
            from OCP.BRepBndLib import BRepBndLib

            shape_tool = self._xcaf_shape_tool
            feature_id = start_feature_id

            def process_label(label, parent_trsf: gp_Trsf):
                """
                遞歸處理 XCAF 標籤，累積變換矩陣
                """
                nonlocal feature_id

                # 獲取此標籤的位置變換
                loc = shape_tool.GetLocation(label)
                current_trsf = loc.Transformation()

                # 累積變換
                accumulated_trsf = gp_Trsf()
                accumulated_trsf.Multiply(parent_trsf)
                accumulated_trsf.Multiply(current_trsf)

                # 檢查是否為組件（有子組件）
                if shape_tool.IsAssembly(label):
                    # 獲取子組件
                    components = TDF_LabelSequence()
                    shape_tool.GetComponents(label, components)
                    for i in range(1, components.Length() + 1):
                        child_label = components.Value(i)
                        process_label(child_label, accumulated_trsf)

                elif shape_tool.IsSimpleShape(label):
                    # 這是一個簡單形狀（可能包含實體）
                    shape = shape_tool.GetShape(label)
                    if not shape.IsNull():
                        # 移除形狀自身的位置
                        shape_no_loc = shape.Located(TopLoc_Location())

                        # 檢查形狀類型
                        if shape.ShapeType() == TopAbs_SOLID:
                            try:
                                # 應用累積變換
                                transformer = BRepBuilderAPI_Transform(shape_no_loc, accumulated_trsf, True)
                                moved_solid = transformer.Shape()

                                # 計算質心和體積
                                props = GProp_GProps()
                                BRepGProp.VolumeProperties_s(moved_solid, props)
                                centroid = props.CentreOfMass()
                                volume = props.Mass()

                                cx = centroid.X()
                                cy = centroid.Y()
                                cz = centroid.Z()

                                # 計算個別實體的邊界框 (Bounding Box)
                                solid_bbox = Bnd_Box()
                                BRepBndLib.Add_s(moved_solid, solid_bbox)
                                xmin, ymin, zmin, xmax, ymax, zmax = solid_bbox.Get()
                                bbox_l = abs(xmax - xmin)
                                bbox_w = abs(ymax - ymin)
                                bbox_h = abs(zmax - zmin)

                                # 去重
                                key = (round(cx, 2), round(cy, 2), round(cz, 2), round(volume, 2))
                                if key not in seen_solids:
                                    seen_solids.add(key)
                                    features.append(GeometricFeature(
                                        f"F{feature_id:02d}",
                                        "solid",
                                        {'volume': volume, 'x': cx, 'y': cy, 'z': cz,
                                         'bbox_l': bbox_l, 'bbox_w': bbox_w, 'bbox_h': bbox_h},
                                        f"solid_{len(seen_solids)}"
                                    ))
                                    feature_id += 1

                            except Exception as e:
                                log_print(f"[CAD Kernel] Warning: Could not process XCAF solid: {e}", "warning")

                elif shape_tool.IsReference(label):
                    # 這是一個引用（指向另一個標籤）
                    ref_label = shape_tool.GetReferredShape(label)
                    if not ref_label.IsNull():
                        process_label(ref_label, accumulated_trsf)

            # 獲取所有自由形狀並處理
            free_shapes = TDF_LabelSequence()
            shape_tool.GetFreeShapes(free_shapes)

            identity_trsf = gp_Trsf()
            for i in range(1, free_shapes.Length() + 1):
                label = free_shapes.Value(i)
                process_label(label, identity_trsf)

            return len(seen_solids) > 0

        except Exception as e:
            log_print(f"[CAD Kernel] XCAF extraction error: {e}", "warning")
            return False

    def _extract_features(self) -> List[GeometricFeature]:
        """
        從 CadQuery 3D 模型中提取幾何特徵
        使用 OCC API 正確提取每個特徵的實際參數
        重要：使用 XCAF 遍歷組件樹並累積變換矩陣以獲取全域座標
        """
        features = []
        feature_id = 1

        if self.cad_model is None:
            return features

        try:
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_SOLID, TopAbs_EDGE
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            from OCP.GeomAbs import GeomAbs_Circle
            from OCP.GProp import GProp_GProps
            from OCP.BRepGProp import BRepGProp
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            from OCP.gp import gp_Trsf
            from OCP.TopLoc import TopLoc_Location
            from OCP.Bnd import Bnd_Box
            from OCP.BRepBndLib import BRepBndLib

            # 獲取模型的邊界框 (Bounding Box)
            bbox = self.cad_model.val().BoundingBox()
            width = bbox.xmax - bbox.xmin
            height = bbox.ymax - bbox.ymin
            depth = bbox.zmax - bbox.zmin
            center_x = (bbox.xmin + bbox.xmax) / 2
            center_y = (bbox.ymin + bbox.ymax) / 2

            # 提取基本外型 (矩形底座)
            features.append(GeometricFeature(
                f"F{feature_id:02d}",
                "rect",
                {'w': width, 'h': height, 'x': center_x, 'y': center_y},
                "base_plate"
            ))
            feature_id += 1

            # 獲取根形狀
            root_shape = self.cad_model.val().wrapped

            # ===== 提取實體特徵（使用 XCAF 累積變換）=====
            seen_solids = set()
            solid_count = 0

            # 嘗試使用 XCAF 標籤樹（更準確的組件遍歷）
            xcaf_extracted = False
            if hasattr(self, '_xcaf_shape_tool') and self._xcaf_shape_tool is not None:
                try:
                    xcaf_extracted = self._extract_solids_from_xcaf(
                        features, seen_solids, feature_id
                    )
                    if xcaf_extracted:
                        feature_id += len(seen_solids)
                        log_print(f"[CAD Kernel] 使用 XCAF 標籤樹提取了 {len(seen_solids)} 個實體")
                except Exception as e:
                    log_print(f"[CAD Kernel] XCAF extraction failed, using fallback: {e}", "warning")
                    xcaf_extracted = False

            if not xcaf_extracted:
                # 使用遞歸形狀樹遍歷作為後備方案
                def extract_solids_with_transform(shape, parent_trsf: gp_Trsf):
                    """
                    遞歸遍歷形狀樹，累積從根到葉的變換矩陣
                    這確保組件實例獲得正確的全域座標
                    """
                    nonlocal feature_id, solid_count

                    # 獲取當前形狀的位置變換
                    loc = shape.Location()
                    current_trsf = loc.Transformation()

                    # 累積變換：parent_trsf * current_trsf
                    accumulated_trsf = gp_Trsf()
                    accumulated_trsf.Multiply(parent_trsf)
                    accumulated_trsf.Multiply(current_trsf)

                    # 檢查此形狀是否為實體
                    shape_type = shape.ShapeType()
                    from OCP.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND, TopAbs_COMPSOLID

                    if shape_type == TopAbs_SOLID:
                        solid_count += 1
                        try:
                            # 移除形狀自身的位置（因為我們要用累積變換）
                            shape_no_loc = shape.Located(TopLoc_Location())

                            # 應用累積變換以獲取全域座標
                            transformer = BRepBuilderAPI_Transform(shape_no_loc, accumulated_trsf, True)
                            moved_solid = transformer.Shape()

                            # 計算移動後實體的質心（全域座標）
                            props = GProp_GProps()
                            BRepGProp.VolumeProperties_s(moved_solid, props)
                            centroid = props.CentreOfMass()
                            volume = props.Mass()

                            cx = centroid.X()
                            cy = centroid.Y()
                            cz = centroid.Z()

                            # 計算個別實體的邊界框 (Bounding Box)
                            solid_bbox = Bnd_Box()
                            BRepBndLib.Add_s(moved_solid, solid_bbox)
                            xmin, ymin, zmin, xmax, ymax, zmax = solid_bbox.Get()
                            bbox_l = abs(xmax - xmin)
                            bbox_w = abs(ymax - ymin)
                            bbox_h = abs(zmax - zmin)

                            # 建立唯一鍵（避免重複）
                            key = (round(cx, 2), round(cy, 2), round(cz, 2), round(volume, 2))

                            if key not in seen_solids:
                                seen_solids.add(key)
                                features.append(GeometricFeature(
                                    f"F{feature_id:02d}",
                                    "solid",
                                    {'volume': volume, 'x': cx, 'y': cy, 'z': cz,
                                     'bbox_l': bbox_l, 'bbox_w': bbox_w, 'bbox_h': bbox_h},
                                    f"solid_{len(seen_solids)}"
                                ))
                                feature_id += 1

                        except Exception as e:
                            log_print(f"[CAD Kernel] Warning: Could not process solid {solid_count}: {e}", "warning")

                    elif shape_type in (TopAbs_COMPOUND, TopAbs_COMPSOLID):
                        # 遞歸遍歷複合形狀的子元素
                        from OCP.TopoDS import TopoDS_Iterator
                        iterator = TopoDS_Iterator(shape)
                        while iterator.More():
                            child_shape = iterator.Value()
                            extract_solids_with_transform(child_shape, accumulated_trsf)
                            iterator.Next()

                # 從單位變換開始遍歷
                identity_trsf = gp_Trsf()
                extract_solids_with_transform(root_shape, identity_trsf)

                log_print(f"[CAD Kernel] 提取了 {len(seen_solids)} 個實體特徵（全域座標，累積變換）")

            # ===== 提取圓形特徵（應用全域變換）=====
            # 重要：必須將邊緣的 TopLoc_Location 應用到圓心座標
            seen_circles = set()

            def extract_circles_with_transform(shape, parent_trsf: gp_Trsf):
                """
                遞歸遍歷形狀樹，提取圓形邊緣並應用累積變換
                """
                nonlocal feature_id

                # 獲取當前形狀的位置變換
                loc = shape.Location()
                current_trsf = loc.Transformation()

                # 累積變換
                accumulated_trsf = gp_Trsf()
                accumulated_trsf.Multiply(parent_trsf)
                accumulated_trsf.Multiply(current_trsf)

                shape_type = shape.ShapeType()
                from OCP.TopAbs import TopAbs_EDGE, TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE

                if shape_type == TopAbs_EDGE:
                    # 這是一條邊緣
                    try:
                        # 移除邊緣的位置（使用累積變換）
                        edge_no_loc = shape.Located(TopLoc_Location())
                        curve = BRepAdaptor_Curve(edge_no_loc)

                        if curve.GetType() == GeomAbs_Circle:
                            circle = curve.Circle()

                            # 獲取局部圓心
                            local_center = circle.Location()

                            # 應用累積變換到圓心
                            from OCP.gp import gp_Pnt
                            global_center = gp_Pnt(local_center.X(), local_center.Y(), local_center.Z())
                            global_center.Transform(accumulated_trsf)

                            cx = global_center.X()
                            cy = global_center.Y()
                            cz = global_center.Z()
                            radius = circle.Radius()

                            # 去重
                            key = (round(cx, 2), round(cy, 2), round(cz, 2), round(radius, 2))
                            if key not in seen_circles:
                                seen_circles.add(key)
                                features.append(GeometricFeature(
                                    f"F{feature_id:02d}",
                                    "circle",
                                    {'radius': radius, 'x': cx, 'y': cy, 'z': cz},
                                    f"hole_{len(seen_circles)}"
                                ))
                                feature_id += 1

                    except Exception:
                        pass

                elif shape_type in (TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE):
                    # 遞歸處理子元素
                    from OCP.TopoDS import TopoDS_Iterator
                    iterator = TopoDS_Iterator(shape)
                    while iterator.More():
                        child_shape = iterator.Value()
                        extract_circles_with_transform(child_shape, accumulated_trsf)
                        iterator.Next()

            try:
                # 從根形狀開始遞歸提取圓形
                identity_trsf = gp_Trsf()
                extract_circles_with_transform(root_shape, identity_trsf)

            except Exception as e:
                log_print(f"[CAD Kernel] Warning: Could not extract circles: {e}", "warning")

            log_print(f"[CAD Kernel] 提取了 {len(seen_circles)} 個圓形特徵（全域座標）")

        except ImportError as e:
            log_print(f"[CAD Kernel] Warning: OCC API not available: {e}", "warning")
            features = [
                GeometricFeature("F01", "rect", {'w': 100, 'h': 100, 'x': 0, 'y': 0}, "base_plate"),
            ]
        except Exception as e:
            log_print(f"[CAD Kernel] Error extracting features: {e}", "error")
            # 返回基本特徵
            features = [
                GeometricFeature("F01", "rect", {'w': 100, 'h': 100, 'x': 0, 'y': 0}, "base_plate"),
            ]

        return features

    def modify_feature(self, feature_id: str, parameter: str, value: float, operation: str = "set"):
        """執行幾何修改"""
        for f in self.features:
            if f.id == feature_id:
                current_val = f.params.get(parameter)
                if current_val is None:
                    raise ValueError(f"Parameter {parameter} not found on feature {f.id}")
                
                # 根據操作類型更新數值
                if operation == "multiply":
                    f.params[parameter] = current_val * value
                elif operation == "add":
                    f.params[parameter] = current_val + value
                else: # set
                    f.params[parameter] = value
                
                log_print(f"[CAD Kernel] Modified {f.description}: {parameter} -> {f.params[parameter]}")
                return True
        return False

    def get_model_info(self) -> Dict:
        """
        取得 3D 模型的詳細資訊
        返回包含模型資訊的字典，包括 BOM、來源軟體、零件、單位等
        """
        info = {
            "has_model": self.cad_model is not None,
            "model_file": self.model_file,
            "file_name": os.path.basename(self.model_file) if self.model_file else None,
            "file_extension": os.path.splitext(self.model_file)[1].lower() if self.model_file else None,
            "file_size": None,
            "file_type": None,        # 檔案類型描述
            "is_binary": None,        # 是否為 binary 格式
            "is_encrypted": None,     # 是否加密
            "read_status": "OK",      # 讀取狀態
            "read_errors": [],        # 讀取錯誤列表
            "features": [],
            "bounding_box": None,
            # 擴展資訊
            "source_software": None,  # 來源軟體
            "units": None,            # 單位
            "parts": [],              # 零件列表
            "bom": [],                # BOM (Bill of Materials)
            "author": None,           # 作者
            "organization": None,     # 組織
            "creation_date": None,    # 建立日期
            "description": None,      # 描述
            "product_name": None,     # 產品名稱
            "version": None,          # 版本
            "solid_count": 0,         # 實體數量
            "face_count": 0,          # 面數量
            "edge_count": 0,          # 邊數量
            "vertex_count": 0,        # 頂點數量
            "volume": None,           # 體積
            "surface_area": None,     # 表面積
            # STL 特有資訊
            "triangle_count": 0,      # 三角形數量 (STL)
            "stl_solid_name": None,   # STL solid 名稱
        }

        # 檢查檔案類型和格式
        if self.model_file and os.path.exists(self.model_file):
            try:
                # 檢查是否為 binary 檔案
                with open(self.model_file, 'rb') as f:
                    header = f.read(1024)
                    # 檢查是否為 ASCII STEP 檔案
                    if b'ISO-10303-21' in header:
                        info["file_type"] = "STEP (ISO-10303-21)"
                        info["is_binary"] = False
                    elif header.startswith(b'solid') or b'facet normal' in header:
                        info["file_type"] = "STL (ASCII)"
                        info["is_binary"] = False
                    elif header[:80].replace(b'\x00', b'').strip() == b'' or not header[:80].startswith(b'solid'):
                        # STL binary 通常前 80 bytes 是 header
                        info["file_type"] = "STL (Binary)"
                        info["is_binary"] = True
                    else:
                        # 嘗試判斷是否為文字檔案
                        try:
                            header.decode('utf-8')
                            info["is_binary"] = False
                        except:
                            info["is_binary"] = True
                            info["file_type"] = "Binary 格式"

                    # 檢查是否可能加密（簡單檢測）
                    # 加密檔案通常有高熵值或特殊標頭
                    if info["is_binary"] and b'ENCRYPTED' in header.upper():
                        info["is_encrypted"] = True
                        info["read_status"] = "檔案可能已加密"
                        info["read_errors"].append("偵測到加密標記，部分資訊可能無法讀取")
                    else:
                        info["is_encrypted"] = False

            except Exception as e:
                info["read_errors"].append(f"檔案格式檢查失敗: {e}")

        # 獲取檔案大小
        if self.model_file and os.path.exists(self.model_file):
            try:
                file_size_bytes = os.path.getsize(self.model_file)
                if file_size_bytes < 1024:
                    info["file_size"] = f"{file_size_bytes} bytes"
                elif file_size_bytes < 1024 * 1024:
                    info["file_size"] = f"{file_size_bytes / 1024:.2f} KB"
                else:
                    info["file_size"] = f"{file_size_bytes / (1024 * 1024):.2f} MB"
            except:
                pass

        # 如果有 3D 模型，獲取詳細資訊
        if self.cad_model is not None and CADQUERY_AVAILABLE:
            try:
                # 獲取邊界框資訊
                bbox = self.cad_model.val().BoundingBox()
                info["bounding_box"] = {
                    "x_min": bbox.xmin,
                    "x_max": bbox.xmax,
                    "y_min": bbox.ymin,
                    "y_max": bbox.ymax,
                    "z_min": bbox.zmin,
                    "z_max": bbox.zmax,
                    "width": bbox.xmax - bbox.xmin,
                    "height": bbox.ymax - bbox.ymin,
                    "depth": bbox.zmax - bbox.zmin
                }
            except Exception as e:
                log_print(f"[CAD Kernel] Warning: Could not get bounding box: {e}", "warning")

            # 嘗試獲取幾何統計資訊
            try:
                shape = self.cad_model.val()

                # 計算面、邊、頂點數量
                try:
                    from OCP.TopExp import TopExp_Explorer
                    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID

                    # 計算實體數量
                    solid_explorer = TopExp_Explorer(shape.wrapped, TopAbs_SOLID)
                    solid_count = 0
                    while solid_explorer.More():
                        solid_count += 1
                        solid_explorer.Next()
                    info["solid_count"] = solid_count

                    # 計算面數量
                    face_explorer = TopExp_Explorer(shape.wrapped, TopAbs_FACE)
                    face_count = 0
                    while face_explorer.More():
                        face_count += 1
                        face_explorer.Next()
                    info["face_count"] = face_count

                    # 計算邊數量
                    edge_explorer = TopExp_Explorer(shape.wrapped, TopAbs_EDGE)
                    edge_count = 0
                    while edge_explorer.More():
                        edge_count += 1
                        edge_explorer.Next()
                    info["edge_count"] = edge_count

                    # 計算頂點數量
                    vertex_explorer = TopExp_Explorer(shape.wrapped, TopAbs_VERTEX)
                    vertex_count = 0
                    while vertex_explorer.More():
                        vertex_count += 1
                        vertex_explorer.Next()
                    info["vertex_count"] = vertex_count

                except Exception as e:
                    log_print(f"[CAD Kernel] Warning: Could not count topology: {e}", "warning")

                # 嘗試計算體積和表面積
                try:
                    from OCP.GProp import GProp_GProps
                    from OCP.BRepGProp import BRepGProp

                    props = GProp_GProps()
                    BRepGProp.VolumeProperties_s(shape.wrapped, props)
                    info["volume"] = props.Mass()

                    surf_props = GProp_GProps()
                    BRepGProp.SurfaceProperties_s(shape.wrapped, surf_props)
                    info["surface_area"] = surf_props.Mass()
                except Exception as e:
                    log_print(f"[CAD Kernel] Warning: Could not calculate volume/area: {e}", "warning")

            except Exception as e:
                log_print(f"[CAD Kernel] Warning: Could not get geometry stats: {e}", "warning")

            # 嘗試從 STEP 檔案讀取元資料
            if self.model_file and info["file_extension"] in ['.step', '.stp']:
                info = self._extract_step_metadata(info)

            # 填充 STL 元資料
            if hasattr(self, '_stl_metadata') and self._stl_metadata:
                info["triangle_count"] = self._stl_metadata.get("triangle_count", 0)
                info["stl_solid_name"] = self._stl_metadata.get("solid_name")
                # 如果沒有產品名稱，使用 STL solid 名稱或檔案名
                if not info["product_name"]:
                    if self._stl_metadata.get("solid_name"):
                        info["product_name"] = self._stl_metadata["solid_name"]
                    elif self.model_file:
                        # 使用檔案名（不含副檔名）作為產品名稱
                        info["product_name"] = os.path.splitext(os.path.basename(self.model_file))[0]

        # 設定預設單位（根據檔案類型推測）
        if info["units"] is None:
            if info["file_extension"] in ['.step', '.stp']:
                info["units"] = "mm (預設)"
            elif info["file_extension"] == '.stl':
                info["units"] = "無單位資訊 (STL 格式)"
            else:
                info["units"] = "未知"

        # 加入特徵資訊
        for f in self.features:
            info["features"].append({
                "id": f.id,
                "type": f.type,
                "description": f.description,
                "params": f.params
            })

        # 從 STEP 檔案解析材料資訊與實體清單
        step_materials = self._parse_step_materials()
        step_entities = self._parse_step_solid_entities()

        # 生成零件列表與分級 BOM
        # 收集所有 solid 特徵（按順序）
        solid_features = [f for f in info["features"] if f["type"] == "solid"]

        # 將 STEP 實體名稱/ID 對應到提取的 solid 特徵
        for idx, feat in enumerate(solid_features):
            params = feat["params"]

            # 取得 STEP 實體資訊（按順序對應）
            if idx < len(step_entities):
                entity_info = step_entities[idx]
                part_name = entity_info['name']
                entity_id = entity_info['entity_id']
            else:
                part_name = f"Solid_{idx + 1}"
                entity_id = "-"

            # 使用實際邊界框尺寸 (L x W x H)
            bbox_l = params.get('bbox_l', 0)
            bbox_w = params.get('bbox_w', 0)
            bbox_h = params.get('bbox_h', 0)
            if bbox_l > 0 or bbox_w > 0 or bbox_h > 0:
                dimension_str = f"{bbox_l:.2f} x {bbox_w:.2f} x {bbox_h:.2f}"
            else:
                volume = params.get('volume', 0)
                if volume > 0:
                    estimated_size = volume ** (1/3)
                    dimension_str = f"≈{estimated_size:.1f}³"
                else:
                    dimension_str = "-"

            info["parts"].append({
                "item": idx + 1,
                "name": part_name,
                "type": "實體 (Solid)",
                "quantity": 1,
                "material": step_materials.get(idx + 1, "未指定"),
                "dimension": dimension_str,
                "entity_id": entity_id,
                "volume": params.get('volume', 0)
            })

        # 自動分組 BOM：偵測零件名稱重複作為分組邊界
        # BOM 1 = 首次出現的名稱序列，BOM 2 = 名稱重複後的序列
        total_parts = len(info["parts"])
        if total_parts > 0:
            seen_names = set()
            split_idx = total_parts  # 預設不分組
            for i, part in enumerate(info["parts"]):
                name = part.get("name", "")
                if name in seen_names:
                    # 發現重複名稱，此處為分組邊界
                    split_idx = i
                    break
                seen_names.add(name)

            bom_groups = []

            group1 = info["parts"][:split_idx]
            if group1:
                bom_groups.append({
                    "group_name": f"BOM 1 - 基礎實體組件 (Part 1-{len(group1)})",
                    "items": group1
                })

            group2 = info["parts"][split_idx:]
            if group2:
                bom_groups.append({
                    "group_name": f"BOM 2 - 延伸/鏡射組件 (Part {split_idx + 1}-{total_parts})",
                    "items": group2
                })

            info["bom_groups"] = bom_groups

        # 保留 bom 列表以向後相容
        for idx, part in enumerate(info["parts"], 1):
            info["bom"].append({
                "item": idx,
                "name": part["name"],
                "quantity": part["quantity"],
                "material": part["material"],
                "dimension": part["dimension"],
                "entity_id": part.get("entity_id", "-"),
            })

        return info

    def _extract_step_metadata(self, info: Dict) -> Dict:
        """
        從 STEP 檔案中提取元資料
        STEP 檔案包含豐富的元資料，如作者、組織、軟體來源等
        """
        if not self.model_file or not os.path.exists(self.model_file):
            return info

        try:
            with open(self.model_file, 'r', encoding='utf-8', errors='ignore') as f:
                # 只讀取前 500 行來查找 header 資訊
                lines = []
                for i, line in enumerate(f):
                    if i > 500:
                        break
                    lines.append(line)
                content = ''.join(lines)

            # 提取來源軟體
            import re

            # 提取產品名稱 - 從 PRODUCT 實體中查找
            # STEP 格式: PRODUCT('id', 'Product Name', 'description', (context));
            product_patterns = [
                r"PRODUCT\s*\(\s*'[^']*'\s*,\s*'([^']+)'",  # PRODUCT 第二個參數是產品名稱
                r"PRODUCT_DEFINITION_FORMATION.*?'([^']+)'",
                r"SHAPE_DEFINITION_REPRESENTATION.*?PRODUCT.*?'([^']+)'",
            ]
            for pattern in product_patterns:
                product_match = re.search(pattern, content, re.IGNORECASE)
                if product_match and product_match.group(1) and product_match.group(1).strip():
                    name = product_match.group(1).strip()
                    # 過濾掉明顯不是產品名稱的內容（如 ID、空字串、純數字）
                    if name and not name.isdigit() and len(name) > 1:
                        info["product_name"] = name
                        break

            # 查找來源軟體 - 常見格式
            software_patterns = [
                r"ORIGINATING_SYSTEM\s*\(\s*'([^']+)'",
                r"FILE_DESCRIPTION.*?implementation_level.*?'([^']+)'",
                r"preprocessor_version\s*=\s*'([^']+)'",
                r"originating_system\s*=\s*'([^']+)'",
                # 常見 CAD 軟體名稱
                r"(SolidWorks|CATIA|AutoCAD|Inventor|Fusion\s*360|Creo|NX|FreeCAD|OpenSCAD|Onshape)",
            ]

            for pattern in software_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    info["source_software"] = match.group(1).strip()
                    break

            # 查找作者
            author_match = re.search(r"AUTHOR\s*\(\s*'([^']*)'", content, re.IGNORECASE)
            if author_match and author_match.group(1):
                info["author"] = author_match.group(1)

            # 查找組織
            org_match = re.search(r"ORGANIZATION\s*\(\s*'([^']*)'", content, re.IGNORECASE)
            if org_match and org_match.group(1):
                info["organization"] = org_match.group(1)

            # 查找日期
            date_match = re.search(r"TIME_STAMP\s*\(\s*'([^']*)'", content, re.IGNORECASE)
            if date_match and date_match.group(1):
                info["creation_date"] = date_match.group(1)

            # 查找描述
            desc_match = re.search(r"FILE_DESCRIPTION\s*\(\s*\(\s*'([^']*)'", content, re.IGNORECASE)
            if desc_match and desc_match.group(1):
                info["description"] = desc_match.group(1)

            # 查找單位
            unit_patterns = [
                (r"SI_UNIT.*?\.MILLI\.", "mm"),
                (r"SI_UNIT.*?\.CENTI\.", "cm"),
                (r"SI_UNIT.*?METRE", "m"),
                (r"LENGTH_UNIT.*?MILLI", "mm"),
                (r"LENGTH_UNIT.*?INCH", "inch"),
            ]

            for pattern, unit in unit_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    info["units"] = unit
                    break

        except Exception as e:
            log_print(f"[CAD Kernel] Warning: Could not extract STEP metadata: {e}", "warning")

        return info

    def _parse_step_materials(self) -> Dict:
        """
        從 STEP 檔案解析材料資訊

        解析 STEP 實體包括:
        - MATERIAL_DESIGNATION: 材料指定
        - DRAUGHTING_PRE_DEFINED_COLOUR: 預定義顏色（可推測材料）
        - STYLED_ITEM / PRESENTATION_STYLE: 樣式資訊
        - DESCRIPTIVE_REPRESENTATION_ITEM: 描述性項目

        Returns:
            Dict: 零件索引 -> 材料名稱 的對應
        """
        materials = {}

        if not self.model_file or not os.path.exists(self.model_file):
            return materials

        file_ext = os.path.splitext(self.model_file)[1].lower()
        if file_ext not in ['.step', '.stp']:
            return materials

        try:
            with open(self.model_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            import re

            # 1. 解析 MATERIAL_DESIGNATION (標準材料定義)
            # 格式: #n=MATERIAL_DESIGNATION('材料名稱', #ref);
            material_pattern = r"#(\d+)\s*=\s*MATERIAL_DESIGNATION\s*\(\s*'([^']*)'"
            for match in re.finditer(material_pattern, content, re.IGNORECASE):
                entity_id = int(match.group(1))
                material_name = match.group(2).strip()
                if material_name:
                    materials[entity_id] = material_name

            # 2. 解析 DESCRIPTIVE_REPRESENTATION_ITEM（描述性項目，常包含材料）
            # 格式: #n=DESCRIPTIVE_REPRESENTATION_ITEM('name', 'description');
            desc_pattern = r"#(\d+)\s*=\s*DESCRIPTIVE_REPRESENTATION_ITEM\s*\(\s*'([^']*)'\s*,\s*'([^']*)'"
            for match in re.finditer(desc_pattern, content, re.IGNORECASE):
                entity_id = int(match.group(1))
                name = match.group(2).strip().upper()
                value = match.group(3).strip()
                # 常見材料相關關鍵字
                if any(kw in name for kw in ['MATERIAL', 'MATL', '材料', '材質']):
                    if value:
                        materials[entity_id] = value

            # 3. 解析顏色資訊推測材料
            # DRAUGHTING_PRE_DEFINED_COLOUR 或 COLOUR_RGB
            colour_materials = {
                'red': '紅色 (可能為銅/紅銅)',
                'green': '綠色',
                'blue': '藍色',
                'yellow': '黃色 (可能為黃銅)',
                'white': '白色 (可能為鋁/不鏽鋼)',
                'black': '黑色 (可能為碳鋼)',
                'silver': '銀色 (可能為鋁/不鏽鋼)',
                'grey': '灰色 (可能為鑄鐵)',
                'gray': '灰色 (可能為鑄鐵)',
                'orange': '橙色 (可能為銅)',
            }

            colour_pattern = r"#(\d+)\s*=\s*DRAUGHTING_PRE_DEFINED_COLOUR\s*\(\s*'([^']*)'"
            for match in re.finditer(colour_pattern, content, re.IGNORECASE):
                entity_id = int(match.group(1))
                colour_name = match.group(2).strip().lower()
                if colour_name in colour_materials:
                    materials[entity_id] = colour_materials[colour_name]

            # 4. 解析 PRODUCT 中的描述欄位（有時包含材料）
            # 格式: #n=PRODUCT('id', 'name', 'description', (context));
            product_pattern = r"#(\d+)\s*=\s*PRODUCT\s*\(\s*'[^']*'\s*,\s*'([^']*)'\s*,\s*'([^']*)'"
            for match in re.finditer(product_pattern, content, re.IGNORECASE):
                entity_id = int(match.group(1))
                product_name = match.group(2).strip()
                description = match.group(3).strip()
                # 檢查描述是否包含材料關鍵字
                combined = f"{product_name} {description}".upper()
                material_keywords = {
                    'SUS304': 'SUS304 不鏽鋼',
                    'SUS316': 'SUS316 不鏽鋼',
                    'SS304': 'SS304 不鏽鋼',
                    'SS316': 'SS316 不鏽鋼',
                    'STAINLESS': '不鏽鋼',
                    'STEEL': '鋼',
                    'CARBON STEEL': '碳鋼',
                    'ALUMINUM': '鋁',
                    'ALUMINIUM': '鋁',
                    'COPPER': '銅',
                    'BRASS': '黃銅',
                    'BRONZE': '青銅',
                    'IRON': '鐵',
                    'CAST IRON': '鑄鐵',
                    'PLASTIC': '塑膠',
                    'PVC': 'PVC',
                    'PTFE': 'PTFE (鐵氟龍)',
                    'RUBBER': '橡膠',
                    'TITANIUM': '鈦',
                    'A105': 'A105 碳鋼',
                    'A182': 'A182 合金鋼',
                    'A193': 'A193 合金鋼螺栓',
                    'A194': 'A194 碳鋼/合金鋼螺帽',
                }
                for kw, mat_name in material_keywords.items():
                    if kw in combined:
                        materials[entity_id] = mat_name
                        break

            # 如果沒有找到任何材料，嘗試從整體內容推測
            if not materials:
                content_upper = content.upper()
                for kw, mat_name in material_keywords.items():
                    if kw in content_upper:
                        materials[1] = mat_name  # 預設給第一個零件
                        break

            log_print(f"[CAD Kernel] 從 STEP 解析到 {len(materials)} 個材料資訊")

        except Exception as e:
            log_print(f"[CAD Kernel] Warning: 解析 STEP 材料失敗: {e}", "warning")

        return materials

    @staticmethod
    def _decode_step_unicode(text: str) -> str:
        """
        解碼 STEP ISO 10303-21 的 Unicode 編碼字串
        例如: \\X2\\51E04F554F53\\X0\\.1 → 几何体.1
        """
        import re
        def replace_x2(m):
            hex_str = m.group(1)
            chars = []
            for i in range(0, len(hex_str), 4):
                if i + 4 <= len(hex_str):
                    code_point = int(hex_str[i:i+4], 16)
                    chars.append(chr(code_point))
            return ''.join(chars)
        return re.sub(r'\\X2\\([0-9A-Fa-f]+)\\X0\\', replace_x2, text)

    def _parse_step_solid_entities(self) -> list:
        """
        從 STEP 檔案解析 MANIFOLD_SOLID_BREP 實體清單
        取得每個實體的 STEP ID 和名稱

        Returns:
            list of dict: [{'entity_id': '#21', 'name': '几何体.1'}, ...]
        """
        entities = []
        if not self.model_file or not os.path.exists(self.model_file):
            return entities

        file_ext = os.path.splitext(self.model_file)[1].lower()
        if file_ext not in ['.step', '.stp']:
            return entities

        try:
            import re
            with open(self.model_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 解析 ADVANCED_BREP_SHAPE_REPRESENTATION 取得實體 ID 順序
            brep_repr_pattern = r"ADVANCED_BREP_SHAPE_REPRESENTATION\s*\(\s*'[^']*'\s*,\s*\(([^)]+)\)"
            brep_match = re.search(brep_repr_pattern, content)
            ordered_ids = []
            if brep_match:
                ids_str = brep_match.group(1)
                ordered_ids = [s.strip() for s in ids_str.split(',') if s.strip().startswith('#')]

            # 解析所有 MANIFOLD_SOLID_BREP 實體
            msb_pattern = r"(#\d+)\s*=\s*MANIFOLD_SOLID_BREP\s*\(\s*'([^']*)'"
            msb_dict = {}
            for match in re.finditer(msb_pattern, content):
                eid = match.group(1)
                raw_name = match.group(2).strip()
                decoded_name = self._decode_step_unicode(raw_name)
                msb_dict[eid] = decoded_name

            # 依照 ADVANCED_BREP_SHAPE_REPRESENTATION 中的順序排列
            if ordered_ids:
                for eid in ordered_ids:
                    if eid in msb_dict:
                        entities.append({
                            'entity_id': eid,
                            'name': msb_dict[eid]
                        })
            else:
                # 沒有找到排序資訊，按 entity ID 數字排序
                for eid in sorted(msb_dict.keys(), key=lambda x: int(x[1:])):
                    entities.append({
                        'entity_id': eid,
                        'name': msb_dict[eid]
                    })

            log_print(f"[CAD Kernel] 從 STEP 解析到 {len(entities)} 個 MANIFOLD_SOLID_BREP 實體")

        except Exception as e:
            log_print(f"[CAD Kernel] Warning: 解析 STEP 實體失敗: {e}", "warning")

        return entities

    def _extract_stl_metadata(self, filename: str):
        """
        從 STL 檔案中提取元資料
        STL 檔案格式較簡單，主要提取：
        - 檔案名稱作為產品名稱
        - ASCII STL 的 solid 名稱
        - Binary STL 的 header 資訊
        """
        self._stl_metadata = {
            "solid_name": None,
            "is_binary": False,
            "triangle_count": 0,
            "header": None,
        }

        try:
            with open(filename, 'rb') as f:
                header = f.read(80)

                # 檢查是否為 ASCII STL
                if header.startswith(b'solid'):
                    # ASCII STL
                    self._stl_metadata["is_binary"] = False
                    f.seek(0)
                    first_line = f.readline().decode('utf-8', errors='ignore').strip()

                    # 提取 solid 名稱 (solid name)
                    if first_line.startswith('solid'):
                        solid_name = first_line[5:].strip()
                        if solid_name:
                            self._stl_metadata["solid_name"] = solid_name

                    # 計算三角形數量
                    f.seek(0)
                    content = f.read().decode('utf-8', errors='ignore')
                    self._stl_metadata["triangle_count"] = content.count('facet normal')

                else:
                    # Binary STL
                    self._stl_metadata["is_binary"] = True
                    self._stl_metadata["header"] = header.decode('utf-8', errors='ignore').strip()

                    # Binary STL: 80 bytes header + 4 bytes triangle count
                    f.seek(80)
                    triangle_count_bytes = f.read(4)
                    if len(triangle_count_bytes) == 4:
                        import struct
                        self._stl_metadata["triangle_count"] = struct.unpack('<I', triangle_count_bytes)[0]

            log_print(f"[CAD Kernel] STL metadata: {self._stl_metadata['triangle_count']} triangles, "
                     f"{'Binary' if self._stl_metadata['is_binary'] else 'ASCII'} format")

        except Exception as e:
            log_print(f"[CAD Kernel] Warning: Could not extract STL metadata: {e}", "warning")

    def get_step_raw_content(self) -> Dict:
        """
        讀取並返回 STEP 檔案的原始內容
        如無法讀取會加註原因

        Returns:
            Dict 包含:
            - readable: bool 是否可讀取
            - reason: str 無法讀取的原因（如果有）
            - content: str 檔案內容
            - header_section: str HEADER 區段
            - data_section: str DATA 區段（完整）
            - total_lines: int 總行數
            - encoding: str 檔案編碼
        """
        result = {
            "readable": False,
            "reason": None,
            "content": None,
            "header_section": None,
            "data_section": None,
            "total_lines": 0,
            "encoding": None,
            "file_format": None
        }

        if not self.model_file:
            result["reason"] = "未指定檔案路徑"
            return result

        if not os.path.exists(self.model_file):
            result["reason"] = f"檔案不存在: {self.model_file}"
            return result

        # 檢查檔案大小
        try:
            file_size = os.path.getsize(self.model_file)
            if file_size == 0:
                result["reason"] = "檔案大小為 0 bytes（空檔案）"
                return result
            if file_size > 100 * 1024 * 1024:  # 100MB
                result["reason"] = f"檔案過大 ({file_size / (1024*1024):.1f} MB)，超過 100MB 限制"
                return result
        except Exception as e:
            result["reason"] = f"無法取得檔案大小: {e}"
            return result

        # 嘗試以不同編碼讀取
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        used_encoding = None

        for encoding in encodings_to_try:
            try:
                with open(self.model_file, 'r', encoding=encoding) as f:
                    content = f.read()
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
            except PermissionError:
                result["reason"] = "權限不足，無法讀取檔案"
                return result
            except IOError as e:
                result["reason"] = f"I/O 錯誤: {e}"
                return result

        # 如果所有編碼都失敗，嘗試以 binary 模式讀取並檢查
        if content is None:
            try:
                with open(self.model_file, 'rb') as f:
                    raw_data = f.read(4096)

                # 檢查是否為 Binary STL 檔案
                # Binary STL: 80 bytes header + 4 bytes triangle count + triangles
                if len(raw_data) >= 84:
                    import struct
                    # 讀取三角形數量（位於第 80-84 bytes）
                    with open(self.model_file, 'rb') as f:
                        header = f.read(80)
                        tri_count_bytes = f.read(4)
                        if len(tri_count_bytes) == 4:
                            triangle_count = struct.unpack('<I', tri_count_bytes)[0]
                            file_size = os.path.getsize(self.model_file)
                            expected_size = 84 + (triangle_count * 50)  # 每個三角形 50 bytes

                            # 驗證是否為 Binary STL（大小應該接近預期）
                            if abs(file_size - expected_size) < 100:
                                result["readable"] = True
                                result["file_format"] = "STL (Binary)"
                                result["total_lines"] = 0  # Binary 沒有行數概念
                                result["encoding"] = "binary"

                                header_text = header.decode('utf-8', errors='ignore').strip()
                                result["header_section"] = f"""STL 檔案資訊
=====================================
格式: Binary STL
Header: {header_text if header_text else '(空白)'}
三角形數量: {triangle_count:,}
檔案大小: {file_size:,} bytes
====================================="""
                                result["data_section"] = f"Binary STL 檔案包含 {triangle_count:,} 個三角形面\n\n（Binary 格式無法直接顯示文字內容）"
                                return result

                # 檢查是否為其他 binary 格式
                non_printable = sum(1 for byte in raw_data if byte < 32 and byte not in (9, 10, 13))
                if non_printable > len(raw_data) * 0.3:
                    result["reason"] = "檔案為 Binary 格式，無法以文字方式讀取（可能為專有格式或已壓縮）"
                    result["file_format"] = "Binary"
                else:
                    result["reason"] = "無法解碼檔案內容（編碼格式未知）"
                return result
            except Exception as e:
                result["reason"] = f"讀取檔案失敗: {e}"
                return result

        # 驗證是否為有效的 STEP 檔案
        if 'ISO-10303-21' not in content[:1000]:
            # 檢查是否可能是其他格式
            if content.strip().startswith('solid') or 'facet normal' in content[:1000]:
                # 這是 STL (ASCII) 檔案 - 正常處理而非報錯
                result["readable"] = True
                result["content"] = content
                result["encoding"] = used_encoding
                result["file_format"] = "STL (ASCII)"
                result["total_lines"] = len(content.split('\n'))

                # 提取 STL 資訊作為 header_section
                lines = content.split('\n')
                first_line = lines[0].strip() if lines else ""
                solid_name = first_line[5:].strip() if first_line.startswith('solid') else "未命名"
                triangle_count = content.count('facet normal')

                result["header_section"] = f"""STL 檔案資訊
=====================================
格式: ASCII STL
Solid 名稱: {solid_name}
三角形數量: {triangle_count:,}
總行數: {result["total_lines"]:,}
檔案編碼: {used_encoding}
====================================="""

                # 提取前 100 個三角形作為 data_section 預覽
                facet_start = content.find('facet normal')
                if facet_start != -1:
                    preview_end = content.find('endsolid')
                    if preview_end == -1:
                        preview_end = min(len(content), facet_start + 10000)
                    result["data_section"] = content[facet_start:preview_end]
                else:
                    result["data_section"] = content[:5000]

                return result

            elif content.strip().startswith('{') or content.strip().startswith('['):
                result["reason"] = "檔案格式為 JSON，非 STEP 格式"
                result["file_format"] = "JSON"
            elif '<?xml' in content[:100]:
                result["reason"] = "檔案格式為 XML，非 STEP 格式"
                result["file_format"] = "XML"
            else:
                result["reason"] = "檔案不包含 STEP 標準標頭 (ISO-10303-21)，可能已損壞或為其他格式"
            return result

        # 成功讀取
        result["readable"] = True
        result["content"] = content
        result["encoding"] = used_encoding
        result["file_format"] = "STEP (ISO-10303-21)"

        # 計算總行數
        lines = content.split('\n')
        result["total_lines"] = len(lines)

        # 提取 HEADER 區段
        import re
        header_match = re.search(r'HEADER\s*;(.*?)ENDSEC\s*;', content, re.DOTALL)
        if header_match:
            result["header_section"] = "HEADER;\n" + header_match.group(1).strip() + "\nENDSEC;"
        else:
            result["header_section"] = "（無法找到 HEADER 區段）"

        # 提取 DATA 區段（完整內容）
        data_match = re.search(r'DATA\s*;(.*?)ENDSEC\s*;', content, re.DOTALL)
        if data_match:
            data_content = data_match.group(1).strip()

            # 解析並按 #n= 排序
            # 匹配 STEP 實體格式: #數字=內容;
            entity_pattern = re.compile(r'(#(\d+)\s*=\s*[^;]*;)', re.DOTALL)
            entities = entity_pattern.findall(data_content)

            if entities:
                # 按實體編號排序
                sorted_entities = sorted(entities, key=lambda x: int(x[1]))
                sorted_data = '\n'.join([e[0].strip() for e in sorted_entities])
                result["data_section"] = "DATA;\n" + sorted_data + "\nENDSEC;"
            else:
                # 如果無法解析實體，返回原始內容
                result["data_section"] = "DATA;\n" + data_content + "\nENDSEC;"
        else:
            result["data_section"] = "（無法找到 DATA 區段）"

        return result

    def display_step_content(self):
        """
        顯示 STEP 檔案的完整內容
        包含 HEADER 和 DATA 區段
        """
        raw = self.get_step_raw_content()

        log_print("\n" + "=" * 60)
        log_print("STEP 檔案原始內容")
        log_print("=" * 60)

        if not raw["readable"]:
            log_print(f"\n[無法讀取] 原因: {raw['reason']}")
            if raw.get("file_format"):
                log_print(f"偵測到的格式: {raw['file_format']}")
            log_print("=" * 60 + "\n")
            return False

        log_print(f"\n檔案格式: {raw['file_format']}")
        log_print(f"編碼: {raw['encoding']}")
        log_print(f"總行數: {raw['total_lines']}")

        log_print("\n--- HEADER 區段 ---")
        log_print(raw["header_section"])

        log_print("\n--- DATA 區段 (預覽) ---")
        log_print(raw["data_section"])

        log_print("\n" + "=" * 60 + "\n")
        return True

    def preview_3d_model(self):
        """
        預覽 3D 模型（使用 EngineeringViewer）
        """
        if not self.model_file or not os.path.exists(self.model_file):
            log_print("[Warning] 無法預覽：未載入 3D 模型檔案", "warning")
            return False

        log_print(f"\n[System] 開啟 3D 模型預覽: {self.model_file}")
        try:
            EngineeringViewer.view_3d_stl(self.model_file)
            return True
        except Exception as e:
            log_print(f"[Error] 3D 預覽失敗: {e}", "error")
            return False

    def export_projections_to_dxf(self, output_dir: str = "output") -> List[str]:
        """
        將 3D 模型投影到 XY/XZ/YZ 平面並輸出 DXF 檔案

        Args:
            output_dir: 輸出目錄

        Returns:
            成功輸出的 DXF 檔案路徑列表
        """
        if not CADQUERY_AVAILABLE:
            log_print("[Error] CadQuery 未安裝，無法進行 3D 投影", "error")
            return []

        if self.cad_model is None:
            log_print("[Error] 未載入 3D 模型，無法進行投影", "error")
            return []

        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log_print(f"[System] 建立輸出目錄: {output_dir}")

        # 取得基本檔名
        if self.model_file:
            base_name = os.path.splitext(os.path.basename(self.model_file))[0]
        else:
            base_name = "model"

        # 定義兩組投影模式
        projections = [
            # Mode 1: 直接投影（不旋轉，直接取對應座標軸）
            {'plane': 'XY', 'suffix': '',     'description': '俯視圖 (Top) - 直接投影'},
            {'plane': 'XZ', 'suffix': '',     'description': '前視圖 (Front) - 直接投影'},
            {'plane': 'YZ', 'suffix': '',     'description': '側視圖 (Right) - 直接投影'},
            # Mode 2: 反向視角投影
            {'plane': 'XY', 'suffix': '_rot', 'description': '俯視圖 (Top) - 反向'},
            {'plane': 'XZ', 'suffix': '_rot', 'description': '前視圖 (Front) - 反向'},
            {'plane': 'YZ', 'suffix': '_rot', 'description': '側視圖 (Right) - 反向'},
        ]

        output_files = []

        log_print("\n" + "=" * 60)
        log_print("開始 3D 模型投影轉換...")
        log_print("=" * 60)

        for proj in projections:
            plane_name = proj['plane']
            suffix = proj['suffix']
            output_path = os.path.join(output_dir, f"{base_name}_{plane_name}{suffix}.dxf")
            log_print(f"\n[{plane_name}{suffix}] 正在生成 {proj['description']}...")

            try:
                # 使用 CadQuery 進行投影
                shape = self.cad_model.val()

                # 建立 DXF 文件
                doc = ezdxf.new()
                msp = doc.modelspace()

                # 獲取所有邊並投影到指定平面
                # 使用 OCP TopExp_Explorer 遍歷複合體中所有 solid 的邊
                from OCP.BRepAdaptor import BRepAdaptor_Curve
                from OCP.TopExp import TopExp_Explorer
                from OCP.TopAbs import TopAbs_EDGE
                from OCP.TopoDS import TopoDS

                shape = self.cad_model.val()
                edge_explorer = TopExp_Explorer(shape.wrapped, TopAbs_EDGE)
                edges = []
                while edge_explorer.More():
                    edges.append(TopoDS.Edge_s(edge_explorer.Current()))
                    edge_explorer.Next()

                edge_count = 0
                for edge_shape in edges:
                    try:
                        curve = BRepAdaptor_Curve(edge_shape)
                        u_start = curve.FirstParameter()
                        u_end = curve.LastParameter()

                        # 取樣點
                        num_points = 20
                        points_2d = []

                        for i in range(num_points + 1):
                            u = u_start + (u_end - u_start) * i / num_points
                            pnt = curve.Value(u)
                            x, y, z = pnt.X(), pnt.Y(), pnt.Z()

                            if suffix == '':
                                # Mode 1: 直接投影（不旋轉）
                                if plane_name == 'XY':
                                    # 俯視圖 (Top): 沿 -Z 方向看 → 取 (x, y)
                                    points_2d.append((x, y))
                                elif plane_name == 'XZ':
                                    # 前視圖 (Front): 沿 -Y 方向看 → 取 (x, z)
                                    points_2d.append((x, z))
                                else:  # YZ
                                    # 側視圖 (Right): 沿 -X 方向看 → 取 (-y, z)
                                    points_2d.append((-y, z))
                            else:
                                # Mode 2: 反向視角投影
                                if plane_name == 'XY':
                                    # 俯視圖（Y 翻轉）: (x, -y)
                                    points_2d.append((x, -y))
                                elif plane_name == 'XZ':
                                    # 前視圖: (x, z)
                                    points_2d.append((x, z))
                                else:  # YZ
                                    # 側視圖（沿 Y 軸看，X 鏡射）: (-x, z)
                                    points_2d.append((-x, z))

                        # 繪製多段線
                        if len(points_2d) >= 2:
                            msp.add_lwpolyline(points_2d)
                            edge_count += 1

                    except Exception as e:
                        # 跳過無法處理的邊
                        continue

                # 儲存 DXF
                doc.saveas(output_path)
                log_print(f"[Success] {proj['description']} 已儲存")
                log_print(f"          路徑: {output_path}")
                log_print(f"          邊數: {edge_count}")
                output_files.append(output_path)

            except Exception as e:
                log_print(f"[Error] {plane_name}{suffix} 投影失敗: {e}", "error")

        log_print("\n" + "=" * 60)
        log_print(f"投影轉換完成！成功生成 {len(output_files)}/6 個視圖")
        log_print("=" * 60 + "\n")

        return output_files

    def generate_assembly_drawing(self, output_dir: str = "output") -> str:
        """
        生成組立施工圖 (Assembly Drawing)
        包含：三視圖、標題欄、零件表 (BOM)、基本資訊

        Args:
            output_dir: 輸出目錄

        Returns:
            輸出的 DXF 檔案路徑，失敗返回 None
        """
        if not CADQUERY_AVAILABLE:
            log_print("[Error] CadQuery 未安裝，無法生成組立圖", "error")
            return None

        if self.cad_model is None:
            log_print("[Error] 未載入 3D 模型，無法生成組立圖", "error")
            return None

        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 取得模型資訊
        info = self.get_model_info()

        # 取得基本檔名
        if self.model_file:
            base_name = os.path.splitext(os.path.basename(self.model_file))[0]
        else:
            base_name = "assembly"

        output_path = os.path.join(output_dir, f"{base_name}_組立圖.dxf")

        log_print("\n" + "=" * 60)
        log_print("開始生成組立施工圖...")
        log_print("=" * 60)

        try:
            # 建立 DXF 文件 (使用 A1 圖紙大小: 841 x 594 mm)
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()

            # 圖紙尺寸 (A1)
            paper_width = 841
            paper_height = 594

            # 定義區域
            margin = 10
            title_block_height = 60
            bom_width = 200
            view_area_width = paper_width - margin * 2 - bom_width
            view_area_height = paper_height - margin * 2 - title_block_height

            # ===== 1. 繪製圖框 =====
            # 外框
            msp.add_line((0, 0), (paper_width, 0))
            msp.add_line((paper_width, 0), (paper_width, paper_height))
            msp.add_line((paper_width, paper_height), (0, paper_height))
            msp.add_line((0, paper_height), (0, 0))

            # 內框
            msp.add_line((margin, margin), (paper_width - margin, margin))
            msp.add_line((paper_width - margin, margin), (paper_width - margin, paper_height - margin))
            msp.add_line((paper_width - margin, paper_height - margin), (margin, paper_height - margin))
            msp.add_line((margin, paper_height - margin), (margin, margin))

            # ===== 2. 繪製標題欄 =====
            title_y = margin
            title_x = margin
            title_width = paper_width - margin * 2

            # 標題欄框線
            msp.add_line((title_x, title_y), (title_x + title_width, title_y))
            msp.add_line((title_x, title_y + title_block_height), (title_x + title_width, title_y + title_block_height))

            # 標題欄分隔線
            col_widths = [200, 150, 150, 150, title_width - 650]
            x_pos = title_x
            for w in col_widths[:-1]:
                x_pos += w
                msp.add_line((x_pos, title_y), (x_pos, title_y + title_block_height))

            # 標題欄水平分隔
            msp.add_line((title_x, title_y + 30), (title_x + title_width, title_y + 30))

            # 標題欄文字
            product_name = info.get('product_name') or base_name
            file_name = info.get('file_name') or 'N/A'
            source_sw = info.get('source_software') or 'N/A'
            units = info.get('units') or 'mm'
            today = datetime.now().strftime("%Y-%m-%d")

            # 添加文字 (使用 add_text)
            text_height = 4
            msp.add_text(f"圖名: {product_name}", dxfattribs={'height': text_height * 1.5}).set_placement(
                (title_x + 10, title_y + 45))
            msp.add_text(f"檔案: {file_name}", dxfattribs={'height': text_height}).set_placement(
                (title_x + 10, title_y + 15))
            msp.add_text(f"來源: {source_sw}", dxfattribs={'height': text_height}).set_placement(
                (title_x + 210, title_y + 15))
            msp.add_text(f"單位: {units}", dxfattribs={'height': text_height}).set_placement(
                (title_x + 360, title_y + 15))
            msp.add_text(f"日期: {today}", dxfattribs={'height': text_height}).set_placement(
                (title_x + 510, title_y + 15))
            msp.add_text("組立施工圖 Assembly Drawing", dxfattribs={'height': text_height * 1.2}).set_placement(
                (title_x + 510, title_y + 45))

            # ===== 3. 繪製 BOM 表 =====
            bom_x = paper_width - margin - bom_width
            bom_y = margin + title_block_height + 10
            bom_height = view_area_height - 10

            # BOM 框線
            msp.add_line((bom_x, bom_y), (bom_x + bom_width, bom_y))
            msp.add_line((bom_x + bom_width, bom_y), (bom_x + bom_width, bom_y + bom_height))
            msp.add_line((bom_x + bom_width, bom_y + bom_height), (bom_x, bom_y + bom_height))
            msp.add_line((bom_x, bom_y + bom_height), (bom_x, bom_y))

            # BOM 標題
            msp.add_text("零件表 (BOM)", dxfattribs={'height': text_height * 1.2}).set_placement(
                (bom_x + 10, bom_y + bom_height - 15))

            # BOM 表頭
            row_height = 12
            header_y = bom_y + bom_height - 30
            msp.add_line((bom_x, header_y), (bom_x + bom_width, header_y))
            msp.add_line((bom_x, header_y - row_height), (bom_x + bom_width, header_y - row_height))

            # 表頭文字
            msp.add_text("項次", dxfattribs={'height': text_height}).set_placement((bom_x + 5, header_y - 9))
            msp.add_text("名稱", dxfattribs={'height': text_height}).set_placement((bom_x + 35, header_y - 9))
            msp.add_text("數量", dxfattribs={'height': text_height}).set_placement((bom_x + 130, header_y - 9))
            msp.add_text("材料", dxfattribs={'height': text_height}).set_placement((bom_x + 165, header_y - 9))

            # 表頭分隔線
            msp.add_line((bom_x + 30, header_y), (bom_x + 30, header_y - row_height))
            msp.add_line((bom_x + 125, header_y), (bom_x + 125, header_y - row_height))
            msp.add_line((bom_x + 160, header_y), (bom_x + 160, header_y - row_height))

            # BOM 內容
            bom_data = info.get('bom', [])
            if not bom_data:
                # 如果沒有 BOM，使用實體資訊
                solid_count = info.get('solid_count', 0)
                if solid_count > 0:
                    for i in range(min(solid_count, 20)):  # 最多顯示 20 項
                        bom_data.append({
                            'item': i + 1,
                            'name': f'Solid_{i + 1}',
                            'quantity': 1,
                            'material': '未指定'
                        })

            for i, item in enumerate(bom_data[:20]):  # 最多 20 項
                item_y = header_y - row_height - (i + 1) * row_height
                if item_y < bom_y + 10:
                    break

                msp.add_line((bom_x, item_y), (bom_x + bom_width, item_y))
                msp.add_text(str(item.get('item', i + 1)), dxfattribs={'height': text_height - 1}).set_placement(
                    (bom_x + 5, item_y + 3))
                name = str(item.get('name', ''))[:15]
                msp.add_text(name, dxfattribs={'height': text_height - 1}).set_placement(
                    (bom_x + 35, item_y + 3))
                msp.add_text(str(item.get('quantity', 1)), dxfattribs={'height': text_height - 1}).set_placement(
                    (bom_x + 130, item_y + 3))
                material = str(item.get('material', ''))[:6]
                msp.add_text(material, dxfattribs={'height': text_height - 1}).set_placement(
                    (bom_x + 165, item_y + 3))

            # ===== 4. 繪製三視圖 =====
            view_x = margin + 10
            view_y = margin + title_block_height + 10
            single_view_width = (view_area_width - bom_width - 30) / 2
            single_view_height = (view_area_height - 30) / 2

            # 獲取模型邊界框
            bbox = info.get('bounding_box', {})
            if bbox:
                model_width = bbox.get('width', 100)
                model_height = bbox.get('height', 100)
                model_depth = bbox.get('depth', 100)
                model_center_x = (bbox.get('x_min', 0) + bbox.get('x_max', 0)) / 2
                model_center_y = (bbox.get('y_min', 0) + bbox.get('y_max', 0)) / 2
                model_center_z = (bbox.get('z_min', 0) + bbox.get('z_max', 0)) / 2
            else:
                model_width = model_height = model_depth = 100
                model_center_x = model_center_y = model_center_z = 0

            # 計算縮放比例
            max_model_dim = max(model_width, model_height, model_depth)
            scale = min(single_view_width, single_view_height) * 0.7 / max_model_dim if max_model_dim > 0 else 1

            # 定義三個視圖的位置
            views = [
                {'name': '前視圖 (Front)', 'plane': 'XZ', 'x': view_x, 'y': view_y + single_view_height + 20,
                 'offset': (model_center_x, model_center_z)},
                {'name': '俯視圖 (Top)', 'plane': 'XY', 'x': view_x, 'y': view_y,
                 'offset': (model_center_x, model_center_y)},
                {'name': '側視圖 (Right)', 'plane': 'YZ', 'x': view_x + single_view_width + 20, 'y': view_y + single_view_height + 20,
                 'offset': (model_center_y, model_center_z)},
            ]

            # 獲取所有邊
            edges = self.cad_model.edges().vals()

            for view in views:
                vx, vy = view['x'], view['y']
                plane = view['plane']
                offset_x, offset_y = view['offset']

                # 繪製視圖框
                msp.add_line((vx, vy), (vx + single_view_width, vy))
                msp.add_line((vx + single_view_width, vy), (vx + single_view_width, vy + single_view_height))
                msp.add_line((vx + single_view_width, vy + single_view_height), (vx, vy + single_view_height))
                msp.add_line((vx, vy + single_view_height), (vx, vy))

                # 視圖標題
                msp.add_text(view['name'], dxfattribs={'height': text_height}).set_placement(
                    (vx + 5, vy + single_view_height - 10))

                # 視圖中心
                center_x = vx + single_view_width / 2
                center_y = vy + single_view_height / 2 - 10

                # 投影邊到視圖
                for edge in edges:
                    try:
                        from OCP.BRepAdaptor import BRepAdaptor_Curve

                        curve = BRepAdaptor_Curve(edge.wrapped)
                        u_start = curve.FirstParameter()
                        u_end = curve.LastParameter()

                        points_2d = []
                        for i in range(21):
                            u = u_start + (u_end - u_start) * i / 20
                            pnt = curve.Value(u)
                            x, y, z = pnt.X(), pnt.Y(), pnt.Z()

                            # 根據投影平面選擇座標並應用縮放（置中 + 第三角投影法）
                            if plane == 'XY':
                                # 俯視圖 (Top): 沿 -Z 看 → (x, y)
                                px = center_x + (x - offset_x) * scale
                                py = center_y + (y - offset_y) * scale
                            elif plane == 'XZ':
                                # 前視圖 (Front): 沿 -Y 看 → (x, z)
                                px = center_x + (x - offset_x) * scale
                                py = center_y + (z - offset_y) * scale
                            else:  # YZ
                                # 側視圖 (Right): 沿 -X 看 → (-y, z)
                                px = center_x + (offset_x - y) * scale
                                py = center_y + (z - offset_y) * scale

                            # 裁剪到視圖範圍
                            px = max(vx + 5, min(vx + single_view_width - 5, px))
                            py = max(vy + 5, min(vy + single_view_height - 15, py))
                            points_2d.append((px, py))

                        if len(points_2d) >= 2:
                            msp.add_lwpolyline(points_2d)

                    except Exception:
                        continue

            # ===== 5. 添加基本資訊框 =====
            info_x = view_x + single_view_width + 20
            info_y = view_y
            info_width = single_view_width
            info_height = single_view_height

            # 資訊框
            msp.add_line((info_x, info_y), (info_x + info_width, info_y))
            msp.add_line((info_x + info_width, info_y), (info_x + info_width, info_y + info_height))
            msp.add_line((info_x + info_width, info_y + info_height), (info_x, info_y + info_height))
            msp.add_line((info_x, info_y + info_height), (info_x, info_y))

            # 資訊標題
            msp.add_text("模型資訊", dxfattribs={'height': text_height * 1.2}).set_placement(
                (info_x + 10, info_y + info_height - 15))

            # 資訊內容
            info_lines = [
                f"檔案: {file_name}",
                f"產品: {product_name}",
                f"來源: {source_sw}",
                f"單位: {units}",
                f"",
                f"尺寸 (寬x高x深):",
                f"  {model_width:.2f} x {model_height:.2f} x {model_depth:.2f}",
                f"",
                f"實體數: {info.get('solid_count', 0)}",
                f"面數: {info.get('face_count', 0)}",
                f"邊數: {info.get('edge_count', 0)}",
            ]

            if info.get('volume'):
                info_lines.append(f"體積: {info['volume']:.2f}")
            if info.get('surface_area'):
                info_lines.append(f"表面積: {info['surface_area']:.2f}")

            for i, line in enumerate(info_lines):
                line_y = info_y + info_height - 30 - i * 10
                if line_y > info_y + 5:
                    msp.add_text(line, dxfattribs={'height': text_height - 0.5}).set_placement(
                        (info_x + 10, line_y))

            # ===== 6. 儲存 DXF =====
            doc.saveas(output_path)

            log_print(f"\n[Success] 組立施工圖已生成!")
            log_print(f"          路徑: {output_path}")
            log_print("=" * 60 + "\n")

            return output_path

        except Exception as e:
            log_print(f"[Error] 生成組立圖失敗: {e}", "error")
            import traceback
            log_print(traceback.format_exc(), "error")
            return None

    def display_info(self):
        """
        顯示 3D 模型的完整資訊
        如果沒有資訊則顯示 "無資訊"
        在控制台輸出簡要資訊
        """
        info = self.get_model_info()

        log_print("\n" + "=" * 50)
        log_print("圖檔資訊 (Model Information)")
        log_print("=" * 50)

        # 檢查是否有資訊可顯示
        if not info["features"] and info["bounding_box"] is None:
            log_print("無資訊")
            log_print("=" * 50 + "\n")
            return False

        # 顯示檔案資訊
        if info["model_file"]:
            log_print(f"\n檔案路徑: {info['model_file']}")
            log_print(f"   檔案名稱: {os.path.basename(info['model_file'])}")
        else:
            log_print("\n檔案: 使用模擬資料 (Mock Data)")

        # 顯示 3D 模型狀態
        if info["has_model"]:
            log_print(f"   3D 模型: 已載入")
        else:
            log_print(f"   3D 模型: 未載入 (使用特徵資料)")

        log_print(f"\n詳細資訊將在獨立視窗中顯示...")
        log_print("=" * 50 + "\n")
        return True

    def show_info_window(self):
        """
        在獨立視窗中顯示完整的圖檔資訊
        包含 BOM、來源軟體、零件、單位等
        如無資訊則顯示「無資訊」
        """
        info = self.get_model_info()

        # 建立獨立視窗
        window = Toplevel()
        window.title("圖檔資訊 (Model Information)")
        window.geometry("700x800")
        window.configure(bg='#f0f0f0')

        # 使視窗置頂
        window.attributes('-topmost', True)
        window.focus_force()

        # 建立主框架
        main_frame = Frame(window, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 標題
        title_label = Label(main_frame, text="圖檔完整資訊", font=('Microsoft JhengHei', 16, 'bold'),
                           bg='#f0f0f0', fg='#333333')
        title_label.pack(pady=(0, 10))

        # 建立文字區域框架
        text_frame = Frame(main_frame, bg='#ffffff', relief=tk.SUNKEN, bd=1)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # 建立捲軸
        scrollbar = Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 建立文字區域
        text_widget = Text(text_frame, wrap=tk.WORD, font=('Consolas', 10),
                          yscrollcommand=scrollbar.set, bg='#ffffff', fg='#333333',
                          padx=10, pady=10)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        # 定義文字標籤樣式
        text_widget.tag_configure('header', font=('Microsoft JhengHei', 12, 'bold'), foreground='#0066cc')
        text_widget.tag_configure('section', font=('Microsoft JhengHei', 11, 'bold'), foreground='#006600')
        text_widget.tag_configure('label', font=('Consolas', 10, 'bold'), foreground='#333333')
        text_widget.tag_configure('value', font=('Consolas', 10), foreground='#666666')
        text_widget.tag_configure('no_info', font=('Microsoft JhengHei', 10, 'italic'), foreground='#999999')
        text_widget.tag_configure('separator', foreground='#cccccc')

        def add_line(text, tag=None):
            """添加一行文字"""
            if tag:
                text_widget.insert(tk.END, text + "\n", tag)
            else:
                text_widget.insert(tk.END, text + "\n")

        def add_field(label, value, indent="    "):
            """添加欄位，如無資訊顯示「無資訊」"""
            text_widget.insert(tk.END, f"{indent}{label}: ", 'label')
            if value is not None and value != "" and value != []:
                text_widget.insert(tk.END, f"{value}\n", 'value')
            else:
                text_widget.insert(tk.END, "無資訊\n", 'no_info')

        def add_separator():
            """添加分隔線"""
            add_line("─" * 60, 'separator')

        # ===== 開始填入資訊 =====

        add_line("=" * 60, 'separator')
        add_line("圖檔資訊報告 (Model Information Report)", 'header')
        add_line("=" * 60, 'separator')
        add_line("")

        # ----- 基本檔案資訊 -----
        add_line("【基本檔案資訊】", 'section')
        add_separator()
        add_field("檔案路徑", info.get("model_file"))
        add_field("檔案名稱", info.get("file_name"))
        add_field("檔案格式", info.get("file_extension"))
        add_field("檔案大小", info.get("file_size"))
        add_field("3D 模型狀態", "已載入" if info.get("has_model") else "未載入 (使用特徵資料)")
        add_line("")

        # ----- 來源軟體與元資料 -----
        add_line("【來源軟體與元資料】", 'section')
        add_separator()
        add_field("來源軟體", info.get("source_software"))
        add_field("作者", info.get("author"))
        add_field("組織", info.get("organization"))
        add_field("建立日期", info.get("creation_date"))
        add_field("產品名稱", info.get("product_name"))
        add_field("版本", info.get("version"))
        add_field("描述", info.get("description"))
        add_line("")

        # ----- 單位資訊 -----
        add_line("【單位資訊】", 'section')
        add_separator()
        add_field("長度單位", info.get("units"))
        add_line("")

        # ----- 幾何統計資訊 -----
        add_line("【幾何統計資訊】", 'section')
        add_separator()
        add_field("實體數量 (Solids)", info.get("solid_count") if info.get("solid_count", 0) > 0 else None)
        add_field("面數量 (Faces)", info.get("face_count") if info.get("face_count", 0) > 0 else None)
        add_field("邊數量 (Edges)", info.get("edge_count") if info.get("edge_count", 0) > 0 else None)
        add_field("頂點數量 (Vertices)", info.get("vertex_count") if info.get("vertex_count", 0) > 0 else None)

        if info.get("volume") is not None:
            add_field("體積", f"{info['volume']:.4f} 立方單位")
        else:
            add_field("體積", None)

        if info.get("surface_area") is not None:
            add_field("表面積", f"{info['surface_area']:.4f} 平方單位")
        else:
            add_field("表面積", None)
        add_line("")

        # ----- 邊界框資訊 -----
        add_line("【邊界框資訊 (Bounding Box)】", 'section')
        add_separator()
        bbox = info.get("bounding_box")
        if bbox:
            add_field("X 範圍", f"{bbox['x_min']:.2f} ~ {bbox['x_max']:.2f}")
            add_field("Y 範圍", f"{bbox['y_min']:.2f} ~ {bbox['y_max']:.2f}")
            add_field("Z 範圍", f"{bbox['z_min']:.2f} ~ {bbox['z_max']:.2f}")
            add_field("寬度 (Width)", f"{bbox['width']:.2f}")
            add_field("高度 (Height)", f"{bbox['height']:.2f}")
            add_field("深度 (Depth)", f"{bbox['depth']:.2f}")
        else:
            add_field("邊界框", None)
        add_line("")

        # ----- 零件列表 -----
        add_line("【零件列表 (Parts)】", 'section')
        add_separator()
        parts = info.get("parts", [])
        if parts:
            for i, part in enumerate(parts, 1):
                add_line(f"    [{i}] ID: {part.get('id')}", 'value')
                add_line(f"        名稱: {part.get('name')}", 'value')
                add_line(f"        類型: {part.get('type')}", 'value')
        else:
            add_line("    無資訊", 'no_info')
        add_line("")

        # ----- BOM (Bill of Materials) -----
        add_line("【BOM 材料清單 (Bill of Materials)】", 'section')
        add_separator()
        bom = info.get("bom", [])
        if bom:
            add_line("    項次 | 名稱                     | 數量 | 材料     | 描述", 'label')
            add_line("    " + "-" * 56, 'separator')
            for item in bom:
                name = item.get('name', '未命名')[:24].ljust(24)
                qty = str(item.get('quantity', 1)).ljust(4)
                material = item.get('material', '未指定')[:8].ljust(8)
                desc = item.get('description', '')[:20]
                add_line(f"    {item.get('item', 1):4d} | {name} | {qty} | {material} | {desc}", 'value')
        else:
            add_line("    無資訊", 'no_info')
        add_line("")

        # ----- 幾何特徵 -----
        add_line("【幾何特徵 (Geometric Features)】", 'section')
        add_separator()
        features = info.get("features", [])
        if features:
            add_line(f"    共 {len(features)} 個特徵", 'value')
            add_line("")
            for i, feat in enumerate(features, 1):
                add_line(f"    [{i}] 特徵 ID: {feat['id']}", 'label')
                add_line(f"        類型: {feat['type']}", 'value')
                add_line(f"        描述: {feat['description']}", 'value')
                params_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in feat['params'].items()])
                add_line(f"        參數: {params_str}", 'value')
                if i < len(features):
                    add_line("")
        else:
            add_line("    無資訊", 'no_info')
        add_line("")

        add_line("=" * 60, 'separator')
        add_line("報告結束", 'header')
        add_line("=" * 60, 'separator')

        # 設定文字區域為唯讀
        text_widget.config(state=tk.DISABLED)

        # 關閉按鈕
        close_button = Button(main_frame, text="關閉視窗", font=('Microsoft JhengHei', 10),
                             command=window.destroy, bg='#4a90d9', fg='white',
                             padx=20, pady=5, relief=tk.FLAT)
        close_button.pack(pady=10)

        # 讓視窗保持開啟但不阻擋主程序
        window.update()

        return window

    def save_info_to_file(self, output_dir: str = "output") -> str:
        """
        將圖檔資訊儲存到檔案
        Args:
            output_dir: 輸出目錄，預設為 "output"
        Returns:
            輸出檔案的完整路徑
        """
        info = self.get_model_info()
        step_raw = self.get_step_raw_content()

        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log_print(f"[System] 建立輸出目錄: {output_dir}")

        # 生成輸出檔名
        if info.get("file_name"):
            base_name = os.path.splitext(info["file_name"])[0]
            output_filename = f"{base_name}_info.txt"
        else:
            output_filename = "model_info.txt"

        output_path = os.path.join(output_dir, output_filename)

        # 輔助函數
        def format_value(value):
            """格式化數值，如無資訊返回「無資訊」"""
            if value is not None and value != "" and value != []:
                return str(value)
            return "無資訊"

        # 建立報告內容
        lines = []
        lines.append("=" * 80)
        lines.append("STEP 檔案完整資訊報告 (Model Information Report)")
        lines.append("=" * 80)
        lines.append("")

        # ----- 基本檔案資訊 -----
        lines.append("【基本檔案資訊】")
        lines.append("-" * 80)
        lines.append(f"    檔案路徑: {format_value(info.get('model_file'))}")
        lines.append(f"    檔案名稱: {format_value(info.get('file_name'))}")
        lines.append(f"    檔案格式: {format_value(info.get('file_extension'))}")
        lines.append(f"    檔案類型: {format_value(info.get('file_type'))}")
        lines.append(f"    檔案大小: {format_value(info.get('file_size'))}")
        lines.append(f"    3D 模型狀態: {'已載入' if info.get('has_model') else '未載入'}")
        lines.append("")

        # ----- 檔案讀取狀態 -----
        lines.append("【檔案讀取狀態】")
        lines.append("-" * 80)
        if step_raw["readable"]:
            lines.append(f"    讀取狀態: 成功")
            lines.append(f"    檔案編碼: {step_raw['encoding']}")
            lines.append(f"    檔案格式: {step_raw['file_format']}")
            lines.append(f"    總行數: {step_raw['total_lines']}")
        else:
            lines.append(f"    讀取狀態: 失敗")
            lines.append(f"    [無法讀取原因] {step_raw['reason']}")
            if step_raw.get("file_format"):
                lines.append(f"    偵測到的格式: {step_raw['file_format']}")
        lines.append("")

        # ----- 來源軟體與元資料 -----
        lines.append("【來源軟體與元資料】")
        lines.append("-" * 80)
        lines.append(f"    來源軟體: {format_value(info.get('source_software'))}")
        lines.append(f"    作者: {format_value(info.get('author'))}")
        lines.append(f"    組織: {format_value(info.get('organization'))}")
        lines.append(f"    建立日期: {format_value(info.get('creation_date'))}")
        lines.append(f"    產品名稱: {format_value(info.get('product_name'))}")
        lines.append(f"    版本: {format_value(info.get('version'))}")
        lines.append(f"    描述: {format_value(info.get('description'))}")
        lines.append("")

        # ----- 單位資訊 -----
        lines.append("【單位資訊】")
        lines.append("-" * 80)
        lines.append(f"    長度單位: {format_value(info.get('units'))}")
        lines.append("")

        # ----- 幾何統計資訊 -----
        lines.append("【幾何統計資訊】")
        lines.append("-" * 80)
        solid_count = info.get("solid_count", 0)
        face_count = info.get("face_count", 0)
        edge_count = info.get("edge_count", 0)
        vertex_count = info.get("vertex_count", 0)
        lines.append(f"    實體數量 (Solids): {solid_count if solid_count > 0 else '無資訊'}")
        lines.append(f"    面數量 (Faces): {face_count if face_count > 0 else '無資訊'}")
        lines.append(f"    邊數量 (Edges): {edge_count if edge_count > 0 else '無資訊'}")
        lines.append(f"    頂點數量 (Vertices): {vertex_count if vertex_count > 0 else '無資訊'}")

        if info.get("volume") is not None:
            lines.append(f"    體積: {info['volume']:.4f} 立方單位")
        else:
            lines.append(f"    體積: 無資訊")

        if info.get("surface_area") is not None:
            lines.append(f"    表面積: {info['surface_area']:.4f} 平方單位")
        else:
            lines.append(f"    表面積: 無資訊")

        # STL 特有資訊
        triangle_count = info.get("triangle_count", 0)
        if triangle_count > 0:
            lines.append(f"    三角形數量 (Triangles): {triangle_count:,}")
        stl_solid_name = info.get("stl_solid_name")
        if stl_solid_name:
            lines.append(f"    STL Solid 名稱: {stl_solid_name}")
        lines.append("")

        # ----- 邊界框資訊 -----
        lines.append("【邊界框資訊 (Bounding Box)】")
        lines.append("-" * 80)
        bbox = info.get("bounding_box")
        if bbox:
            lines.append(f"    X 範圍: {bbox['x_min']:.4f} ~ {bbox['x_max']:.4f} (寬度: {bbox['width']:.4f})")
            lines.append(f"    Y 範圍: {bbox['y_min']:.4f} ~ {bbox['y_max']:.4f} (高度: {bbox['height']:.4f})")
            lines.append(f"    Z 範圍: {bbox['z_min']:.4f} ~ {bbox['z_max']:.4f} (深度: {bbox['depth']:.4f})")
        else:
            lines.append(f"    邊界框: 無資訊")
        lines.append("")

        # ----- 分級物料彙整總表 (Grouped BOM Report) -----
        sep_line = "=" * 105
        dash_line = "-" * 105
        lines.append(sep_line)
        product_id = info.get("product_name", info.get("file_name", "未知"))
        lines.append(f"{'分級物料彙整總表 (Grouped BOM Report)':^105}")
        lines.append(sep_line)
        source_file = os.path.basename(info.get("model_file", "")) if info.get("model_file") else "未知"
        lines.append(f"主件號: {product_id} | 來源: {source_file}")
        lines.append("")

        bom_groups = info.get("bom_groups", [])
        if bom_groups:
            for group in bom_groups:
                lines.append(f"[ {group['group_name']} ]")
                lines.append(dash_line)
                lines.append(f"{'零件名稱':<25} | {'尺寸 (L x W x H) mm':<35} | {'數量':<7} | {'原始實體 ID'}")
                lines.append(dash_line)
                for part in group['items']:
                    name = str(part.get('name', ''))[:25].ljust(25)
                    dimension = str(part.get('dimension', '-'))[:35].ljust(35)
                    qty = str(part.get('quantity', 1))[:7].ljust(7)
                    entity_id = str(part.get('entity_id', '-'))
                    lines.append(f"{name}| {dimension}| {qty}| {entity_id}")
                lines.append("")
        else:
            # 沒有分組時，以平面列表顯示
            parts = info.get("parts", [])
            if parts:
                lines.append(dash_line)
                lines.append(f"{'零件名稱':<25} | {'尺寸 (L x W x H) mm':<35} | {'數量':<7} | {'原始實體 ID'}")
                lines.append(dash_line)
                for part in parts:
                    name = str(part.get('name', ''))[:25].ljust(25)
                    dimension = str(part.get('dimension', '-'))[:35].ljust(35)
                    qty = str(part.get('quantity', 1))[:7].ljust(7)
                    entity_id = str(part.get('entity_id', '-'))
                    lines.append(f"{name}| {dimension}| {qty}| {entity_id}")
            else:
                lines.append("    無資訊")
        lines.append(sep_line)
        lines.append("")

        # ----- 幾何特徵 (完整顯示) -----
        lines.append("【幾何特徵 (Geometric Features)】")
        lines.append("-" * 80)
        features = info.get("features", [])
        if features:
            lines.append(f"    共 {len(features)} 個特徵 (完整列表)")
            lines.append("")
            # 完整顯示所有特徵
            for i, feat in enumerate(features, 1):
                lines.append(f"    [{i}] 特徵 ID: {feat['id']}")
                lines.append(f"        類型: {feat['type']}")
                lines.append(f"        描述: {feat['description']}")
                params_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in feat['params'].items()])
                lines.append(f"        參數: {params_str}")
                if i < len(features):
                    lines.append("")
        else:
            lines.append("    無資訊 (無法提取幾何特徵)")
        lines.append("")

        # ----- STEP 檔案原始內容 -----
        lines.append("=" * 80)
        lines.append("STEP 檔案原始內容")
        lines.append("=" * 80)
        lines.append("")

        if step_raw["readable"]:
            # HEADER 區段
            lines.append("【HEADER 區段】")
            lines.append("-" * 80)
            if step_raw.get("header_section"):
                lines.append(step_raw["header_section"])
            else:
                lines.append("（無法取得 HEADER 區段）")
            lines.append("")

            # DATA 區段（完整）
            lines.append("【DATA 區段】")
            lines.append("-" * 80)
            if step_raw.get("data_section"):
                lines.append(step_raw["data_section"])
            else:
                lines.append("（無法取得 DATA 區段）")
        else:
            lines.append(f"[無法讀取檔案內容]")
            lines.append(f"原因: {step_raw['reason']}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("報告結束")
        lines.append("=" * 80)

        # 寫入檔案
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            log_print(f"[System] 圖檔資訊已儲存至: {output_path}")
            return output_path
        except Exception as e:
            log_print(f"[System] 儲存圖檔資訊失敗: {e}", "error")
            return None

    def project_to_2d(self, base_output_name="output_tube"):
        """
        轉換成D工程圖（三視圖，含彎管資料表）
        """
        if not self.filepath:
            print("[Error] 未選擇檔案")
            return False
            
        if not self.cadquery_available or not self.stl_available:
            print("[Error] 缺少必要套件")
            return False

        output_dir = os.path.dirname(base_output_name) or "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        projections = {
            'xy': '俯視圖 (Top View)',
            'xz': '前視圖 (Front View)', 
            'yz': '側視圖 (Right View)'
        }
        
        success_files = []
        
        print("\n" + "="*60)
        log_with_time("[System] 開始彎管工程圖轉換...")
        print("="*60)
        
        for plane, description in projections.items():
            output_path = f"{base_output_name}_{plane.upper()}.dxf"
            abs_output_path = os.path.abspath(output_path)
            
            print(f"\n{'='*50}")
            print(f"[{plane.upper()}] 正在生成 {description}...")
            print(f"{'='*50}")
            log_with_time(f"[{plane.upper()}] 目標: {abs_output_path}")
            
            p = multiprocessing.Process(
                target=export_tube_worker_process,
                args=(self.filepath, abs_output_path, plane)
            )
            p.start()
            
            timeout = 180
            start_time = time.time()
            
            while True:
                elapsed = time.time() - start_time
                
                if p.is_alive():
                    if elapsed > timeout:
                        log_with_time(f"[Warning] {plane.upper()} 轉檔超時，強制終止...")
                        p.terminate()
                        p.join()
                        break
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    time.sleep(1)
                else:
                    break
            
            print("")
            
            exit_code = p.exitcode
            
            if exit_code == 0 and os.path.exists(abs_output_path):
                log_with_time(f"[Success] {description} 轉檔成功！")
                success_files.append((plane, abs_output_path, description))
            else:
                log_with_time(f"[Error] {description} 轉檔失敗")
        
        print("\n" + "="*60)
        log_with_time(f"[Summary] 轉換完成！成功生成 {len(success_files)}/3 個視圖")
        print("="*60)
        
        for plane, path, desc in success_files:
            file_size = os.path.getsize(path) / 1024
            print(f"  ✓ [{plane.upper()}] {desc}")
            print(f"    → {path}")
            print(f"    → 檔案大小: {file_size:.1f} KB")
        
        if success_files:
            print("\n[System] 開啟預覽視窗...")
            for plane, path, desc in success_files:
                try:
                    print(f"\n正在顯示: {desc}")
                    if VIEWER_AVAILABLE:
                        EngineeringViewer.view_2d_dxf(path, fast_mode=False)
                    else:
                        print(f"[Info] DXF 已生成: {path}")
                        print("[Info] 請使用 AutoCAD 或其他 DXF 檢視器開啟")
                except Exception as e:
                    print(f"[Warning] 無法顯示 {desc}: {e}")
            
            return True
        
        return False

# ==========================================
# 2. AI 語意解析層 (The "Brain")
# ==========================================

class AIIntentParser:
    """
    負責將自然語言轉換為 CAD 操作指令
    實際應用中，這裡會呼叫 OpenAI API
    """
    def parse_instruction(self, user_prompt: str, context_features: List[GeometricFeature]) -> dict:
        log_print(f"[AI Agent] Analyzing prompt: '{user_prompt}'")
        
        # --- 模擬 LLM 的推理過程 ---
        # 規則：如果 prompt 包含 "大" 和 "孔"，且有 "倍"，則解析為縮放操作
        
        target_feature = None
        # 簡單的關鍵字匹配 (真實場景會用 Embedding 向量搜尋)
        if "孔" in user_prompt or "hole" in user_prompt.lower():
            target_feature = next((f for f in context_features if f.type == 'circle'), None)
            
        if not target_feature:
            return {"error": "No matching feature found"}

        action = "set"
        value = 0.0
        param = "radius"

        if "倍" in user_prompt: # "兩倍大"
            import re
            numbers = re.findall(r'\d+', user_prompt) # 提取數字
            factor = float(numbers[0]) if numbers else 2.0
            action = "multiply"
            value = factor
        elif "mm" in user_prompt: # "改為 20mm"
            # (省略正則提取邏輯，簡化處理)
            value = 20.0
            action = "set"

        return {
            "target_id": target_feature.id,
            "parameter": param,
            "operation": action,
            "value": value
        }

# ==========================================
# 3. 系統協調器 (Orchestrator)
# ==========================================

class AutoDraftingSystem:
    def __init__(self, model_file: Optional[str] = None):
        """
        初始化自動繪圖系統
        Args:
            model_file: 可選的 3D 模型檔案路徑 (.step, .stp)
        """
        self.cad = MockCADEngine(model_file)
        self.ai = AIIntentParser()

    def display_model_info(self) -> bool:
        """
        顯示載入的模型資訊
        返回 True 表示有資訊顯示，False 表示無資訊
        """
        return self.cad.display_info()

    def show_info_window(self):
        """
        開啟獨立視窗顯示完整的圖檔資訊
        包含 BOM、來源軟體、零件、單位等
        """
        return self.cad.show_info_window()

    def save_info_to_file(self, output_dir: str = "output") -> str:
        """
        將圖檔資訊儲存到檔案
        Args:
            output_dir: 輸出目錄，預設為 "output"
        Returns:
            輸出檔案的完整路徑
        """
        return self.cad.save_info_to_file(output_dir)

    def process_request(self, user_prompt: str, output_filename="output.dxf"):
        # 1. AI 解析
        instruction = self.ai.parse_instruction(user_prompt, self.cad.features)
        
        if "error" in instruction:
            log_print(f"Error: {instruction['error']}", "error")
            return

        # 2. 執行修改
        log_print(f"Executing: {instruction}")
        self.cad.modify_feature(
            instruction["target_id"], 
            instruction["parameter"], 
            instruction["value"], 
            instruction["operation"]
        )

        # 3. 生成 2D 視圖
        projection_data = self.cad.project_to_2d()

        # 4. 輸出 DXF 圖檔
        self._export_dxf(projection_data, output_filename)
        
        # 5. 自動開啟生成的檔案
        self._open_file(output_filename)

    def _export_dxf(self, features: List[GeometricFeature], filename: str):
        """使用 ezdxf 產生真實的 CAD 檔案"""
        doc = ezdxf.new()
        msp = doc.modelspace()

        log_print(f"[DXF Gen] Drawing {len(features)} entities...")
        
        for f in features:
            if f.type == 'rect':
                # 畫矩形 (由四條線組成)
                # 使用絕對座標，與圓形保持一致
                w, h = f.params['w'], f.params['h']
                x, y = f.params.get('x', 0), f.params.get('y', 0)
                # 計算矩形的四個角點（相對於中心點）
                points = [
                    (x - w/2, y - h/2), 
                    (x + w/2, y - h/2), 
                    (x + w/2, y + h/2), 
                    (x - w/2, y + h/2), 
                    (x - w/2, y - h/2)  # 閉合
                ]
                msp.add_lwpolyline(points)
            elif f.type == 'circle':
                # 畫圓
                msp.add_circle((f.params['x'], f.params['y']), f.params['radius'])
        
        # 標註邏輯 (自動標註尺寸)
        # 這裡簡單添加一個半徑標註
        circle = next((f for f in features if f.type == 'circle'), None)
        if circle:
            r = circle.params['radius']
            msp.add_text(f"R{r:.1f}", height=2.5).set_placement((r, r))

        doc.saveas(filename)
        log_print(f"[Success] Saved drawing to {filename}")
    
    def _open_file(self, filename: str):
        """
        自動開啟生成的檔案
        使用 EngineeringViewer 來檢視 DXF 檔案
        其他格式使用系統預設程式開啟
        """
        if not os.path.exists(filename):
            log_print(f"[Warning] File not found: {filename}", "warning")
            return
        
        file_ext = os.path.splitext(filename)[1].lower()
        
        try:
            # 如果是 DXF 檔案，使用 EngineeringViewer 開啟
            if file_ext == '.dxf':
                log_print(f"[System] Opening DXF file with EngineeringViewer: {filename}")
                EngineeringViewer.view_2d_dxf(filename)
            # 如果是 STL 檔案，使用 3D 檢視器
            elif file_ext == '.stl':
                log_print(f"[System] Opening STL file with EngineeringViewer: {filename}")
                EngineeringViewer.view_3d_stl(filename)
            # 其他格式使用系統預設程式開啟
            else:
                system = platform.system()
                if system == "Windows":
                    os.startfile(filename)
                elif system == "Darwin":  # macOS
                    subprocess.run(["open", filename])
                else:  # Linux
                    subprocess.run(["xdg-open", filename])
                log_print(f"[System] Opened file with system default: {filename}")
        except Exception as e:
            log_print(f"[Warning] Could not open file: {e}", "warning")

# ==========================================
# 4. 輔助函數：檔案選擇對話框
# ==========================================

def select_3d_file() -> Optional[str]:
    """
    開啟檔案選擇對話框，讓使用者選擇 3D 模型檔案
    返回選取的檔案路徑，如果取消則返回 None
    """
    # 隱藏主視窗（只顯示對話框）
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # 開啟檔案選擇對話框
    file_path = filedialog.askopenfilename(
        title="選擇 3D 模型檔案",
        filetypes=[
            ("STEP 檔案", "*.step *.stp"),
            ("STL 檔案", "*.stl"),
            ("所有檔案", "*.*")
        ],
        initialdir=os.getcwd()
    )
    
    root.destroy()
    
    if file_path:
        return file_path
    return None

# ==========================================
# 5. 執行範例
# ==========================================
if __name__ == "__main__":
    # 記錄程式開始時間
    start_time = datetime.now()
    log_print("=" * 50)
    log_print("自動繪圖系統 (Auto Drafter System)")
    log_print(f"程式啟動時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 50)

    # 建立主視窗（隱藏）供後續對話框使用
    root = Tk()
    root.withdraw()

    # 讓使用者選擇 3D 模型檔案
    log_print("請選擇 3D 模型檔案...")
    model_file = select_3d_file()

    if model_file:
        log_print(f"[System] 已選擇檔案: {model_file}")
    else:
        log_print("[System] 未選擇檔案，將使用模擬資料")

    # 初始化系統
    system = AutoDraftingSystem(model_file=model_file)

    # =============================================
    # 步驟 1: 完整顯示 STEP 檔案原始內容
    # =============================================
    log_print("步驟 1: 顯示 STEP 檔案原始內容")

    # 取得 STEP 原始內容
    step_raw = system.cad.get_step_raw_content()

    # 建立資訊視窗顯示完整內容
    info = system.cad.get_model_info()

    info_window = Toplevel(root)
    info_window.title("STEP 檔案完整資訊")
    info_window.geometry("1000x800")
    info_window.configure(bg='#f0f0f0')
    info_window.attributes('-topmost', True)
    info_window.focus_force()

    # 建立 Notebook（分頁）
    from tkinter import ttk
    notebook = ttk.Notebook(info_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # ===== 分頁 1: 基本資訊 =====
    tab1 = Frame(notebook, bg='#ffffff')
    notebook.add(tab1, text="基本資訊")

    tab1_scroll = Scrollbar(tab1)
    tab1_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tab1_text = Text(tab1, wrap=tk.WORD, font=('Consolas', 10),
                    yscrollcommand=tab1_scroll.set, bg='#ffffff', fg='#333333', padx=10, pady=10)
    tab1_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    tab1_scroll.config(command=tab1_text.yview)

    def format_value(value):
        if value is not None and value != "" and value != []:
            return str(value)
        return "無資訊"

    tab1_content = []
    tab1_content.append("=" * 70)
    tab1_content.append("基本檔案資訊")
    tab1_content.append("=" * 70)
    tab1_content.append("")
    tab1_content.append(f"檔案路徑: {format_value(info.get('model_file'))}")
    tab1_content.append(f"檔案名稱: {format_value(info.get('file_name'))}")
    tab1_content.append(f"檔案格式: {format_value(info.get('file_extension'))}")
    tab1_content.append(f"檔案類型: {format_value(info.get('file_type'))}")
    tab1_content.append(f"檔案大小: {format_value(info.get('file_size'))}")
    tab1_content.append(f"3D 模型狀態: {'已載入' if info.get('has_model') else '未載入'}")
    tab1_content.append("")

    tab1_content.append("=" * 70)
    tab1_content.append("檔案讀取狀態")
    tab1_content.append("=" * 70)
    tab1_content.append("")
    if step_raw["readable"]:
        tab1_content.append(f"讀取狀態: 成功")
        tab1_content.append(f"檔案編碼: {step_raw['encoding']}")
        tab1_content.append(f"檔案格式: {step_raw['file_format']}")
        tab1_content.append(f"總行數: {step_raw['total_lines']}")
    else:
        tab1_content.append(f"讀取狀態: 失敗")
        tab1_content.append(f"[無法讀取原因] {step_raw['reason']}")
        if step_raw.get("file_format"):
            tab1_content.append(f"偵測到的格式: {step_raw['file_format']}")
    tab1_content.append("")

    tab1_content.append("=" * 70)
    tab1_content.append("來源軟體與元資料")
    tab1_content.append("=" * 70)
    tab1_content.append("")
    tab1_content.append(f"來源軟體: {format_value(info.get('source_software'))}")
    tab1_content.append(f"作者: {format_value(info.get('author'))}")
    tab1_content.append(f"組織: {format_value(info.get('organization'))}")
    tab1_content.append(f"建立日期: {format_value(info.get('creation_date'))}")
    tab1_content.append(f"產品名稱: {format_value(info.get('product_name'))}")
    tab1_content.append(f"描述: {format_value(info.get('description'))}")
    tab1_content.append(f"長度單位: {format_value(info.get('units'))}")
    tab1_content.append("")

    tab1_content.append("=" * 70)
    tab1_content.append("幾何統計資訊")
    tab1_content.append("=" * 70)
    tab1_content.append("")
    tab1_content.append(f"實體數量 (Solids): {info.get('solid_count', 0) or '無資訊'}")
    tab1_content.append(f"面數量 (Faces): {info.get('face_count', 0) or '無資訊'}")
    tab1_content.append(f"邊數量 (Edges): {info.get('edge_count', 0) or '無資訊'}")
    tab1_content.append(f"頂點數量 (Vertices): {info.get('vertex_count', 0) or '無資訊'}")
    if info.get("volume"):
        tab1_content.append(f"體積: {info['volume']:.4f} 立方單位")
    else:
        tab1_content.append(f"體積: 無資訊")
    if info.get("surface_area"):
        tab1_content.append(f"表面積: {info['surface_area']:.4f} 平方單位")
    else:
        tab1_content.append(f"表面積: 無資訊")
    tab1_content.append("")

    tab1_content.append("=" * 70)
    tab1_content.append("邊界框資訊 (Bounding Box)")
    tab1_content.append("=" * 70)
    tab1_content.append("")
    bbox = info.get("bounding_box")
    if bbox:
        tab1_content.append(f"X 範圍: {bbox['x_min']:.4f} ~ {bbox['x_max']:.4f} (寬度: {bbox['width']:.4f})")
        tab1_content.append(f"Y 範圍: {bbox['y_min']:.4f} ~ {bbox['y_max']:.4f} (高度: {bbox['height']:.4f})")
        tab1_content.append(f"Z 範圍: {bbox['z_min']:.4f} ~ {bbox['z_max']:.4f} (深度: {bbox['depth']:.4f})")
    else:
        tab1_content.append("無資訊")

    tab1_text.insert(tk.END, '\n'.join(tab1_content))
    tab1_text.config(state=tk.DISABLED)

    # ===== 分頁 2: 零件列表 / BOM =====
    tab2 = Frame(notebook, bg='#ffffff')
    notebook.add(tab2, text="零件列表 / BOM")

    tab2_scroll = Scrollbar(tab2)
    tab2_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tab2_text = Text(tab2, wrap=tk.WORD, font=('Consolas', 10),
                    yscrollcommand=tab2_scroll.set, bg='#ffffff', fg='#333333', padx=10, pady=10)
    tab2_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    tab2_scroll.config(command=tab2_text.yview)

    tab2_content = []
    sep_line = "=" * 105
    dash_line = "-" * 105
    tab2_content.append(sep_line)
    product_id = info.get("product_name", info.get("file_name", "未知"))
    source_file = os.path.basename(info.get("model_file", "")) if info.get("model_file") else "未知"
    tab2_content.append(f"{'分級物料彙整總表 (Grouped BOM Report)':^105}")
    tab2_content.append(sep_line)
    tab2_content.append(f"主件號: {product_id} | 來源: {source_file}")
    tab2_content.append("")

    bom_groups = info.get("bom_groups", [])
    if bom_groups:
        for group in bom_groups:
            tab2_content.append(f"[ {group['group_name']} ]")
            tab2_content.append(dash_line)
            tab2_content.append(f"{'零件名稱':<25} | {'尺寸 (L x W x H) mm':<35} | {'數量':<7} | {'原始實體 ID'}")
            tab2_content.append(dash_line)
            for part in group['items']:
                name = str(part.get('name', ''))[:25].ljust(25)
                dimension = str(part.get('dimension', '-'))[:35].ljust(35)
                qty = str(part.get('quantity', 1))[:7].ljust(7)
                entity_id = str(part.get('entity_id', '-'))
                tab2_content.append(f"{name}| {dimension}| {qty}| {entity_id}")
            tab2_content.append("")
    else:
        parts = info.get("parts", [])
        if parts:
            tab2_content.append(dash_line)
            tab2_content.append(f"{'零件名稱':<25} | {'尺寸 (L x W x H) mm':<35} | {'數量':<7} | {'原始實體 ID'}")
            tab2_content.append(dash_line)
            for part in parts:
                name = str(part.get('name', ''))[:25].ljust(25)
                dimension = str(part.get('dimension', '-'))[:35].ljust(35)
                qty = str(part.get('quantity', 1))[:7].ljust(7)
                entity_id = str(part.get('entity_id', '-'))
                tab2_content.append(f"{name}| {dimension}| {qty}| {entity_id}")
        else:
            tab2_content.append("無資訊")
    tab2_content.append(sep_line)

    tab2_text.insert(tk.END, '\n'.join(tab2_content))
    tab2_text.config(state=tk.DISABLED)

    # ===== 分頁 3: HEADER 區段 =====
    tab3 = Frame(notebook, bg='#ffffff')
    notebook.add(tab3, text="HEADER 區段")

    tab3_frame = Frame(tab3, bg='#ffffff')
    tab3_frame.pack(fill=tk.BOTH, expand=True)

    tab3_scroll_y = Scrollbar(tab3_frame)
    tab3_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    tab3_scroll_x = Scrollbar(tab3_frame, orient=tk.HORIZONTAL)
    tab3_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    tab3_text = Text(tab3_frame, wrap=tk.NONE, font=('Consolas', 9),
                    yscrollcommand=tab3_scroll_y.set, xscrollcommand=tab3_scroll_x.set,
                    bg='#ffffff', fg='#333333', padx=10, pady=10)
    tab3_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    tab3_scroll_y.config(command=tab3_text.yview)
    tab3_scroll_x.config(command=tab3_text.xview)

    if step_raw["readable"] and step_raw.get("header_section"):
        tab3_text.insert(tk.END, step_raw["header_section"])
    else:
        if not step_raw["readable"]:
            tab3_text.insert(tk.END, f"[無法讀取檔案內容]\n原因: {step_raw['reason']}")
        else:
            tab3_text.insert(tk.END, "（無法取得 HEADER 區段）")
    tab3_text.config(state=tk.DISABLED)

    # ===== 分頁 4: DATA 區段 (完整) =====
    tab4 = Frame(notebook, bg='#ffffff')
    notebook.add(tab4, text="DATA 區段")

    tab4_frame = Frame(tab4, bg='#ffffff')
    tab4_frame.pack(fill=tk.BOTH, expand=True)

    tab4_scroll_y = Scrollbar(tab4_frame)
    tab4_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    tab4_scroll_x = Scrollbar(tab4_frame, orient=tk.HORIZONTAL)
    tab4_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    tab4_text = Text(tab4_frame, wrap=tk.NONE, font=('Consolas', 9),
                    yscrollcommand=tab4_scroll_y.set, xscrollcommand=tab4_scroll_x.set,
                    bg='#ffffff', fg='#333333', padx=10, pady=10)
    tab4_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    tab4_scroll_y.config(command=tab4_text.yview)
    tab4_scroll_x.config(command=tab4_text.xview)

    if step_raw["readable"]:
        data_content = step_raw.get("data_section")
        if data_content and data_content != "（無法找到 DATA 區段）":
            # 顯示 DATA 區段資訊
            data_lines = data_content.count('\n') + 1
            tab4_text.insert(tk.END, f"=== DATA 區段 ===\n")
            tab4_text.insert(tk.END, f"總字元數: {len(data_content):,}\n")
            tab4_text.insert(tk.END, f"總行數: {data_lines:,}\n")
            tab4_text.insert(tk.END, "=" * 50 + "\n\n")
            # 插入完整 DATA 區段內容
            tab4_text.insert(tk.END, data_content)
        else:
            # 調試信息：顯示檔案內容的一部分來幫助診斷
            tab4_text.insert(tk.END, "（無法取得 DATA 區段）\n\n")
            tab4_text.insert(tk.END, "=== 調試信息 ===\n")
            tab4_text.insert(tk.END, f"檔案總行數: {step_raw.get('total_lines', 0)}\n")
            if step_raw.get("content"):
                content_preview = step_raw["content"][:2000]
                tab4_text.insert(tk.END, f"檔案開頭 2000 字元:\n{content_preview}\n")
    else:
        tab4_text.insert(tk.END, f"[無法讀取檔案內容]\n原因: {step_raw['reason']}")
    tab4_text.config(state=tk.DISABLED)

    # 默認選擇 DATA 區段分頁（索引 3）
    notebook.select(3)

    # 底部按鈕框架
    button_frame = Frame(info_window, bg='#f0f0f0')
    button_frame.pack(fill=tk.X, padx=10, pady=10)

    # 提示標籤
    hint_label = Label(button_frame, text="請點擊上方分頁標籤切換：基本資訊 / 零件列表 / HEADER 區段 / DATA 區段",
                      font=('Microsoft JhengHei', 9), bg='#f0f0f0', fg='#666666')
    hint_label.pack(pady=(0, 5))

    close_button = Button(button_frame, text="關閉視窗，繼續預覽 3D 模型", font=('Microsoft JhengHei', 10),
                         command=info_window.destroy, bg='#4a90d9', fg='white',
                         padx=20, pady=5)
    close_button.pack()

    # 等待視窗關閉
    log_print("[System] 顯示 STEP 檔案資訊視窗...")
    root.wait_window(info_window)

    # 將圖檔資訊儲存到 output 目錄
    info_file = system.save_info_to_file("output")

    # =============================================
    # 步驟 2: 預覽 3D 模型
    # =============================================
    log_print("步驟 2: 預覽 3D 模型")

    if model_file and system.cad.cad_model is not None:
        log_print("[System] 開啟 3D 模型預覽視窗...")
        system.cad.preview_3d_model()
    else:
        log_print("[System] 無 3D 模型可預覽（未載入或載入失敗）", "warning")

    # =============================================
    # 步驟 3: 將 3D 模型轉換為 XY/XZ/YZ DXF 檔
    # =============================================
    log_print("步驟 3: 將 3D 模型轉換為三視圖 DXF")

    if model_file and system.cad.cad_model is not None:
        dxf_files = system.cad.export_projections_to_dxf("output")

        if dxf_files:
            log_print("[System] DXF 檔案已生成，開啟預覽...")

            # 預覽每個 DXF 檔案
            for dxf_path in dxf_files:
                if os.path.exists(dxf_path):
                    log_print(f"正在預覽: {os.path.basename(dxf_path)}")
                    try:
                        EngineeringViewer.view_2d_dxf(dxf_path, fast_mode=True)
                    except Exception as e:
                        log_print(f"[Warning] 無法預覽 {dxf_path}: {e}", "warning")
        else:
            log_print("[Warning] 未能生成任何 DXF 檔案", "warning")
    else:
        log_print("[System] 無 3D 模型可轉換（未載入或載入失敗）", "warning")

    # 記錄程式結束時間
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    log_print("處理完成！")
    log_print(f"程式結束時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"總執行時間: {elapsed_time}")

    # 清理
    root.destroy()