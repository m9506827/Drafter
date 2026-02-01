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
        self._solid_shapes = {}  # 儲存 OCC solid shapes: feature_id -> TopoDS_Shape
        self._pipe_centerlines = []  # 管路中心線資料
        self._part_classifications = []  # 零件分類資料
        self._angles = []  # 角度計算結果
        self._cutting_list = {}  # 取料明細

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

        self.model_file = filename

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
                                    fid = f"F{feature_id:02d}"
                                    features.append(GeometricFeature(
                                        fid,
                                        "solid",
                                        {'volume': volume, 'x': cx, 'y': cy, 'z': cz,
                                         'bbox_l': bbox_l, 'bbox_w': bbox_w, 'bbox_h': bbox_h},
                                        f"solid_{len(seen_solids)}"
                                    ))
                                    self._solid_shapes[fid] = moved_solid
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
                                fid = f"F{feature_id:02d}"
                                features.append(GeometricFeature(
                                    fid,
                                    "solid",
                                    {'volume': volume, 'x': cx, 'y': cy, 'z': cz,
                                     'bbox_l': bbox_l, 'bbox_w': bbox_w, 'bbox_h': bbox_h},
                                    f"solid_{len(seen_solids)}"
                                ))
                                self._solid_shapes[fid] = moved_solid
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

            # ===== 提取圓形特徵（使用 TopExp_Explorer 直接遍歷所有邊緣）=====
            # TopExp_Explorer 會正確累積位置變換，BRepAdaptor_Curve 直接回傳全域座標
            seen_circles = set()

            try:
                from OCP.TopoDS import TopoDS
                edge_explorer = TopExp_Explorer(root_shape, TopAbs_EDGE)
                while edge_explorer.More():
                    try:
                        edge = TopoDS.Edge_s(edge_explorer.Current())
                        curve = BRepAdaptor_Curve(edge)

                        if curve.GetType() == GeomAbs_Circle:
                            circle = curve.Circle()
                            center = circle.Location()
                            cx = center.X()
                            cy = center.Y()
                            cz = center.Z()
                            radius = circle.Radius()

                            # 去重（相同位置與半徑的圓只保留一個）
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
                    edge_explorer.Next()

            except Exception as e:
                log_print(f"[CAD Kernel] Warning: Could not extract circles: {e}", "warning")

            log_print(f"[CAD Kernel] 提取了 {len(seen_circles)} 個圓形特徵（全域座標）")

            # ===== 進階分析管線 =====
            # 暫存 features 供分析方法使用
            self.features = features
            try:
                self._pipe_centerlines = self._extract_pipe_centerlines()
                log_print(f"[CAD Kernel] 提取了 {len(self._pipe_centerlines)} 個管路中心線")
                self._part_classifications = self._classify_parts(features)
                log_print(f"[CAD Kernel] 分類了 {len(self._part_classifications)} 個零件")
                self._angles = self._calculate_angles()
                log_print(f"[CAD Kernel] 計算了 {len(self._angles)} 個角度")
                self._cutting_list = self._generate_cutting_list()
                log_print(f"[CAD Kernel] 已生成取料明細")
            except Exception as e:
                log_print(f"[CAD Kernel] Warning: 進階分析管線錯誤: {e}", "warning")

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

    # ==========================================
    # 進階分析元件 (Advanced Analysis Components)
    # ==========================================

    def _analyze_bspline_pipe(self, solid_shape, feature_id, params):
        """
        對具有 BSpline 表面的實體進行管路中心線分析。
        透過取樣表面路徑來提取管徑、中心線長度、直/彎管判斷。
        """
        try:
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.TopoDS import TopoDS
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            from OCP.GeomAbs import GeomAbs_BSplineSurface
            from OCP.BRepGProp import BRepGProp
            from OCP.GProp import GProp_GProps
        except ImportError:
            return None

        # 找出大面積 BSpline 表面（排除端蓋）
        bspline_faces = []
        fexp = TopExp_Explorer(solid_shape, TopAbs_FACE)
        while fexp.More():
            try:
                face = TopoDS.Face_s(fexp.Current())
                surf = BRepAdaptor_Surface(face)
                if surf.GetType() == GeomAbs_BSplineSurface:
                    props = GProp_GProps()
                    BRepGProp.SurfaceProperties_s(face, props)
                    area = props.Mass()
                    if area > 500:
                        bspline_faces.append((face, area))
            except Exception:
                pass
            fexp.Next()

        if len(bspline_faces) < 2:
            return None

        bspline_faces.sort(key=lambda x: x[1])

        # 從最大 BSpline 表面的 V 方向取得管徑
        largest_face = bspline_faces[-1][0]
        surf = BRepAdaptor_Surface(largest_face)
        u1, u2 = surf.FirstUParameter(), surf.LastUParameter()
        v1, v2 = surf.FirstVParameter(), surf.LastVParameter()
        u_mid = (u1 + u2) / 2
        pt_v1 = surf.Value(u_mid, v1)
        pt_v2 = surf.Value(u_mid, v2)
        pipe_diameter = math.sqrt(
            (pt_v2.X() - pt_v1.X()) ** 2 +
            (pt_v2.Y() - pt_v1.Y()) ** 2 +
            (pt_v2.Z() - pt_v1.Z()) ** 2
        )

        if pipe_diameter < 1.0:
            return None

        # 取樣最大表面的路徑作為中心線近似
        # （直管時，表面路徑長度＝中心線長度；彎管時誤差很小）
        n_samples = 100

        def sample_path(face, n):
            s = BRepAdaptor_Surface(face)
            su1, su2 = s.FirstUParameter(), s.LastUParameter()
            sv_mid = (s.FirstVParameter() + s.LastVParameter()) / 2
            pts = []
            for i in range(n + 1):
                t = i / n
                u = su1 + t * (su2 - su1)
                pt = s.Value(u, sv_mid)
                pts.append((pt.X(), pt.Y(), pt.Z()))
            return pts

        centerline = sample_path(largest_face, n_samples)

        # 計算中心線總長
        total_length = 0
        for i in range(len(centerline) - 1):
            dx = centerline[i + 1][0] - centerline[i][0]
            dy = centerline[i + 1][1] - centerline[i][1]
            dz = centerline[i + 1][2] - centerline[i][2]
            total_length += math.sqrt(dx * dx + dy * dy + dz * dz)

        # 判斷直管或彎管
        end_dx = centerline[-1][0] - centerline[0][0]
        end_dy = centerline[-1][1] - centerline[0][1]
        end_dz = centerline[-1][2] - centerline[0][2]
        end_dist = math.sqrt(end_dx ** 2 + end_dy ** 2 + end_dz ** 2)
        straightness = end_dist / total_length if total_length > 0 else 0

        segments = []
        if straightness > 0.99:
            # 直管
            d = (end_dx / end_dist, end_dy / end_dist, end_dz / end_dist) if end_dist > 0 else (0, 0, 0)
            segments.append({
                'type': 'straight',
                'length': round(total_length, 1),
                'direction': d,
            })
        else:
            # 彎管 — 找出曲率所在的軸（起點≈終點的軸），投影到含該軸的平面
            start_p = centerline[0]
            end_p = centerline[-1]
            deltas = [
                (abs(end_p[0] - start_p[0]), 0),
                (abs(end_p[1] - start_p[1]), 1),
                (abs(end_p[2] - start_p[2]), 2),
            ]
            deltas.sort(key=lambda x: x[0])
            # 曲率軸：起終點值最接近的軸（圓弧往返的方向）
            curve_axis = deltas[0][1]
            # 投影平面：包含曲率軸和第二大變化軸
            other_axis = deltas[1][1]
            # 高度方向：起終點變化最大的軸
            height_axis = deltas[2][1]
            arc_axes = [curve_axis, other_axis]

            def proj(p):
                return (p[arc_axes[0]], p[arc_axes[1]])

            p0 = proj(centerline[0])
            p_mid = proj(centerline[n_samples // 2])
            p_end = proj(centerline[-1])

            # 三點擬合圓
            ax_, ay_ = p0
            bx_, by_ = p_mid
            cx_, cy_ = p_end
            D = 2 * (ax_ * (by_ - cy_) + bx_ * (cy_ - ay_) + cx_ * (ay_ - by_))

            if abs(D) > 1e-6:
                ux = ((ax_ ** 2 + ay_ ** 2) * (by_ - cy_) +
                      (bx_ ** 2 + by_ ** 2) * (cy_ - ay_) +
                      (cx_ ** 2 + cy_ ** 2) * (ay_ - by_)) / D
                uy = ((ax_ ** 2 + ay_ ** 2) * (cx_ - bx_) +
                      (bx_ ** 2 + by_ ** 2) * (ax_ - cx_) +
                      (cx_ ** 2 + cy_ ** 2) * (bx_ - ax_)) / D

                r0 = math.sqrt((ax_ - ux) ** 2 + (ay_ - uy) ** 2)
                r1 = math.sqrt((bx_ - ux) ** 2 + (by_ - uy) ** 2)
                r2 = math.sqrt((cx_ - ux) ** 2 + (cy_ - uy) ** 2)
                arc_radius = (r0 + r1 + r2) / 3

                angle_start = math.atan2(ay_ - uy, ax_ - ux)
                angle_end = math.atan2(cy_ - uy, cx_ - ux)
                arc_angle = abs(angle_end - angle_start)
                if arc_angle > math.pi:
                    arc_angle = 2 * math.pi - arc_angle
                arc_angle_deg = round(math.degrees(arc_angle), 1)

                height_gain = abs(centerline[-1][height_axis] - centerline[0][height_axis])

                if height_gain > 1:
                    cl_arc = math.sqrt((arc_radius * arc_angle) ** 2 + height_gain ** 2)
                else:
                    cl_arc = arc_radius * arc_angle

                outer_r = arc_radius + pipe_diameter / 2
                if height_gain > 1:
                    outer_arc = math.sqrt((outer_r * arc_angle) ** 2 + height_gain ** 2)
                else:
                    outer_arc = outer_r * arc_angle

                segments.append({
                    'type': 'arc',
                    'radius': round(arc_radius, 0),
                    'angle_deg': arc_angle_deg,
                    'arc_length': round(cl_arc, 1),
                    'outer_arc_length': round(outer_arc, 0),
                    'height_gain': round(height_gain, 1),
                })
            else:
                # 退化情況（共線）→ 視為直管
                d = (end_dx / end_dist, end_dy / end_dist, end_dz / end_dist) if end_dist > 0 else (0, 0, 0)
                segments.append({
                    'type': 'straight',
                    'length': round(total_length, 1),
                    'direction': d,
                })

        return {
            'solid_id': feature_id,
            'pipe_radius': round(pipe_diameter / 2, 2),
            'pipe_diameter': round(pipe_diameter, 1),
            'total_length': round(total_length, 1),
            'segments': segments,
            'start_point': centerline[0],
            'end_point': centerline[-1],
            'cylinder_faces': [],
            'method': 'bspline',
        }

    def _extract_pipe_centerlines(self) -> List[Dict]:
        """
        提取管路中心線資料
        對每個實體，檢查圓柱面以識別管件，提取中心線路徑
        支援 OCP 精確分析、BSpline 表面分析與啟發式後備方案
        """
        results = []

        solid_features = [f for f in self.features if f.type == "solid"]
        if not solid_features:
            return results

        ocp_available = False
        try:
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.TopoDS import TopoDS
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            from OCP.GeomAbs import GeomAbs_Cylinder
            from OCP.GProp import GProp_GProps
            from OCP.BRepGProp import BRepGProp
            ocp_available = True
        except ImportError:
            pass

        for feat in solid_features:
            fid = feat.id
            params = feat.params
            solid_shape = self._solid_shapes.get(fid)

            cylinder_faces = []

            if ocp_available and solid_shape is not None:
                # === OCP 精確路徑 ===
                try:
                    face_explorer = TopExp_Explorer(solid_shape, TopAbs_FACE)
                    while face_explorer.More():
                        face = TopoDS.Face_s(face_explorer.Current())
                        try:
                            surface = BRepAdaptor_Surface(face)
                            if surface.GetType() == GeomAbs_Cylinder:
                                cyl = surface.Cylinder()
                                radius = cyl.Radius()
                                axis = cyl.Axis()
                                loc = axis.Location()
                                direction = axis.Direction()

                                # 計算面積以估算長度
                                face_props = GProp_GProps()
                                BRepGProp.SurfaceProperties_s(face, face_props)
                                area = face_props.Mass()
                                est_length = area / (2 * math.pi * radius) if radius > 0 else 0

                                cylinder_faces.append({
                                    'radius': radius,
                                    'axis_origin': (loc.X(), loc.Y(), loc.Z()),
                                    'axis_direction': (direction.X(), direction.Y(), direction.Z()),
                                    'area': area,
                                    'est_length': est_length,
                                })
                        except Exception:
                            pass
                        face_explorer.Next()
                except Exception as e:
                    log_print(f"[Pipe] OCP face exploration failed for {fid}: {e}", "warning")

            if not cylinder_faces:
                # === 嘗試 BSpline 表面分析 ===
                if ocp_available and solid_shape is not None:
                    bspline_result = self._analyze_bspline_pipe(solid_shape, fid, params)
                    if bspline_result:
                        results.append(bspline_result)
                        continue

                # === 啟發式後備方案 ===
                bbox_l = params.get('bbox_l', 0)
                bbox_w = params.get('bbox_w', 0)
                bbox_h = params.get('bbox_h', 0)
                volume = params.get('volume', 0)

                if bbox_l <= 0 or bbox_w <= 0 or bbox_h <= 0:
                    continue

                bbox_volume = bbox_l * bbox_w * bbox_h
                fill_ratio = volume / bbox_volume if bbox_volume > 0 else 0

                # 管件的填充比通常在 0.2~0.7（圓形截面在矩形框中）
                if fill_ratio < 0.15 or fill_ratio > 0.85:
                    continue

                dims = sorted([bbox_l, bbox_w, bbox_h])
                min_dim = dims[0]
                mid_dim = dims[1]
                max_dim = dims[2]

                # 判斷管件：兩短邊近似且長邊遠大於短邊
                if min_dim > 0 and mid_dim / min_dim < 2.0 and max_dim / min_dim > 2.0:
                    est_radius = (min_dim + mid_dim) / 4.0
                    est_length = max_dim

                    # 推算方向（沿最長軸）
                    if max_dim == bbox_l:
                        direction = (1, 0, 0)
                    elif max_dim == bbox_w:
                        direction = (0, 1, 0)
                    else:
                        direction = (0, 0, 1)

                    results.append({
                        'solid_id': fid,
                        'pipe_radius': est_radius,
                        'pipe_diameter': est_radius * 2,
                        'total_length': est_length,
                        'segments': [
                            {'type': 'straight', 'length': est_length,
                             'direction': direction}
                        ],
                        'start_point': (params.get('x', 0), params.get('y', 0), params.get('z', 0)),
                        'end_point': (params.get('x', 0), params.get('y', 0), params.get('z', 0)),
                        'cylinder_faces': [],
                        'method': 'heuristic',
                    })
                continue

            # === 處理 OCP 圓柱面資料 ===
            # 依半徑分組
            radius_groups = {}
            tolerance = 0.5  # mm
            for cf in cylinder_faces:
                r = cf['radius']
                matched = False
                for group_r in radius_groups:
                    if abs(r - group_r) < tolerance:
                        radius_groups[group_r].append(cf)
                        matched = True
                        break
                if not matched:
                    radius_groups[r] = [cf]

            if not radius_groups:
                continue

            # 找出主要管徑（面積最大的半徑群組）
            dominant_radius = max(radius_groups.keys(),
                                 key=lambda r: sum(cf['area'] for cf in radius_groups[r]))
            dominant_faces = radius_groups[dominant_radius]

            # 建構中心線段
            segments = []
            total_length = 0

            # 收集不同方向的圓柱軸
            dir_groups = []
            dir_tolerance = 0.1
            for cf in dominant_faces:
                d = cf['axis_direction']
                matched_group = None
                for dg in dir_groups:
                    ref_d = dg[0]['axis_direction']
                    # 點積判斷方向是否相同（考慮反向）
                    dot = abs(d[0]*ref_d[0] + d[1]*ref_d[1] + d[2]*ref_d[2])
                    if dot > (1.0 - dir_tolerance):
                        matched_group = dg
                        break
                if matched_group is not None:
                    matched_group.append(cf)
                else:
                    dir_groups.append([cf])

            for dg in dir_groups:
                group_length = sum(cf['est_length'] for cf in dg)
                direction = dg[0]['axis_direction']

                if len(dir_groups) > 1:
                    # 多個方向 → 可能有彎曲段
                    segments.append({
                        'type': 'straight',
                        'length': group_length,
                        'direction': direction,
                    })
                else:
                    segments.append({
                        'type': 'straight',
                        'length': group_length,
                        'direction': direction,
                    })
                total_length += group_length

            # 如果有多個方向群組，嘗試偵測彎弧
            if len(dir_groups) >= 2:
                for i in range(len(dir_groups) - 1):
                    d1 = dir_groups[i][0]['axis_direction']
                    d2 = dir_groups[i + 1][0]['axis_direction']
                    dot = d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2]
                    dot = max(-1.0, min(1.0, dot))
                    angle_rad = math.acos(abs(dot))
                    angle_deg = math.degrees(angle_rad)

                    if angle_deg > 1.0:
                        # 從小半徑圓柱面估算彎曲半徑
                        other_radii = [r for r in radius_groups if abs(r - dominant_radius) > tolerance]
                        if other_radii:
                            bend_radius = min(other_radii)
                        else:
                            bend_radius = dominant_radius * 4  # 預設彎曲半徑

                        arc_length = bend_radius * angle_rad
                        outer_arc = (bend_radius + dominant_radius) * angle_rad

                        segments.append({
                            'type': 'arc',
                            'radius': bend_radius,
                            'angle_deg': round(angle_deg, 1),
                            'arc_length': round(arc_length, 1),
                            'outer_arc_length': round(outer_arc, 1),
                        })
                        total_length += arc_length

            # 起點/終點
            cx = params.get('x', 0)
            cy = params.get('y', 0)
            cz = params.get('z', 0)
            bbox_l = params.get('bbox_l', 0)
            bbox_w = params.get('bbox_w', 0)
            bbox_h = params.get('bbox_h', 0)

            results.append({
                'solid_id': fid,
                'pipe_radius': round(dominant_radius, 2),
                'pipe_diameter': round(dominant_radius * 2, 2),
                'total_length': round(total_length, 1),
                'segments': segments,
                'start_point': (cx - bbox_l/2, cy - bbox_w/2, cz - bbox_h/2),
                'end_point': (cx + bbox_l/2, cy + bbox_w/2, cz + bbox_h/2),
                'cylinder_faces': cylinder_faces,
                'method': 'ocp',
            })

        return results

    def _classify_parts(self, features: List[GeometricFeature]) -> List[Dict]:
        """
        依據幾何啟發式規則自動分類零件
        分類為: 腳架(leg), 支撐架(bracket), 軌道(track), 底座/連接件(base)
        """
        solid_features = [f for f in features if f.type == "solid"]
        if not solid_features:
            return []

        # 建立管路查詢表
        pipe_solid_ids = set()
        pipe_map = {pc['solid_id']: pc for pc in self._pipe_centerlines}
        for pc in self._pipe_centerlines:
            pipe_solid_ids.add(pc['solid_id'])

        # 偵測 BSpline 曲面零件（彎管軌道的特徵）
        bspline_solid_ids = set()
        try:
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.TopoDS import TopoDS
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            from OCP.GeomAbs import GeomAbs_BSplineSurface, GeomAbs_Plane
            for feat in solid_features:
                fid = feat.id
                solid_shape = self._solid_shapes.get(fid)
                if solid_shape is None:
                    continue
                face_exp = TopExp_Explorer(solid_shape, TopAbs_FACE)
                has_bspline = False
                plane_count = 0
                bspline_count = 0
                while face_exp.More():
                    face = TopoDS.Face_s(face_exp.Current())
                    try:
                        surf = BRepAdaptor_Surface(face)
                        if surf.GetType() == GeomAbs_BSplineSurface:
                            bspline_count += 1
                        elif surf.GetType() == GeomAbs_Plane:
                            plane_count += 1
                    except Exception:
                        pass
                    face_exp.Next()
                if bspline_count > 0:
                    bspline_solid_ids.add(fid)
        except ImportError:
            pass

        # 收集所有 solid 的統計資料
        volumes = [f.params.get('volume', 0) for f in solid_features]
        heights = [f.params.get('bbox_w', 0) for f in solid_features]  # Y 軸高度
        y_positions = [f.params.get('y', 0) for f in solid_features]
        max_volume = max(volumes) if volumes else 1
        min_y = min(y_positions) if y_positions else 0
        max_y = max(y_positions) if y_positions else 0
        assembly_y_span = max_y - min_y if max_y > min_y else 1

        classifications = []

        # 第一輪：偵測體積相同的群組（支撐架）
        # 使用容差分組（1% 體積差視為相同，應對浮點誤差）
        bracket_indices = set()
        bracket_group_id = 0
        assigned = [False] * len(solid_features)
        for i, feat_i in enumerate(solid_features):
            if assigned[i]:
                continue
            vi = feat_i.params.get('volume', 0)
            if vi <= 0:
                continue
            group = [i]
            for j in range(i + 1, len(solid_features)):
                if assigned[j]:
                    continue
                vj = solid_features[j].params.get('volume', 0)
                if vj > 0 and abs(vi - vj) / vi < 0.01:
                    group.append(j)
            if len(group) >= 3:
                bracket_group_id += 1
                for idx in group:
                    bracket_indices.add(idx)
                    assigned[idx] = True

        # 分類每個 solid
        group_counters = {'leg': 0, 'bracket': 0, 'track': 0, 'base': 0}

        for i, feat in enumerate(solid_features):
            fid = feat.id
            params = feat.params
            volume = params.get('volume', 0)
            bbox_l = params.get('bbox_l', 0)
            bbox_w = params.get('bbox_w', 0)
            bbox_h = params.get('bbox_h', 0)
            cx = params.get('x', 0)
            cy = params.get('y', 0)
            cz = params.get('z', 0)

            dims = sorted([bbox_l, bbox_w, bbox_h])
            min_dim = dims[0] if dims[0] > 0 else 0.01
            mid_dim = dims[1] if dims[1] > 0 else 0.01
            max_dim = dims[2]
            slenderness = max_dim / min_dim if min_dim > 0 else 0

            classification = None
            confidence = 0.5

            # R2: 支撐架（3+ 個相同體積的零件，優先於所有判定）
            if i in bracket_indices:
                classification = 'bracket'
                confidence = 0.85
                group_counters['bracket'] += 1

            # R1: 腳架（OCP 圓柱面偵測中，管徑遠小於 bbox 截面 → 矩形管倒角）
            # 注意：只對 OCP 圓柱面偵測結果適用，BSpline 偵測的管件跳過此規則
            if classification is None and fid in pipe_solid_ids:
                pipe_data = pipe_map.get(fid)
                if pipe_data and pipe_data.get('method') == 'ocp':
                    pipe_d = pipe_data['pipe_diameter']
                    ratio = pipe_d / min_dim if min_dim > 0 else 0
                    if ratio < 0.7 and slenderness > 3.0:
                        classification = 'leg'
                        confidence = 0.85
                        group_counters['leg'] += 1

            # R3a: 有 BSpline 曲面 → 軌道（彎管）
            if classification is None and fid in bspline_solid_ids:
                classification = 'track'
                confidence = 0.85
                group_counters['track'] += 1

            # R3b: 有圓柱面且管徑接近 bbox 截面 → 軌道（直管）
            if classification is None and fid in pipe_solid_ids:
                pipe_data = pipe_map.get(fid)
                if pipe_data and volume > max_volume * 0.05:
                    pipe_d = pipe_data['pipe_diameter']
                    ratio = pipe_d / min_dim if min_dim > 0 else 0
                    if ratio >= 0.7:
                        classification = 'track'
                        confidence = 0.9 if pipe_data.get('method') == 'ocp' else 0.7
                        group_counters['track'] += 1

            # R1b: 細長矩形截面 → 腳架（後備）
            if classification is None and slenderness > 4.0:
                if mid_dim / min_dim < 2.5:
                    classification = 'leg'
                    confidence = 0.75
                    group_counters['leg'] += 1

            # R4: 底座/連接件（剩餘）
            if classification is None:
                classification = 'base'
                confidence = 0.6
                group_counters['base'] += 1

            class_zh_map = {
                'leg': '腳架',
                'bracket': '支撐架',
                'track': '軌道',
                'base': '底座/連接件',
            }

            classifications.append({
                'feature_id': fid,
                'class': classification,
                'class_zh': class_zh_map.get(classification, classification),
                'confidence': confidence,
                'group_id': group_counters[classification],
                'volume': volume,
                'bbox': (bbox_l, bbox_w, bbox_h),
                'centroid': (cx, cy, cz),
                'slenderness': round(slenderness, 1),
            })

        return classifications

    def _calculate_angles(self) -> List[Dict]:
        """
        計算分類零件間的角度關係
        包括腳架安裝角、軌道仰角、彎管角度等
        支援 OCP 面法線分析與 bbox 啟發式後備
        """
        angles = []

        if not self._part_classifications:
            return angles

        # 建立分類查詢表
        class_map = {c['feature_id']: c for c in self._part_classifications}
        pipe_map = {pc['solid_id']: pc for pc in self._pipe_centerlines}

        ocp_available = False
        try:
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.TopoDS import TopoDS
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            from OCP.GeomAbs import GeomAbs_Plane
            from OCP.GProp import GProp_GProps
            from OCP.BRepGProp import BRepGProp
            ocp_available = True
        except ImportError:
            pass

        def get_principal_axis(fid):
            """取得零件的主軸方向"""
            cls_info = class_map.get(fid)
            if not cls_info:
                return None

            # 優先使用管路中心線方向
            pipe_data = pipe_map.get(fid)
            if pipe_data and pipe_data['segments']:
                straight_segs = [s for s in pipe_data['segments'] if s['type'] == 'straight']
                if straight_segs:
                    # 取面積最大的直線段方向
                    seg = max(straight_segs, key=lambda s: s.get('length', 0))
                    return seg['direction']

            # OCP: 使用平面法線的叉積取軸向
            solid_shape = self._solid_shapes.get(fid)
            if ocp_available and solid_shape is not None:
                try:
                    face_explorer = TopExp_Explorer(solid_shape, TopAbs_FACE)
                    planar_faces = []
                    while face_explorer.More():
                        face = TopoDS.Face_s(face_explorer.Current())
                        try:
                            surface = BRepAdaptor_Surface(face)
                            if surface.GetType() == GeomAbs_Plane:
                                plane = surface.Plane()
                                normal = plane.Axis().Direction()
                                face_props = GProp_GProps()
                                BRepGProp.SurfaceProperties_s(face, face_props)
                                area = face_props.Mass()
                                planar_faces.append((
                                    (normal.X(), normal.Y(), normal.Z()),
                                    area
                                ))
                        except Exception:
                            pass
                        face_explorer.Next()

                    if len(planar_faces) >= 2:
                        planar_faces.sort(key=lambda x: x[1], reverse=True)
                        n1 = planar_faces[0][0]
                        n2 = planar_faces[1][0]
                        # 叉積 = 主軸方向
                        axis = (
                            n1[1]*n2[2] - n1[2]*n2[1],
                            n1[2]*n2[0] - n1[0]*n2[2],
                            n1[0]*n2[1] - n1[1]*n2[0],
                        )
                        mag = math.sqrt(sum(a**2 for a in axis))
                        if mag > 1e-6:
                            return tuple(a/mag for a in axis)
                except Exception:
                    pass

            # 後備：使用 bbox 最長維度方向
            feat = next((f for f in self.features if f.id == fid), None)
            if feat:
                bl = feat.params.get('bbox_l', 0)
                bw = feat.params.get('bbox_w', 0)
                bh = feat.params.get('bbox_h', 0)
                max_dim = max(bl, bw, bh)
                if max_dim > 0:
                    if max_dim == bl:
                        return (1, 0, 0)
                    elif max_dim == bw:
                        return (0, 1, 0)
                    else:
                        return (0, 0, 1)

            return None

        def angle_between(v1, v2):
            """計算兩向量間的角度（度）"""
            dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
            dot = max(-1.0, min(1.0, dot))
            return math.degrees(math.acos(abs(dot)))

        # 偵測模型朝向：Y 軸向上假設（以組件最低點判斷地面）
        ground_normal = (0, 1, 0)  # Y-up

        # 計算各類角度
        legs = [c for c in self._part_classifications if c['class'] == 'leg']
        tracks = [c for c in self._part_classifications if c['class'] == 'track']

        # 腳架 vs 地面
        for leg in legs:
            axis = get_principal_axis(leg['feature_id'])
            if axis:
                angle = angle_between(axis, ground_normal)
                # 腳架角度通常相對於地面法線量測
                # 與地面的夾角 = 90 - 與法線的夾角
                ground_angle = 90.0 - angle if angle < 90 else angle - 90.0
                angles.append({
                    'type': 'leg_to_ground',
                    'part_a': leg['feature_id'],
                    'part_b': 'ground',
                    'angle_deg': round(ground_angle, 1),
                    'description': '腳架安裝角度',
                    'axis': axis,
                })

        # 軌道仰角
        for track in tracks:
            axis = get_principal_axis(track['feature_id'])
            if axis:
                horizontal = (axis[0], 0, axis[2])
                hmag = math.sqrt(horizontal[0]**2 + horizontal[2]**2)
                if hmag > 1e-6:
                    horizontal = (horizontal[0]/hmag, 0, horizontal[2]/hmag)
                    elevation = angle_between(axis, horizontal)
                    angles.append({
                        'type': 'track_elevation',
                        'part_a': track['feature_id'],
                        'part_b': 'horizontal',
                        'angle_deg': round(elevation, 1),
                        'description': '軌道仰角',
                        'axis': axis,
                    })

        # 軌道間彎管角度（從 pipe segments）
        for pc in self._pipe_centerlines:
            cls_info = class_map.get(pc['solid_id'])
            if cls_info and cls_info['class'] == 'track':
                for seg in pc['segments']:
                    if seg['type'] == 'arc':
                        angles.append({
                            'type': 'track_bend',
                            'part_a': pc['solid_id'],
                            'part_b': pc['solid_id'],
                            'angle_deg': seg['angle_deg'],
                            'description': f"軌道彎管角度 (R={seg.get('radius', 0):.0f})",
                        })

        # 腳架 vs 軌道
        for leg in legs:
            leg_axis = get_principal_axis(leg['feature_id'])
            if not leg_axis:
                continue
            for track in tracks:
                track_axis = get_principal_axis(track['feature_id'])
                if not track_axis:
                    continue
                angle = angle_between(leg_axis, track_axis)
                angles.append({
                    'type': 'leg_to_track',
                    'part_a': leg['feature_id'],
                    'part_b': track['feature_id'],
                    'angle_deg': round(angle, 1),
                    'description': '腳架與軌道夾角',
                })

        return angles

    def _generate_cutting_list(self) -> Dict:
        """
        生成取料明細（軌道取料 + 腳架 + 支撐架）
        結合管路中心線資料與零件分類
        """
        result = {
            'track_items': [],
            'leg_items': [],
            'bracket_items': [],
        }

        if not self._part_classifications:
            return result

        class_map = {c['feature_id']: c for c in self._part_classifications}
        pipe_map = {pc['solid_id']: pc for pc in self._pipe_centerlines}

        # 軌道取料明細
        track_parts = [c for c in self._part_classifications if c['class'] == 'track']

        def _estimate_pipe_diameter(feat):
            """從 bbox 和體積估算管徑（適用於 BSpline 彎管）"""
            p = feat.params
            vol = p.get('volume', 0)
            bl, bw, bh = p.get('bbox_l', 0), p.get('bbox_w', 0), p.get('bbox_h', 0)
            dims = sorted([bl, bw, bh])
            # 估算管長（最長維度或對角線）
            est_length = max(dims[2], math.sqrt(dims[1]**2 + dims[2]**2))
            if est_length > 0:
                # 截面積 = vol / length
                cross_area = vol / est_length
                # 截面積 = π * r_outer² - π * r_inner² ≈ π * d * t
                # 對薄壁管近似: cross_area ≈ π * d * t
                # 無法精確求解 d 和 t，但可用其他軌道的管徑作參考
                pass
            return dims[0], est_length  # 返回 (估算管短邊尺寸, 估算長度)

        def _build_track_items(tracks, prefix):
            """為一組軌道零件生成取料項目"""
            items = []
            item_num = 1
            # 從已知管路取得參考管徑
            ref_diameter = None
            for t in tracks:
                pd = pipe_map.get(t['feature_id'])
                if pd:
                    ref_diameter = pd['pipe_diameter']
                    break
            # 如果沒有，從所有管路取
            if ref_diameter is None and self._pipe_centerlines:
                # 取最大管徑作為軌道參考
                ref_diameter = max(pc['pipe_diameter'] for pc in self._pipe_centerlines)

            for track in tracks:
                fid = track['feature_id']
                pipe_data = pipe_map.get(fid)

                if pipe_data:
                    # 有管路中心線資料 → 按段生成
                    diameter = pipe_data['pipe_diameter']
                    for seg in pipe_data['segments']:
                        item_id = f"{prefix}{item_num}"
                        if seg['type'] == 'straight':
                            spec = f"直徑{diameter:.1f} 長度{seg['length']:.1f}"
                            items.append({
                                'item': item_id, 'diameter': diameter,
                                'spec': spec, 'type': 'straight',
                                'length': seg['length'],
                            })
                        elif seg['type'] == 'arc':
                            outer_arc = seg.get('outer_arc_length', seg.get('arc_length', 0))
                            spec = (f"直徑{diameter:.1f} "
                                    f"角度{seg['angle_deg']}度"
                                    f"(半徑{seg.get('radius', 0):.0f})"
                                    f"外弧長{outer_arc:.0f}")
                            items.append({
                                'item': item_id, 'diameter': diameter,
                                'spec': spec, 'type': 'arc',
                                'angle_deg': seg['angle_deg'],
                                'radius': seg.get('radius', 0),
                                'outer_arc_length': outer_arc,
                            })
                        item_num += 1
                else:
                    # 無管路中心線（BSpline 彎管）→ 從 bbox 估算
                    feat = next((f for f in self.features if f.id == fid), None)
                    if not feat:
                        continue
                    p = feat.params
                    bl = p.get('bbox_l', 0)
                    bw = p.get('bbox_w', 0)
                    bh = p.get('bbox_h', 0)
                    dims = sorted([bl, bw, bh])
                    diameter = ref_diameter if ref_diameter else dims[0]
                    est_length = dims[2]  # 最長邊作為估算長度

                    item_id = f"{prefix}{item_num}"
                    spec = f"直徑{diameter:.1f} 長度{est_length:.1f}"
                    items.append({
                        'item': item_id, 'diameter': diameter,
                        'spec': spec, 'type': 'straight',
                        'length': est_length,
                        'estimated': True,
                    })
                    item_num += 1

            return items

        # 依 Z 間距配對軌道，分為兩條平行軌道（上軌/下軌）
        if track_parts:
            # 配對法：找出 XY 接近但 Z 不同的軌道對
            track_infos = []
            for tp in track_parts:
                cx, cy, cz = tp['centroid']
                track_infos.append({'data': tp, 'cx': cx, 'cy': cy, 'cz': cz})

            paired = set()
            pairs = []
            for i in range(len(track_infos)):
                if i in paired:
                    continue
                best_j = None
                best_xy = float('inf')
                for j in range(i + 1, len(track_infos)):
                    if j in paired:
                        continue
                    xy_d = math.sqrt(
                        (track_infos[i]['cx'] - track_infos[j]['cx']) ** 2 +
                        (track_infos[i]['cy'] - track_infos[j]['cy']) ** 2)
                    z_d = abs(track_infos[i]['cz'] - track_infos[j]['cz'])
                    if z_d > 50 and xy_d < z_d and xy_d < best_xy:
                        best_xy = xy_d
                        best_j = j
                if best_j is not None:
                    paired.add(i)
                    paired.add(best_j)
                    pairs.append((i, best_j))

            rail_a, rail_b = [], []
            for i, j in pairs:
                if track_infos[i]['cz'] < track_infos[j]['cz']:
                    rail_a.append(track_infos[i])
                    rail_b.append(track_infos[j])
                else:
                    rail_a.append(track_infos[j])
                    rail_b.append(track_infos[i])

            for i, ti in enumerate(track_infos):
                if i not in paired:
                    rail_a.append(ti)

            # 各軌道內依 Y 位置排序（沿軌道路徑方向）
            rail_a.sort(key=lambda t: t['cy'])
            rail_b.sort(key=lambda t: t['cy'])

            upper_tracks = [t['data'] for t in rail_a]
            lower_tracks = [t['data'] for t in rail_b]

            result['track_items'] = (
                _build_track_items(upper_tracks, "U") +
                _build_track_items(lower_tracks, "D")
            )

        # 腳架明細
        leg_parts = [c for c in self._part_classifications if c['class'] == 'leg']
        for i, leg in enumerate(leg_parts, 1):
            fid = leg['feature_id']
            feat = next((f for f in self.features if f.id == fid), None)
            if feat:
                bl = feat.params.get('bbox_l', 0)
                bw = feat.params.get('bbox_w', 0)
                bh = feat.params.get('bbox_h', 0)
                # 線長 = 對角線或最長邊
                diagonal = math.sqrt(bl**2 + bw**2 + bh**2)
                max_dim = max(bl, bw, bh)
                line_length = max_dim  # 使用最長邊作為線長

                result['leg_items'].append({
                    'item': i,
                    'name': '腳架',
                    'quantity': 1,
                    'spec': f"線長L={line_length:.1f}",
                    'feature_id': fid,
                })

        # 支撐架明細
        bracket_parts = [c for c in self._part_classifications if c['class'] == 'bracket']
        if bracket_parts:
            # 按體積分組
            bracket_groups = {}
            for b in bracket_parts:
                v_key = round(b['volume'], 0)
                if v_key not in bracket_groups:
                    bracket_groups[v_key] = []
                bracket_groups[v_key].append(b)

            for i, (v_key, group) in enumerate(bracket_groups.items(), 1):
                sample = group[0]
                bbox = sample['bbox']
                spec = f"{bbox[0]:.1f}x{bbox[1]:.1f}x{bbox[2]:.1f}"
                result['bracket_items'].append({
                    'item': i,
                    'name': '支撐架',
                    'quantity': len(group),
                    'spec': spec,
                    'feature_ids': [b['feature_id'] for b in group],
                })

        return result

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

        # 加入進階分析結果
        info["pipe_centerlines"] = self._pipe_centerlines
        info["part_classifications"] = self._part_classifications
        info["angles"] = self._angles
        info["cutting_list"] = self._cutting_list

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

    def _annotate_xy_rot(self, msp, doc):
        """在 XY_rot 投影圖上加入 R260 弧線半徑與 568.1 垂直跨距標註"""
        import numpy as np

        # --- 從 DXF polyline 分析弧線和關鍵座標 ---
        all_points = []
        arc_candidates = []  # (points, xmax, y_span)

        for entity in msp.query('LWPOLYLINE'):
            pts = list(entity.get_points('xy'))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            all_points.extend(pts)
            xmax_e = max(xs)
            y_span = max(ys) - min(ys)
            if xmax_e > 700 and y_span > 200:
                arc_candidates.append(np.array(pts))

        if not all_points:
            return

        # 擬合圓：最小二乘法
        inner_radii = []
        outer_radii = []
        centers = []

        for pts_arr in arc_candidates:
            x = pts_arr[:, 0]
            y = pts_arr[:, 1]
            # 建立矩陣 Ax = b, 其中 [x, y, 1] * [D, E, F]^T = x^2 + y^2
            A = np.column_stack([x, y, np.ones_like(x)])
            b = x**2 + y**2
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx = result[0] / 2
            cy = result[1] / 2
            r = math.sqrt(result[2] + cx**2 + cy**2)
            centers.append((cx, cy))
            if r < 260:
                inner_radii.append(r)
            else:
                outer_radii.append(r)

        if not centers:
            return

        # 取平均弧心
        arc_cx = sum(c[0] for c in centers) / len(centers)
        arc_cy = sum(c[1] for c in centers) / len(centers)

        # 中心線半徑
        r_inner = sum(inner_radii) / len(inner_radii) if inner_radii else 236
        r_outer = sum(outer_radii) / len(outer_radii) if outer_radii else 284
        r_center = (r_inner + r_outer) / 2

        # 找水平橫桿 (Y 變化 < 5, X 跨距 > 100)
        bar_ys = []
        for entity in msp.query('LWPOLYLINE'):
            pts = list(entity.get_points('xy'))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            y_var = max(ys) - min(ys)
            x_span = max(xs) - min(xs)
            if y_var < 5 and x_span > 100:
                bar_ys.append(sum(ys) / len(ys))

        # 取最接近 Y=243 的橫桿
        if bar_ys:
            bar_y = min(bar_ys, key=lambda y: abs(y - 243))
        else:
            bar_y = 243.0

        # 結構底部 Y — 用百分位數排除極端離群點（如螺栓邊緣）
        all_ys_sorted = sorted(p[1] for p in all_points)
        pct_idx = max(1, len(all_ys_sorted) // 200)  # ~0.5th percentile
        arc_bottom_y = all_ys_sorted[pct_idx]

        # 找延伸線起始 X：各 Y 水平附近最左邊的幾何
        top_near = [p[0] for p in all_points if abs(p[1] - bar_y) < 15]
        bot_near = [p[0] for p in all_points if abs(p[1] - arc_bottom_y) < 15]
        top_x = min(top_near) if top_near else min(p[0] for p in all_points)
        bot_x = min(bot_near) if bot_near else min(p[0] for p in all_points)
        ext_left = min(top_x, bot_x)
        dim_x = ext_left - 40  # 尺寸線位置（圖左邊外側）

        # --- R260 標註：引線 + 文字 ---
        text_height = 15
        angle = math.radians(30)
        # 引線起點：弧面上 30° 位置（用外徑）
        arc_pt = (arc_cx + r_outer * math.cos(angle),
                  arc_cy + r_outer * math.sin(angle))
        # 引線終點：向外延伸
        leader_end = (arc_pt[0] + 50, arc_pt[1] + 30)
        msp.add_line(arc_pt, leader_end)
        # 水平延伸線
        text_end = (leader_end[0] + 60, leader_end[1])
        msp.add_line(leader_end, text_end)
        # 文字
        msp.add_text(f"R{r_center:.0f}", dxfattribs={
            'height': text_height,
            'insert': (leader_end[0] + 5, leader_end[1] + 5),
        })

        # --- 垂直跨距標註 ---
        height = abs(bar_y - arc_bottom_y)

        # 垂直尺寸線
        msp.add_line((dim_x, bar_y), (dim_x, arc_bottom_y))
        # 上端延伸線
        msp.add_line((dim_x - 8, bar_y), (dim_x + 8, bar_y))
        msp.add_line((top_x, bar_y), (dim_x + 8, bar_y))
        # 下端延伸線
        msp.add_line((dim_x - 8, arc_bottom_y), (dim_x + 8, arc_bottom_y))
        msp.add_line((bot_x, arc_bottom_y), (dim_x + 8, arc_bottom_y))
        # 箭頭
        arrow_len = 8
        msp.add_line((dim_x, bar_y), (dim_x - 3, bar_y - arrow_len))
        msp.add_line((dim_x, bar_y), (dim_x + 3, bar_y - arrow_len))
        msp.add_line((dim_x, arc_bottom_y), (dim_x - 3, arc_bottom_y + arrow_len))
        msp.add_line((dim_x, arc_bottom_y), (dim_x + 3, arc_bottom_y + arrow_len))
        # 文字（旋轉 90 度）
        text_y = (bar_y + arc_bottom_y) / 2
        msp.add_text(f"{height:.1f}", dxfattribs={
            'height': 12,
            'insert': (dim_x - 18, text_y),
            'rotation': 90,
        })

    def _annotate_yz_rot(self, msp, doc):
        """在 YZ_rot 投影圖上加入 850.8 斜向跨距標註"""
        # --- 從 DXF polyline 收集所有端點 ---
        endpoints = []  # (x, y)

        for entity in msp.query('LWPOLYLINE'):
            pts = list(entity.get_points('xy'))
            if not pts:
                continue
            endpoints.append((pts[0][0], pts[0][1]))
            endpoints.append((pts[-1][0], pts[-1][1]))

        if not endpoints:
            return

        # 搜尋所有端點對，找距離最接近 850.8 的
        target = 850.8
        best_pair = None
        best_diff = float('inf')

        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                x1, y1 = endpoints[i]
                x2, y2 = endpoints[j]
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                diff = abs(dist - target)
                if diff < best_diff:
                    best_diff = diff
                    best_pair = ((x1, y1), (x2, y2), dist)

        if best_pair is None or best_diff > 5:
            return

        p1, p2, actual_dist = best_pair
        # 確保 p1 在上方（Y 較大），p2 在下方
        if p1[1] < p2[1]:
            p1, p2 = p2, p1

        text_height = 15

        # --- 斜向尺寸線 ---
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx**2 + dy**2)
        # 法向量：向左上方（結構外側），遠離結構中心
        nx = dy / length
        ny = -dx / length
        offset = 80

        d1 = (p1[0] + nx * offset, p1[1] + ny * offset)
        d2 = (p2[0] + nx * offset, p2[1] + ny * offset)

        # 尺寸線
        msp.add_line(d1, d2)
        # 延伸線
        msp.add_line(p1, d1)
        msp.add_line(p2, d2)

        # 箭頭
        arrow_len = 10
        ux = dx / length
        uy = dy / length
        msp.add_line(d1, (d1[0] + ux * arrow_len - nx * 3,
                          d1[1] + uy * arrow_len - ny * 3))
        msp.add_line(d1, (d1[0] + ux * arrow_len + nx * 3,
                          d1[1] + uy * arrow_len + ny * 3))
        msp.add_line(d2, (d2[0] - ux * arrow_len - nx * 3,
                          d2[1] - uy * arrow_len - ny * 3))
        msp.add_line(d2, (d2[0] - ux * arrow_len + nx * 3,
                          d2[1] - uy * arrow_len + ny * 3))

        # 文字
        mid = ((d1[0] + d2[0]) / 2 + nx * 5,
               (d1[1] + d2[1]) / 2 + ny * 5)
        text_angle = math.degrees(math.atan2(dy, dx))
        if text_angle < -90:
            text_angle += 180
        elif text_angle > 90:
            text_angle -= 180

        msp.add_text(f"{actual_dist:.1f}", dxfattribs={
            'height': text_height,
            'insert': mid,
            'rotation': text_angle,
        })

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

        # 定義投影模式：使用 HLR (Hidden Line Removal) 產生含輪廓線的正確投影
        # gp_Ax2(origin, main_direction, x_direction) 定義投影座標系
        #   main_direction = 視線方向 (Z軸)
        #   x_direction = 投影平面水平軸
        #   y_direction = 自動計算 (main × x)
        from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
        from OCP.HLRAlgo import HLRAlgo_Projector
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopoDS import TopoDS

        O = gp_Pnt(0, 0, 0)
        projections = [
            # (plane, suffix, description, main_dir, x_dir, flip_v)
            ('XY', '',     '俯視圖 (Top) - 直接投影',
             gp_Dir(0, 0, -1), gp_Dir(1, 0, 0), False),       # => (x, y)
            ('XZ', '',     '前視圖 (Front) - 直接投影',
             gp_Dir(0, -1, 0), gp_Dir(1, 0, 0), False),       # => (x, z)
            ('YZ', '',     '側視圖 (Right) - 直接投影',
             gp_Dir(-1, 0, 0), gp_Dir(0, -1, 0), False),      # => (-y, z)
            ('XY', '_rot', '俯視圖 (Top) - 反向',
             gp_Dir(0, 0, -1), gp_Dir(1, 0, 0), True),        # => (x, -y)
            ('XZ', '_rot', '前視圖 (Front) - 反向',
             gp_Dir(0, -1, 0), gp_Dir(1, 0, 0), False),       # => (x, z)
            ('YZ', '_rot', '側視圖 (Right) - 反向',
             gp_Dir(0, 1, 0),  gp_Dir(-1, 0, 0), False),      # => (-x, z)
        ]

        output_files = []
        occ_shape = self.cad_model.val().wrapped

        log_print("\n" + "=" * 60)
        log_print("開始 3D 模型投影轉換 (HLR)...")
        log_print("=" * 60)

        for plane_name, suffix, description, main_dir, x_dir, flip_v in projections:
            output_path = os.path.join(output_dir, f"{base_name}_{plane_name}{suffix}.dxf")
            log_print(f"\n[{plane_name}{suffix}] 正在生成 {description}...")

            try:
                # 1. 執行 HLR 投影
                hlr = HLRBRep_Algo()
                hlr.Add(occ_shape)

                ax2 = gp_Ax2(O, main_dir, x_dir)
                projector = HLRAlgo_Projector(ax2)
                hlr.Projector(projector)
                hlr.Update()
                hlr.Hide()

                hlr_shapes = HLRBRep_HLRToShape(hlr)

                # 2. 收集可見邊（銳邊 + 平滑邊 + 縫合邊 + 輪廓線）
                visible_compounds = [
                    hlr_shapes.VCompound(),              # 銳邊 (sharp)
                    hlr_shapes.Rg1LineVCompound(),        # 平滑邊 (smooth/fillet)
                    hlr_shapes.RgNLineVCompound(),        # 縫合邊 (sewn)
                    hlr_shapes.OutLineVCompound(),        # 輪廓線 (silhouette)
                ]

                # 3. 建立 DXF 並寫入邊
                doc = ezdxf.new()
                msp = doc.modelspace()

                edge_count = 0
                for compound in visible_compounds:
                    if compound.IsNull():
                        continue
                    exp = TopExp_Explorer(compound, TopAbs_EDGE)
                    while exp.More():
                        edge = TopoDS.Edge_s(exp.Current())
                        try:
                            curve = BRepAdaptor_Curve(edge)
                            u_start = curve.FirstParameter()
                            u_end = curve.LastParameter()

                            num_points = 20
                            points_2d = []
                            for i in range(num_points + 1):
                                u = u_start + (u_end - u_start) * i / num_points
                                pnt = curve.Value(u)
                                px, py = pnt.X(), pnt.Y()
                                if flip_v:
                                    py = -py
                                points_2d.append((px, py))

                            if len(points_2d) >= 2:
                                msp.add_lwpolyline(points_2d)
                                edge_count += 1
                        except Exception:
                            pass
                        exp.Next()

                # 4. 加入尺寸標註
                if plane_name == 'XY' and suffix == '_rot':
                    self._annotate_xy_rot(msp, doc)
                elif plane_name == 'YZ' and suffix == '_rot':
                    self._annotate_yz_rot(msp, doc)

                # 5. 儲存 DXF
                doc.saveas(output_path)
                log_print(f"[Success] {description} 已儲存")
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

            # ===== 3.5 繪製取料明細表 (Cutting List Table) =====
            cutting_list = info.get('cutting_list', {})
            track_items = cutting_list.get('track_items', [])
            if track_items:
                # 取料明細放在 BOM 上方
                # 計算 BOM 已用高度
                bom_used_rows = min(len(bom_data), 20)
                bom_content_height = 30 + row_height + bom_used_rows * row_height + 20

                cut_x = bom_x
                cut_y = bom_y + bom_content_height
                cut_width = bom_width
                cut_row_height = 10
                cut_header_height = 15

                # 計算取料明細表高度
                remaining_space = (bom_y + bom_height) - cut_y - 5
                max_cut_rows = max(1, int((remaining_space - cut_header_height - cut_row_height) / cut_row_height))
                num_items = min(len(track_items), max_cut_rows)
                cut_table_height = cut_header_height + cut_row_height + num_items * cut_row_height + 5

                if remaining_space >= cut_header_height + cut_row_height * 2:
                    # 取料明細框
                    msp.add_line((cut_x, cut_y), (cut_x + cut_width, cut_y))
                    msp.add_line((cut_x + cut_width, cut_y), (cut_x + cut_width, cut_y + cut_table_height))
                    msp.add_line((cut_x + cut_width, cut_y + cut_table_height), (cut_x, cut_y + cut_table_height))
                    msp.add_line((cut_x, cut_y + cut_table_height), (cut_x, cut_y))

                    # 標題
                    msp.add_text("軌道取料明細", dxfattribs={'height': text_height * 1.1}).set_placement(
                        (cut_x + 10, cut_y + cut_table_height - 12))

                    # 表頭
                    ch_y = cut_y + cut_table_height - cut_header_height
                    msp.add_line((cut_x, ch_y), (cut_x + cut_width, ch_y))
                    msp.add_line((cut_x, ch_y - cut_row_height), (cut_x + cut_width, ch_y - cut_row_height))
                    msp.add_text("球號", dxfattribs={'height': text_height - 1}).set_placement(
                        (cut_x + 5, ch_y - 8))
                    msp.add_text("取料尺寸 (mm)", dxfattribs={'height': text_height - 1}).set_placement(
                        (cut_x + 35, ch_y - 8))
                    msp.add_line((cut_x + 30, ch_y), (cut_x + 30, ch_y - cut_row_height))

                    # 內容
                    for ci, citem in enumerate(track_items[:num_items]):
                        ci_y = ch_y - cut_row_height - (ci + 1) * cut_row_height
                        if ci_y < cut_y + 3:
                            break
                        msp.add_line((cut_x, ci_y), (cut_x + cut_width, ci_y))
                        msp.add_text(str(citem.get('item', '')),
                                     dxfattribs={'height': text_height - 1.5}).set_placement(
                            (cut_x + 5, ci_y + 2))
                        spec_text = str(citem.get('spec', ''))[:25]
                        msp.add_text(spec_text,
                                     dxfattribs={'height': text_height - 1.5}).set_placement(
                            (cut_x + 35, ci_y + 2))

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

    def generate_sub_assembly_drawing(self, output_dir: str = "output") -> str:
        """
        生成子系統施工圖 (Sub-assembly / Pipe Schematic)
        包含：管路展開示意圖、彎管資料表、取料明細表、管路概要、標題欄

        Args:
            output_dir: 輸出目錄

        Returns:
            輸出的 DXF 檔案路徑，失敗返回 None
        """
        if not CADQUERY_AVAILABLE:
            log_print("[Error] CadQuery 未安裝，無法生成子系統施工圖", "error")
            return None

        if self.cad_model is None:
            log_print("[Error] 未載入 3D 模型，無法生成子系統施工圖", "error")
            return None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        info = self.get_model_info()

        if self.model_file:
            base_name = os.path.splitext(os.path.basename(self.model_file))[0]
        else:
            base_name = "sub_assembly"

        output_path = os.path.join(output_dir, f"{base_name}_子系統施工圖.dxf")

        log_print("\n" + "=" * 60)
        log_print("開始生成子系統施工圖 (Pipe Schematic)...")
        log_print("=" * 60)

        try:
            doc = ezdxf.new(dxfversion='R2010')
            msp = doc.modelspace()

            # A3 圖紙 (420 x 297 mm)
            pw, ph = 420, 297
            margin = 10
            title_h = 50
            right_panel_w = 160

            # 區域定義
            inner_x = margin
            inner_y = margin + title_h
            inner_w = pw - margin * 2
            inner_h = ph - margin * 2 - title_h
            schematic_w = inner_w - right_panel_w
            rp_x = inner_x + schematic_w  # 右面板 X 起點

            # ===== 圖框 =====
            for pts in [
                [(0, 0), (pw, 0), (pw, ph), (0, ph), (0, 0)],
                [(margin, margin), (pw - margin, margin),
                 (pw - margin, ph - margin), (margin, ph - margin), (margin, margin)],
            ]:
                msp.add_lwpolyline(pts)

            # 標題欄上緣
            msp.add_line((margin, inner_y), (pw - margin, inner_y))
            # 右面板左緣
            msp.add_line((rp_x, inner_y), (rp_x, ph - margin))

            # ===== 標題欄 =====
            product_name = info.get('product_name') or base_name
            file_name = info.get('file_name') or 'N/A'
            source_sw = info.get('source_software') or 'N/A'
            units = info.get('units') or 'mm'
            today = datetime.now().strftime("%Y-%m-%d")
            th = 3.5

            # 標題欄水平分隔
            msp.add_line((margin, margin + 25), (pw - margin, margin + 25))
            # 垂直分隔
            col_xs = [margin + 140, margin + 250, margin + 320]
            for cx in col_xs:
                msp.add_line((cx, margin), (cx, margin + 25))

            msp.add_text(f"圖名: {product_name}",
                         dxfattribs={'height': th * 1.4}).set_placement(
                (margin + 8, margin + 33))
            msp.add_text(f"檔案: {file_name}",
                         dxfattribs={'height': th}).set_placement(
                (margin + 8, margin + 10))
            msp.add_text(f"來源: {source_sw}",
                         dxfattribs={'height': th}).set_placement(
                (margin + 145, margin + 10))
            msp.add_text(f"單位: {units}",
                         dxfattribs={'height': th}).set_placement(
                (margin + 255, margin + 10))
            msp.add_text(f"日期: {today}",
                         dxfattribs={'height': th}).set_placement(
                (margin + 325, margin + 10))
            msp.add_text("子系統施工圖 Sub-assembly Pipe Schematic",
                         dxfattribs={'height': th * 1.1}).set_placement(
                (margin + 255, margin + 33))

            # ===== 收集管路資料 =====
            cutting_list = info.get('cutting_list', {})
            track_items = cutting_list.get('track_items', [])
            leg_items = cutting_list.get('leg_items', [])
            bracket_items = cutting_list.get('bracket_items', [])
            pipe_centerlines = info.get('pipe_centerlines', [])
            part_classifications = info.get('part_classifications', [])
            angles = info.get('angles', [])

            # 管路統計
            track_pipes = [pc for pc in pipe_centerlines
                           if any(c['feature_id'] == pc['solid_id'] and c['class'] == 'track'
                                  for c in part_classifications)]
            all_bends = []
            total_pipe_length = 0
            pipe_diameter = 0
            for pc in track_pipes:
                total_pipe_length += pc.get('total_length', 0)
                pipe_diameter = max(pipe_diameter, pc.get('pipe_diameter', 0))
                for seg in pc.get('segments', []):
                    if seg['type'] == 'arc':
                        all_bends.append(seg)

            # ===== 右面板：管路概要 =====
            summary_h = 55
            summary_y = ph - margin - summary_h
            msp.add_line((rp_x, summary_y), (pw - margin, summary_y))

            msp.add_text("管路概要 (Pipe Summary)",
                         dxfattribs={'height': th * 1.1, 'color': 5}).set_placement(
                (rp_x + 5, ph - margin - 12))

            summary_lines = [
                f"管徑: Ø{pipe_diameter:.1f} mm" if pipe_diameter > 0 else "管徑: --",
                f"軌道總長: {total_pipe_length:.1f} mm" if total_pipe_length > 0 else "軌道總長: --",
                f"彎次: {len(all_bends)}",
                f"軌道數: {len(track_pipes)}",
                f"腳架數: {len(leg_items)}",
                f"支撐架數: {sum(b.get('quantity', 1) for b in bracket_items)}",
            ]
            for i, line in enumerate(summary_lines):
                msp.add_text(line, dxfattribs={'height': th}).set_placement(
                    (rp_x + 5, summary_y + summary_h - 25 - i * 7))

            # ===== 右面板：彎管資料表 =====
            bend_table_h = max(40, 20 + len(all_bends) * 8 + 10)
            bend_table_y = summary_y - bend_table_h - 5
            # 框
            msp.add_lwpolyline([
                (rp_x, bend_table_y), (pw - margin, bend_table_y),
                (pw - margin, bend_table_y + bend_table_h),
                (rp_x, bend_table_y + bend_table_h), (rp_x, bend_table_y)
            ])
            msp.add_text("彎管資料表 (Bend Schedule)",
                         dxfattribs={'height': th * 1.0, 'color': 5}).set_placement(
                (rp_x + 5, bend_table_y + bend_table_h - 10))

            # 表頭
            bt_header_y = bend_table_y + bend_table_h - 18
            msp.add_line((rp_x, bt_header_y), (pw - margin, bt_header_y))
            bt_row_h = 8
            bt_cols = [rp_x, rp_x + 25, rp_x + 65, rp_x + 105, pw - margin]
            headers_bend = ["#", "角度(°)", "CLR(mm)", "弧長(mm)"]
            for ci, header in enumerate(headers_bend):
                msp.add_text(header, dxfattribs={'height': th - 0.5, 'color': 2}).set_placement(
                    (bt_cols[ci] + 3, bt_header_y + 1))
            msp.add_line((rp_x, bt_header_y - bt_row_h), (pw - margin, bt_header_y - bt_row_h))
            # 垂直分隔
            for cx in bt_cols[1:-1]:
                msp.add_line((cx, bt_header_y + bt_row_h), (cx, bend_table_y))

            if all_bends:
                for bi, bend in enumerate(all_bends):
                    row_y = bt_header_y - bt_row_h - bi * bt_row_h
                    if row_y < bend_table_y + 3:
                        break
                    msp.add_line((rp_x, row_y), (pw - margin, row_y))
                    vals = [
                        f"B{bi + 1}",
                        f"{bend.get('angle_deg', 0):.1f}",
                        f"{bend.get('radius', 0):.0f}",
                        f"{bend.get('arc_length', bend.get('outer_arc_length', 0)):.1f}",
                    ]
                    for ci, v in enumerate(vals):
                        msp.add_text(v, dxfattribs={'height': th - 0.5}).set_placement(
                            (bt_cols[ci] + 3, row_y + 1))
            else:
                msp.add_text("(無彎管資料)", dxfattribs={'height': th}).set_placement(
                    (rp_x + 5, bt_header_y - bt_row_h - 5))

            # ===== 右面板：取料明細表 =====
            cut_table_y_top = bend_table_y - 5
            cut_table_y = inner_y
            cut_table_h = cut_table_y_top - cut_table_y
            if cut_table_h > 30:
                msp.add_lwpolyline([
                    (rp_x, cut_table_y), (pw - margin, cut_table_y),
                    (pw - margin, cut_table_y_top),
                    (rp_x, cut_table_y_top), (rp_x, cut_table_y)
                ])
                msp.add_text("取料明細 (Cutting List)",
                             dxfattribs={'height': th * 1.0, 'color': 5}).set_placement(
                    (rp_x + 5, cut_table_y_top - 10))

                # 表頭
                ct_header_y = cut_table_y_top - 18
                msp.add_line((rp_x, ct_header_y), (pw - margin, ct_header_y))
                ct_cols = [rp_x, rp_x + 25, pw - margin]
                msp.add_text("球號", dxfattribs={'height': th - 0.5, 'color': 2}).set_placement(
                    (ct_cols[0] + 3, ct_header_y + 1))
                msp.add_text("規格 (mm)", dxfattribs={'height': th - 0.5, 'color': 2}).set_placement(
                    (ct_cols[1] + 3, ct_header_y + 1))
                msp.add_line((rp_x, ct_header_y - bt_row_h), (pw - margin, ct_header_y - bt_row_h))
                msp.add_line((ct_cols[1], ct_header_y + bt_row_h), (ct_cols[1], cut_table_y))

                # 合併所有取料項目
                all_cut_items = []
                for ti in track_items:
                    all_cut_items.append((str(ti.get('item', '')), str(ti.get('spec', ''))))
                for li in leg_items:
                    all_cut_items.append((f"L{li.get('item', '')}", f"腳架 {li.get('spec', '')}"))
                for bi_item in bracket_items:
                    all_cut_items.append((
                        f"K{bi_item.get('item', '')}",
                        f"支撐架 x{bi_item.get('quantity', 1)} {bi_item.get('spec', '')}"
                    ))

                ct_row_y = ct_header_y - bt_row_h
                for ci_idx, (item_id, spec) in enumerate(all_cut_items):
                    row_y = ct_row_y - ci_idx * bt_row_h
                    if row_y < cut_table_y + 3:
                        break
                    msp.add_line((rp_x, row_y), (pw - margin, row_y))
                    msp.add_text(item_id, dxfattribs={'height': th - 0.5}).set_placement(
                        (ct_cols[0] + 3, row_y + 1))
                    spec_text = spec[:22]
                    msp.add_text(spec_text, dxfattribs={'height': th - 0.5}).set_placement(
                        (ct_cols[1] + 3, row_y + 1))

            # ===== 左面板：管路展開示意圖 =====
            schem_x = inner_x + 5
            schem_y = inner_y + 5
            schem_w = schematic_w - 10
            schem_h = inner_h - 10

            msp.add_text("管路展開示意圖 (Pipe Schematic)",
                         dxfattribs={'height': th * 1.2, 'color': 5}).set_placement(
                (schem_x, inner_y + inner_h - 12))

            # 分離上軌/下軌
            upper_items = [t for t in track_items if str(t.get('item', '')).startswith('U')]
            lower_items = [t for t in track_items if str(t.get('item', '')).startswith('D')]

            def _draw_pipe_schematic(msp, items, origin_x, origin_y, avail_w, avail_h, label):
                """繪製單條軌道的展開示意圖"""
                if not items:
                    msp.add_text(f"{label}: (無資料)", dxfattribs={'height': th}).set_placement(
                        (origin_x, origin_y + avail_h / 2))
                    return

                # 標籤
                msp.add_text(label, dxfattribs={'height': th * 1.0, 'color': 3}).set_placement(
                    (origin_x, origin_y + avail_h - 5))

                # 計算管路總展開長
                total_dev = 0
                for it in items:
                    if it.get('type') == 'straight':
                        total_dev += it.get('length', 0)
                    elif it.get('type') == 'arc':
                        total_dev += it.get('outer_arc_length', it.get('arc_length', 0))

                if total_dev <= 0:
                    total_dev = 1

                # 繪圖參數
                pipe_y = origin_y + avail_h * 0.45
                draw_w = avail_w - 20
                scale_x = draw_w / total_dev
                pipe_half = 4  # 管壁半寬度（圖面 mm）
                cursor_x = origin_x + 10

                bend_num = 0
                seg_num = 0

                for it in items:
                    if it.get('type') == 'straight':
                        seg_num += 1
                        seg_len = it.get('length', 0)
                        draw_len = seg_len * scale_x
                        if draw_len < 2:
                            draw_len = 2

                        # 管壁（雙線）
                        msp.add_line(
                            (cursor_x, pipe_y - pipe_half),
                            (cursor_x + draw_len, pipe_y - pipe_half))
                        msp.add_line(
                            (cursor_x, pipe_y + pipe_half),
                            (cursor_x + draw_len, pipe_y + pipe_half))
                        # 中心線（虛線用細線表示）
                        msp.add_line(
                            (cursor_x, pipe_y),
                            (cursor_x + draw_len, pipe_y),
                            dxfattribs={'color': 1, 'linetype': 'CENTER'})

                        # 尺寸標註（上方）
                        dim_y = pipe_y + pipe_half + 8
                        msp.add_line((cursor_x, dim_y - 2), (cursor_x, dim_y + 2))
                        msp.add_line((cursor_x + draw_len, dim_y - 2),
                                     (cursor_x + draw_len, dim_y + 2))
                        msp.add_line((cursor_x, dim_y), (cursor_x + draw_len, dim_y),
                                     dxfattribs={'color': 1})
                        # 箭頭
                        arrow_l = min(2, draw_len * 0.15)
                        msp.add_line((cursor_x, dim_y),
                                     (cursor_x + arrow_l, dim_y + 1),
                                     dxfattribs={'color': 1})
                        msp.add_line((cursor_x, dim_y),
                                     (cursor_x + arrow_l, dim_y - 1),
                                     dxfattribs={'color': 1})
                        msp.add_line((cursor_x + draw_len, dim_y),
                                     (cursor_x + draw_len - arrow_l, dim_y + 1),
                                     dxfattribs={'color': 1})
                        msp.add_line((cursor_x + draw_len, dim_y),
                                     (cursor_x + draw_len - arrow_l, dim_y - 1),
                                     dxfattribs={'color': 1})

                        # 尺寸數字
                        label_text = f"{seg_len:.1f}"
                        msp.add_text(label_text, dxfattribs={
                            'height': th - 0.5, 'color': 1
                        }).set_placement((cursor_x + draw_len / 2, dim_y + 2))

                        # 段號（下方）
                        msp.add_text(f"S{seg_num}", dxfattribs={
                            'height': th - 0.5, 'color': 7
                        }).set_placement((cursor_x + draw_len / 2, pipe_y - pipe_half - 8))

                        cursor_x += draw_len

                    elif it.get('type') == 'arc':
                        bend_num += 1
                        angle_deg = it.get('angle_deg', 0)
                        bend_r = it.get('radius', 0)
                        outer_arc = it.get('outer_arc_length', it.get('arc_length', 0))
                        draw_len = outer_arc * scale_x
                        if draw_len < 6:
                            draw_len = 6

                        # 彎曲符號：菱形標記
                        mid_x = cursor_x + draw_len / 2
                        diamond_w = min(draw_len * 0.4, 5)
                        diamond_h = pipe_half + 3

                        # 管壁帶斜線（表示彎曲）
                        msp.add_line(
                            (cursor_x, pipe_y - pipe_half),
                            (mid_x, pipe_y - pipe_half - 2))
                        msp.add_line(
                            (mid_x, pipe_y - pipe_half - 2),
                            (cursor_x + draw_len, pipe_y - pipe_half))
                        msp.add_line(
                            (cursor_x, pipe_y + pipe_half),
                            (mid_x, pipe_y + pipe_half + 2))
                        msp.add_line(
                            (mid_x, pipe_y + pipe_half + 2),
                            (cursor_x + draw_len, pipe_y + pipe_half))

                        # 彎曲角度標註（下方）
                        msp.add_text(f"B{bend_num}", dxfattribs={
                            'height': th, 'color': 6
                        }).set_placement((mid_x - 3, pipe_y - pipe_half - 10))
                        msp.add_text(f"{angle_deg:.0f}°  R{bend_r:.0f}", dxfattribs={
                            'height': th - 0.5, 'color': 6
                        }).set_placement((mid_x - 8, pipe_y - pipe_half - 17))

                        cursor_x += draw_len

                # 總展開長度
                total_len = sum(
                    it.get('length', 0) if it.get('type') == 'straight'
                    else it.get('outer_arc_length', it.get('arc_length', 0))
                    for it in items)
                msp.add_text(f"展開長: {total_len:.1f} mm", dxfattribs={
                    'height': th, 'color': 3
                }).set_placement((origin_x, origin_y + 3))

            # 分配空間：上軌佔上半，下軌佔下半
            half_h = (schem_h - 20) / 2
            if upper_items or lower_items:
                _draw_pipe_schematic(
                    msp, upper_items,
                    schem_x, schem_y + half_h + 10, schem_w, half_h,
                    "上軌 (Upper Rail)")
                _draw_pipe_schematic(
                    msp, lower_items,
                    schem_x, schem_y, schem_w, half_h,
                    "下軌 (Lower Rail)")
            else:
                # 無軌道資料，顯示所有管路中心線概要
                msp.add_text("(未偵測到軌道配對，顯示所有管路概要)",
                             dxfattribs={'height': th, 'color': 1}).set_placement(
                    (schem_x, schem_y + schem_h / 2))
                y_offset = schem_y + schem_h - 30
                for pc in pipe_centerlines[:8]:
                    sid = pc.get('solid_id', '?')
                    d = pc.get('pipe_diameter', 0)
                    tl = pc.get('total_length', 0)
                    n_seg = len(pc.get('segments', []))
                    msp.add_text(
                        f"{sid}: Ø{d:.1f}  L={tl:.1f}  segments={n_seg}",
                        dxfattribs={'height': th}).set_placement(
                        (schem_x + 5, y_offset))
                    y_offset -= 10

            # ===== 管截面示意（左下角）=====
            if pipe_diameter > 0:
                cs_x = schem_x + 10
                cs_y = schem_y + 15
                cs_r = 6  # 圖面上的半徑
                # 外圓
                msp.add_circle((cs_x, cs_y), cs_r)
                # 內圓（假設壁厚比 0.1）
                msp.add_circle((cs_x, cs_y), cs_r * 0.75)
                # 十字中心線
                msp.add_line((cs_x - cs_r - 2, cs_y), (cs_x + cs_r + 2, cs_y),
                             dxfattribs={'color': 1})
                msp.add_line((cs_x, cs_y - cs_r - 2), (cs_x, cs_y + cs_r + 2),
                             dxfattribs={'color': 1})
                msp.add_text(f"Ø{pipe_diameter:.1f}", dxfattribs={
                    'height': th - 0.5
                }).set_placement((cs_x + cs_r + 3, cs_y - 1))

            # ===== 角度資訊（如有）=====
            track_angles = [a for a in angles if a.get('type') in ('track_elevation', 'track_bend')]
            if track_angles:
                ang_x = schem_x + schem_w - 80
                ang_y = schem_y + 5
                msp.add_text("角度資訊:", dxfattribs={'height': th, 'color': 4}).set_placement(
                    (ang_x, ang_y + len(track_angles) * 7 + 3))
                for ai, ang in enumerate(track_angles[:5]):
                    desc = ang.get('description', '')
                    val = ang.get('angle_deg', 0)
                    msp.add_text(f"{desc}: {val:.1f}°", dxfattribs={
                        'height': th - 0.5, 'color': 4
                    }).set_placement((ang_x, ang_y + ai * 7))

            # ===== 儲存 DXF =====
            doc.saveas(output_path)

            log_print(f"\n[Success] 子系統施工圖已生成!")
            log_print(f"          路徑: {output_path}")
            log_print("=" * 60 + "\n")

            # ===== 生成 PNG 預覽 =====
            preview_path = self._render_dxf_preview(output_path, output_dir, base_name)
            if preview_path:
                log_print(f"[Preview] 預覽圖已生成: {preview_path}")

            return output_path

        except Exception as e:
            log_print(f"[Error] 生成子系統施工圖失敗: {e}", "error")
            import traceback
            log_print(traceback.format_exc(), "error")
            return None

    def _render_dxf_preview(self, dxf_path: str, output_dir: str, base_name: str) -> str:
        """
        將 DXF 渲染為 PNG 預覽圖
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle as MplCircle
            import matplotlib.font_manager as fm

            # 使用系統中文字型
            cjk_font = None
            for font_name in ['Microsoft JhengHei', 'Microsoft YaHei',
                               'SimHei', 'Noto Sans CJK TC']:
                matches = fm.findfont(fm.FontProperties(family=font_name),
                                      fallback_to_default=False)
                if matches and 'fallback' not in str(matches).lower():
                    cjk_font = font_name
                    break
            if cjk_font:
                plt.rcParams['font.sans-serif'] = [cjk_font, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False

            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            fig, ax = plt.subplots(1, 1, figsize=(16, 11.3), dpi=120)
            ax.set_aspect('equal')
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#0f0f23')

            color_map = {
                0: '#ffffff', 1: '#ff3333', 2: '#ffff33', 3: '#33ff33',
                4: '#33ffff', 5: '#3333ff', 6: '#ff33ff', 7: '#cccccc',
            }

            for entity in msp:
                etype = entity.dxftype()
                c = color_map.get(entity.dxf.get('color', 7), '#cccccc')
                lw = 0.5

                if etype == 'LINE':
                    ax.plot(
                        [entity.dxf.start.x, entity.dxf.end.x],
                        [entity.dxf.start.y, entity.dxf.end.y],
                        color=c, linewidth=lw)

                elif etype == 'LWPOLYLINE':
                    pts = list(entity.get_points(format='xy'))
                    if pts:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        ax.plot(xs, ys, color=c, linewidth=lw)

                elif etype == 'CIRCLE':
                    cx = entity.dxf.center.x
                    cy = entity.dxf.center.y
                    r = entity.dxf.radius
                    circle_patch = MplCircle(
                        (cx, cy), r, fill=False, edgecolor=c, linewidth=lw)
                    ax.add_patch(circle_patch)

                elif etype == 'TEXT':
                    insert = entity.dxf.insert
                    text_val = entity.dxf.text
                    h = entity.dxf.height
                    font_size = max(3, min(h * 1.2, 8))
                    font_props = {}
                    if cjk_font:
                        font_props['fontfamily'] = cjk_font
                    else:
                        font_props['fontfamily'] = 'monospace'
                    ax.text(insert.x, insert.y, text_val,
                            fontsize=font_size, color=c,
                            verticalalignment='bottom', **font_props)

            ax.autoscale()
            ax.margins(0.02)
            ax.set_title(f"{base_name} - 子系統施工圖", color='white',
                         fontsize=12, pad=10)
            ax.tick_params(colors='#555555', labelsize=6)

            preview_path = os.path.join(output_dir, f"{base_name}_子系統施工圖_preview.png")
            fig.savefig(preview_path, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            return preview_path

        except Exception as e:
            log_print(f"[Preview] 預覽生成失敗: {e}", "warning")
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

        # ----- 零件分類 -----
        lines.append("【零件分類 (Part Classifications)】")
        lines.append("-" * 80)
        part_cls = info.get("part_classifications", [])
        if part_cls:
            lines.append(f"    共 {len(part_cls)} 個零件分類")
            lines.append("")
            lines.append(f"    {'特徵ID':<10} {'分類':<15} {'信心度':<10} {'細長比':<10} {'體積'}")
            lines.append("    " + "-" * 65)
            for pc in part_cls:
                lines.append(
                    f"    {pc['feature_id']:<10} "
                    f"{pc['class_zh']:<13} "
                    f"{pc['confidence']:<10.2f} "
                    f"{pc['slenderness']:<10.1f} "
                    f"{pc['volume']:.2f}"
                )
        else:
            lines.append("    無資訊")
        lines.append("")

        # ----- 管路中心線 -----
        lines.append("【管路中心線 (Pipe Centerlines)】")
        lines.append("-" * 80)
        pipes = info.get("pipe_centerlines", [])
        if pipes:
            for p in pipes:
                method_tag = f" [{p.get('method', '')}]" if p.get('method') else ""
                lines.append(f"    實體: {p['solid_id']}{method_tag}")
                lines.append(f"        管徑: {p['pipe_diameter']:.2f} mm (R={p['pipe_radius']:.2f})")
                lines.append(f"        總長: {p['total_length']:.1f} mm")
                for j, seg in enumerate(p['segments'], 1):
                    if seg['type'] == 'straight':
                        lines.append(f"        段{j}: 直線 長度={seg['length']:.1f}")
                    elif seg['type'] == 'arc':
                        lines.append(
                            f"        段{j}: 弧線 角度={seg['angle_deg']}° "
                            f"R={seg.get('radius', 0):.0f} "
                            f"外弧長={seg.get('outer_arc_length', 0):.0f}"
                        )
                lines.append("")
        else:
            lines.append("    無資訊")
        lines.append("")

        # ----- 角度分析 -----
        lines.append("【角度分析 (Angle Analysis)】")
        lines.append("-" * 80)
        angle_list = info.get("angles", [])
        if angle_list:
            for a in angle_list:
                lines.append(
                    f"    {a['description']}: "
                    f"{a['part_a']} → {a.get('part_b', '-')} = {a['angle_deg']:.1f}°"
                )
        else:
            lines.append("    無資訊")
        lines.append("")

        # ----- 取料明細 -----
        lines.append("=" * 80)
        lines.append("軌道取料明細 (Cutting List)")
        lines.append("=" * 80)
        cutting = info.get("cutting_list", {})
        track_items = cutting.get("track_items", [])
        if track_items:
            lines.append(f"    {'球號':<8} {'取料尺寸 (mm)'}")
            lines.append("    " + "-" * 60)
            for item in track_items:
                lines.append(f"    {item['item']:<8} {item['spec']}")
        else:
            lines.append("    無軌道取料資料")
        lines.append("")

        leg_items = cutting.get("leg_items", [])
        if leg_items:
            lines.append("    腳架明細:")
            for item in leg_items:
                lines.append(f"    {item['item']}. {item['name']} x{item['quantity']} - {item['spec']}")
        lines.append("")

        bracket_items = cutting.get("bracket_items", [])
        if bracket_items:
            lines.append("    支撐架明細:")
            for item in bracket_items:
                lines.append(f"    {item['item']}. {item['name']} x{item['quantity']} - {item['spec']}")
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