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

def log_print(message: str, level: str = "info"):
    """
    同時輸出到控制台和日誌檔案
    Args:
        message: 要輸出的訊息
        level: 日誌級別 (debug, info, warning, error)
    """
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
                # 讀取 STEP 檔案
                self.cad_model = cq.importers.importStep(filename)
                log_print("[CAD Kernel] STEP file loaded successfully")
            elif file_ext == '.stl':
                # STL 是網格格式，需要轉換
                # 注意：STL 無法直接轉換為參數化特徵，這裡僅做基本處理
                log_print("[CAD Kernel] Warning: STL files are mesh-based, feature extraction may be limited", "warning")
                # 可以考慮使用 pyvista 讀取，但這裡先跳過
                raise NotImplementedError("STL file support requires additional processing")
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
    
    def _extract_features(self) -> List[GeometricFeature]:
        """
        從 CadQuery 3D 模型中提取幾何特徵
        這是一個簡化版本，實際應用中需要更複雜的幾何分析
        """
        features = []
        feature_id = 1
        
        if self.cad_model is None:
            return features
        
        try:
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
            
            # 嘗試提取圓形特徵 (孔洞)
            # 這是一個簡化版本，實際需要更複雜的幾何分析
            try:
                # 尋找圓形邊緣或面
                edges = self.cad_model.edges()
                circles = edges.filter(lambda e: e.geomType() == "CIRCLE")
                
                for i, circle_edge in enumerate(circles.objects):
                    # 嘗試獲取圓的半徑和中心
                    # 注意：這需要根據實際的 CadQuery API 調整
                    try:
                        # 簡化處理：假設找到的圓形特徵
                        # 實際應用中需要更精確的幾何分析
                        radius = 10.0  # 預設值，實際應從幾何中提取
                        features.append(GeometricFeature(
                            f"F{feature_id:02d}",
                            "circle",
                            {'radius': radius, 'x': center_x, 'y': center_y},
                            f"hole_{i+1}"
                        ))
                        feature_id += 1
                    except:
                        pass
            except Exception as e:
                log_print(f"[CAD Kernel] Warning: Could not extract circular features: {e}", "warning")
                # 添加一個預設的中心孔
                features.append(GeometricFeature(
                    f"F{feature_id:02d}",
                    "circle",
                    {'radius': 10.0, 'x': center_x, 'y': center_y},
                    "center_hole"
                ))
            
        except Exception as e:
            log_print(f"[CAD Kernel] Error extracting features: {e}", "error")
            # 返回基本特徵
            features = [
                GeometricFeature("F01", "rect", {'w': 100, 'h': 100, 'x': 0, 'y': 0}, "base_plate"),
                GeometricFeature("F02", "circle", {'radius': 10, 'x': 0, 'y': 0}, "center_hole")
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
        }

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

        # 生成 BOM (Bill of Materials)
        if info["solid_count"] > 0 or len(self.features) > 0:
            bom_entry = {
                "item": 1,
                "name": info["file_name"] or "未命名零件",
                "quantity": 1,
                "material": "未指定",
                "description": info["description"] or "3D 模型零件"
            }
            info["bom"].append(bom_entry)

        # 加入零件資訊
        if info["solid_count"] > 0:
            for i in range(info["solid_count"]):
                info["parts"].append({
                    "id": i + 1,
                    "name": f"Solid_{i + 1}",
                    "type": "實體"
                })
        elif len(self.features) > 0:
            for feat in info["features"]:
                info["parts"].append({
                    "id": feat["id"],
                    "name": feat["description"],
                    "type": feat["type"]
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

            # FILE_NAME 區段通常包含軟體資訊
            file_name_match = re.search(r"FILE_NAME\s*\([^)]*'([^']*)'[^)]*\)", content, re.IGNORECASE)
            if file_name_match:
                info["product_name"] = file_name_match.group(1) if file_name_match.group(1) else None

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
        lines.append("=" * 60)
        lines.append("圖檔資訊報告 (Model Information Report)")
        lines.append("=" * 60)
        lines.append("")

        # ----- 基本檔案資訊 -----
        lines.append("【基本檔案資訊】")
        lines.append("-" * 60)
        lines.append(f"    檔案路徑: {format_value(info.get('model_file'))}")
        lines.append(f"    檔案名稱: {format_value(info.get('file_name'))}")
        lines.append(f"    檔案格式: {format_value(info.get('file_extension'))}")
        lines.append(f"    檔案大小: {format_value(info.get('file_size'))}")
        lines.append(f"    3D 模型狀態: {'已載入' if info.get('has_model') else '未載入 (使用特徵資料)'}")
        lines.append("")

        # ----- 來源軟體與元資料 -----
        lines.append("【來源軟體與元資料】")
        lines.append("-" * 60)
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
        lines.append("-" * 60)
        lines.append(f"    長度單位: {format_value(info.get('units'))}")
        lines.append("")

        # ----- 幾何統計資訊 -----
        lines.append("【幾何統計資訊】")
        lines.append("-" * 60)
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
        lines.append("")

        # ----- 邊界框資訊 -----
        lines.append("【邊界框資訊 (Bounding Box)】")
        lines.append("-" * 60)
        bbox = info.get("bounding_box")
        if bbox:
            lines.append(f"    X 範圍: {bbox['x_min']:.2f} ~ {bbox['x_max']:.2f}")
            lines.append(f"    Y 範圍: {bbox['y_min']:.2f} ~ {bbox['y_max']:.2f}")
            lines.append(f"    Z 範圍: {bbox['z_min']:.2f} ~ {bbox['z_max']:.2f}")
            lines.append(f"    寬度 (Width): {bbox['width']:.2f}")
            lines.append(f"    高度 (Height): {bbox['height']:.2f}")
            lines.append(f"    深度 (Depth): {bbox['depth']:.2f}")
        else:
            lines.append(f"    邊界框: 無資訊")
        lines.append("")

        # ----- 零件列表 -----
        lines.append("【零件列表 (Parts)】")
        lines.append("-" * 60)
        parts = info.get("parts", [])
        if parts:
            for i, part in enumerate(parts, 1):
                lines.append(f"    [{i}] ID: {part.get('id')}")
                lines.append(f"        名稱: {part.get('name')}")
                lines.append(f"        類型: {part.get('type')}")
        else:
            lines.append("    無資訊")
        lines.append("")

        # ----- BOM (Bill of Materials) -----
        lines.append("【BOM 材料清單 (Bill of Materials)】")
        lines.append("-" * 60)
        bom = info.get("bom", [])
        if bom:
            lines.append("    項次 | 名稱                     | 數量 | 材料     | 描述")
            lines.append("    " + "-" * 56)
            for item in bom:
                name = str(item.get('name', '未命名'))[:24].ljust(24)
                qty = str(item.get('quantity', 1)).ljust(4)
                material = str(item.get('material', '未指定'))[:8].ljust(8)
                desc = str(item.get('description', ''))[:20]
                lines.append(f"    {item.get('item', 1):4d} | {name} | {qty} | {material} | {desc}")
        else:
            lines.append("    無資訊")
        lines.append("")

        # ----- 幾何特徵 -----
        lines.append("【幾何特徵 (Geometric Features)】")
        lines.append("-" * 60)
        features = info.get("features", [])
        if features:
            lines.append(f"    共 {len(features)} 個特徵")
            lines.append("")
            # 只顯示前 20 個特徵，避免檔案過大
            display_features = features[:20]
            for i, feat in enumerate(display_features, 1):
                lines.append(f"    [{i}] 特徵 ID: {feat['id']}")
                lines.append(f"        類型: {feat['type']}")
                lines.append(f"        描述: {feat['description']}")
                params_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in feat['params'].items()])
                lines.append(f"        參數: {params_str}")
                if i < len(display_features):
                    lines.append("")
            if len(features) > 20:
                lines.append(f"    ... 還有 {len(features) - 20} 個特徵未顯示")
        else:
            lines.append("    無資訊")
        lines.append("")

        lines.append("=" * 60)
        lines.append("報告結束")
        lines.append("=" * 60)

        # 寫入檔案
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            log_print(f"[System] 圖檔資訊已儲存至: {output_path}")
            return output_path
        except Exception as e:
            log_print(f"[System] 儲存圖檔資訊失敗: {e}", "error")
            return None

    def project_to_2d(self) -> List[GeometricFeature]:
        """
        真正的 3D 轉 2D 投影運算
        使用 CadQuery 將 3D 模型投影到 XY 平面（俯視圖）
        """
        log_print("[CAD Kernel] Projecting 3D model to 2D plane...")
        
        # 如果有 3D 模型，進行真正的投影
        if self.cad_model is not None and CADQUERY_AVAILABLE:
            try:
                # 方法：將 3D 模型投影到 XY 平面
                # 使用 CadQuery 的投影功能或直接從邊界框生成 2D 視圖
                
                # 獲取模型的邊界框
                bbox = self.cad_model.val().BoundingBox()
                
                # 生成 2D 投影特徵（俯視圖）
                features_2d = []
                feature_id = 1
                
                # 1. 添加外框（從邊界框生成）
                width = bbox.xmax - bbox.xmin
                height = bbox.ymax - bbox.ymin
                center_x = (bbox.xmin + bbox.xmax) / 2
                center_y = (bbox.ymin + bbox.ymax) / 2
                
                features_2d.append(GeometricFeature(
                    f"F{feature_id:02d}",
                    "rect",
                    {'w': width, 'h': height, 'x': center_x, 'y': center_y},
                    "projected_outline"
                ))
                feature_id += 1
                
                # 2. 嘗試從 3D 模型的邊緣提取圓形特徵
                # 這是一個簡化版本，實際應該分析所有邊緣
                try:
                    # 獲取所有邊緣
                    edges = self.cad_model.edges()
                    
                    # 尋找圓形邊緣
                    circle_edges = edges.filter(lambda e: e.geomType() == "CIRCLE")
                    
                    # 提取圓形特徵（投影到 XY 平面）
                    for i, edge in enumerate(circle_edges.objects):
                        try:
                            # 獲取邊緣的幾何資訊
                            # 注意：這需要根據實際的 CadQuery API 調整
                            # 簡化處理：使用邊界框中心附近的圓形
                            radius = 10.0  # 預設值
                            
                            # 嘗試從邊緣獲取更多資訊
                            # 這裡簡化處理，實際應該分析邊緣的實際幾何
                            
                            features_2d.append(GeometricFeature(
                                f"F{feature_id:02d}",
                                "circle",
                                {'radius': radius, 'x': center_x, 'y': center_y},
                                f"projected_circle_{i+1}"
                            ))
                            feature_id += 1
                            
                            # 限制提取的圓形數量，避免太多
                            if i >= 10:
                                break
                        except:
                            continue
                except Exception as e:
                    log_print(f"[CAD Kernel] Warning: Could not extract circles from edges: {e}", "warning")
                
                # 如果沒有提取到圓形，添加一個預設的中心圓
                if len([f for f in features_2d if f.type == 'circle']) == 0:
                    features_2d.append(GeometricFeature(
                        f"F{feature_id:02d}",
                        "circle",
                        {'radius': min(width, height) * 0.1, 'x': center_x, 'y': center_y},
                        "projected_center_feature"
                    ))
                
                log_print(f"[CAD Kernel] Generated {len(features_2d)} 2D projection features")
                return features_2d
                
            except Exception as e:
                log_print(f"[CAD Kernel] Error in 2D projection: {e}", "error")
                # 回退到原始特徵
                return self.features
        
        # 如果沒有 3D 模型或投影失敗，使用原始特徵
        return self.features

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
    log_print("\n請選擇 3D 模型檔案...")
    model_file = select_3d_file()

    if model_file:
        log_print(f"[System] 已選擇檔案: {model_file}")
    else:
        log_print("[System] 未選擇檔案，將使用模擬資料")

    # 初始化系統
    system = AutoDraftingSystem(model_file=model_file)

    # 讀取並完整顯示圖檔資訊（控制台簡要輸出）
    has_info = system.display_model_info()

    if not has_info:
        log_print("[System] 警告：無法取得圖檔資訊", "warning")

    # 將圖檔資訊儲存到 output 目錄
    info_file = system.save_info_to_file("output")

    # 開啟獨立視窗顯示完整資訊
    log_print("[System] 開啟圖檔資訊視窗...")
    log_print("請檢視資訊後關閉視窗繼續...")

    # 建立資訊視窗（使用 root 作為父視窗）
    info = system.cad.get_model_info()

    info_window = Toplevel(root)
    info_window.title("圖檔資訊 (Model Information)")
    info_window.geometry("700x800")
    info_window.configure(bg='#f0f0f0')
    info_window.attributes('-topmost', True)
    info_window.focus_force()

    # 建立主框架
    main_frame = Frame(info_window, bg='#f0f0f0')
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 標題
    title_label = Label(main_frame, text="圖檔完整資訊", font=('Microsoft JhengHei', 16, 'bold'),
                       bg='#f0f0f0', fg='#333333')
    title_label.pack(pady=(0, 10))

    # 建立文字區域框架
    text_frame = Frame(main_frame, bg='#ffffff', relief=tk.SUNKEN, bd=1)
    text_frame.pack(fill=tk.BOTH, expand=True)

    scrollbar = Scrollbar(text_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text_widget = Text(text_frame, wrap=tk.WORD, font=('Consolas', 10),
                      yscrollcommand=scrollbar.set, bg='#ffffff', fg='#333333',
                      padx=10, pady=10)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=text_widget.yview)

    # 填入資訊內容
    def format_value(value):
        if value is not None and value != "" and value != []:
            return str(value)
        return "無資訊"

    content = []
    content.append("=" * 60)
    content.append("圖檔資訊報告 (Model Information Report)")
    content.append("=" * 60)
    content.append("")
    content.append("【基本檔案資訊】")
    content.append("-" * 60)
    content.append(f"    檔案路徑: {format_value(info.get('model_file'))}")
    content.append(f"    檔案名稱: {format_value(info.get('file_name'))}")
    content.append(f"    檔案格式: {format_value(info.get('file_extension'))}")
    content.append(f"    檔案大小: {format_value(info.get('file_size'))}")
    content.append(f"    3D 模型狀態: {'已載入' if info.get('has_model') else '未載入'}")
    content.append("")
    content.append("【來源軟體與元資料】")
    content.append("-" * 60)
    content.append(f"    來源軟體: {format_value(info.get('source_software'))}")
    content.append(f"    作者: {format_value(info.get('author'))}")
    content.append(f"    組織: {format_value(info.get('organization'))}")
    content.append(f"    建立日期: {format_value(info.get('creation_date'))}")
    content.append(f"    產品名稱: {format_value(info.get('product_name'))}")
    content.append(f"    描述: {format_value(info.get('description'))}")
    content.append("")
    content.append("【單位資訊】")
    content.append("-" * 60)
    content.append(f"    長度單位: {format_value(info.get('units'))}")
    content.append("")
    content.append("【幾何統計資訊】")
    content.append("-" * 60)
    content.append(f"    實體數量: {info.get('solid_count', 0) or '無資訊'}")
    content.append(f"    面數量: {info.get('face_count', 0) or '無資訊'}")
    content.append(f"    邊數量: {info.get('edge_count', 0) or '無資訊'}")
    content.append(f"    頂點數量: {info.get('vertex_count', 0) or '無資訊'}")
    if info.get("volume"):
        content.append(f"    體積: {info['volume']:.4f}")
    if info.get("surface_area"):
        content.append(f"    表面積: {info['surface_area']:.4f}")
    content.append("")
    bbox = info.get("bounding_box")
    if bbox:
        content.append("【邊界框資訊】")
        content.append("-" * 60)
        content.append(f"    寬度: {bbox['width']:.2f}, 高度: {bbox['height']:.2f}, 深度: {bbox['depth']:.2f}")
        content.append("")
    content.append("【BOM 材料清單】")
    content.append("-" * 60)
    bom = info.get("bom", [])
    if bom:
        for item in bom:
            content.append(f"    {item.get('item', 1)}. {item.get('name', '未命名')} x {item.get('quantity', 1)}")
    else:
        content.append("    無資訊")
    content.append("")
    content.append("【幾何特徵】")
    content.append("-" * 60)
    features = info.get("features", [])
    content.append(f"    共 {len(features)} 個特徵")
    content.append("")
    content.append("=" * 60)

    text_widget.insert(tk.END, '\n'.join(content))
    text_widget.config(state=tk.DISABLED)

    # 關閉按鈕
    close_button = Button(main_frame, text="關閉視窗繼續", font=('Microsoft JhengHei', 10),
                         command=info_window.destroy, bg='#4a90d9', fg='white',
                         padx=20, pady=5)
    close_button.pack(pady=10)

    # 等待視窗關閉
    root.wait_window(info_window)

    log_print("[System] 圖檔資訊視窗已關閉，繼續執行...")

    # 繼續執行後續操作
    log_print("\n請輸入修改指令（或按 Enter 使用預設指令）:")
    try:
        user_input = input().strip()
    except (EOFError, KeyboardInterrupt):
        user_input = ""
        log_print("\n使用預設指令")

    if not user_input:
        user_input = "中間的孔太小了，請把它變成2倍大"
        log_print(f"使用預設指令: {user_input}")

    # 處理請求並生成 DXF 檔案
    output_filename = "output/modified_part.dxf"
    system.process_request(user_input, output_filename)

    # 記錄程式結束時間
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    log_print("\n" + "=" * 50)
    log_print("處理完成！")
    log_print(f"程式結束時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"總執行時間: {elapsed_time}")
    log_print("=" * 50)

    # 清理
    root.destroy()