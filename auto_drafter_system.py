import json
import math
import os
import subprocess
import platform
import ezdxf
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from tkinter import filedialog, Tk
from simple_viewer import EngineeringViewer

# 延遲導入 cadquery，避免啟動時的導入錯誤
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError as e:
    CADQUERY_AVAILABLE = False
    print(f"[Warning] CadQuery not available: {e}")
    print("[Warning] 3D file loading will be disabled. Using mock data only.")

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
            print("[CAD Kernel] Using mock data (no 3D file loaded)")
    
    def load_3d_file(self, filename: str):
        """
        從檔案讀取 3D 模型 (支援 STEP/STL)
        並提取幾何特徵
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"找不到檔案: {filename}")
        
        # 檢查 cadquery 是否可用
        if not CADQUERY_AVAILABLE:
            print(f"[CAD Kernel] Warning: CadQuery not available, cannot load 3D file")
            print(f"[CAD Kernel] Falling back to mock data")
            self.features = [
                GeometricFeature("F01", "rect", {'w': 100, 'h': 100, 'x': 0, 'y': 0}, "base_plate"),
                GeometricFeature("F02", "circle", {'radius': 10, 'x': 0, 'y': 0}, "center_hole")
            ]
            return
        
        file_ext = os.path.splitext(filename)[1].lower()
        print(f"[CAD Kernel] Loading 3D file: {filename} ({file_ext})")
        
        try:
            if file_ext in ['.step', '.stp']:
                # 讀取 STEP 檔案
                self.cad_model = cq.importers.importStep(filename)
                print(f"[CAD Kernel] STEP file loaded successfully")
            elif file_ext == '.stl':
                # STL 是網格格式，需要轉換
                # 注意：STL 無法直接轉換為參數化特徵，這裡僅做基本處理
                print(f"[CAD Kernel] Warning: STL files are mesh-based, feature extraction may be limited")
                # 可以考慮使用 pyvista 讀取，但這裡先跳過
                raise NotImplementedError("STL file support requires additional processing")
            else:
                raise ValueError(f"不支援的檔案格式: {file_ext}")
            
            # 從 3D 模型中提取特徵
            self.features = self._extract_features()
            print(f"[CAD Kernel] Extracted {len(self.features)} features from 3D model")
            
        except Exception as e:
            print(f"[CAD Kernel] Error loading file: {e}")
            print(f"[CAD Kernel] Falling back to mock data")
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
                print(f"[CAD Kernel] Warning: Could not extract circular features: {e}")
                # 添加一個預設的中心孔
                features.append(GeometricFeature(
                    f"F{feature_id:02d}",
                    "circle",
                    {'radius': 10.0, 'x': center_x, 'y': center_y},
                    "center_hole"
                ))
            
        except Exception as e:
            print(f"[CAD Kernel] Error extracting features: {e}")
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
                
                print(f"[CAD Kernel] Modified {f.description}: {parameter} -> {f.params[parameter]}")
                return True
        return False

    def project_to_2d(self) -> List[GeometricFeature]:
        """
        真正的 3D 轉 2D 投影運算
        使用 CadQuery 將 3D 模型投影到 XY 平面（俯視圖）
        改進版本：直接使用 CadQuery 導出 DXF，然後從 DXF 中提取特徵
        """
        print("[CAD Kernel] Projecting 3D model to 2D plane...")
        
        # 如果有 3D 模型，進行真正的投影
        if self.cad_model is not None and CADQUERY_AVAILABLE:
            try:
                import tempfile
                
                # 方法：使用 CadQuery 直接導出 DXF，然後從 DXF 中提取特徵
                # 這樣可以保留更多的幾何細節
                with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
                    temp_dxf = tmp_file.name
                
                try:
                    # 使用 CadQuery 導出 DXF（會自動投影到 XY 平面）
                    print(f"[CAD Kernel] Exporting 3D model to temporary DXF...")
                    print(f"[CAD Kernel] Temporary DXF path: {temp_dxf}")
                    cq.exporters.export(self.cad_model, temp_dxf, "DXF")
                    print(f"[CAD Kernel] DXF export completed")
                    
                    # 確認臨時檔案是否存在
                    if not os.path.exists(temp_dxf):
                        print(f"[CAD Kernel] Error: Temporary DXF file was not created")
                        return self._project_from_bbox()
                    
                    file_size = os.path.getsize(temp_dxf)
                    print(f"[CAD Kernel] Temporary DXF file size: {file_size} bytes")
                    
                    # 從導出的 DXF 中讀取特徵
                    print(f"[CAD Kernel] Extracting features from DXF...")
                    features_2d = self._extract_features_from_dxf(temp_dxf)
                    print(f"[CAD Kernel] Extracted {len(features_2d)} features from DXF")
                    
                    # 清理臨時檔案
                    if os.path.exists(temp_dxf):
                        os.unlink(temp_dxf)
                    
                    if features_2d:
                        print(f"[CAD Kernel] Generated {len(features_2d)} 2D projection features from DXF")
                        return features_2d
                    else:
                        print(f"[CAD Kernel] Warning: No features extracted from DXF, using fallback method")
                        # 回退到邊界框方法
                        return self._project_from_bbox()
                        
                except Exception as e:
                    print(f"[CAD Kernel] Error exporting to DXF: {e}")
                    # 清理臨時檔案
                    if os.path.exists(temp_dxf):
                        os.unlink(temp_dxf)
                    # 回退到邊界框方法
                    return self._project_from_bbox()
                
            except Exception as e:
                print(f"[CAD Kernel] Error in 2D projection: {e}")
                import traceback
                traceback.print_exc()
                # 回退到邊界框方法
                return self._project_from_bbox()
        
        # 如果沒有 3D 模型或投影失敗，使用原始特徵
        return self.features
    
    def _project_from_bbox(self) -> List[GeometricFeature]:
        """
        從邊界框生成基本的 2D 投影（備用方法）
        """
        features_2d = []
        feature_id = 1
        
        try:
            bbox = self.cad_model.val().BoundingBox()
            width = bbox.xmax - bbox.xmin
            height = bbox.ymax - bbox.ymin
            center_x = (bbox.xmin + bbox.xmax) / 2
            center_y = (bbox.ymin + bbox.ymax) / 2
            
            # 添加外框
            features_2d.append(GeometricFeature(
                f"F{feature_id:02d}",
                "rect",
                {'w': width, 'h': height, 'x': center_x, 'y': center_y},
                "projected_outline"
            ))
        except:
            pass
        
        return features_2d if features_2d else self.features
    
    def _extract_features_from_dxf(self, dxf_file: str) -> List[GeometricFeature]:
        """
        從 DXF 檔案中提取幾何特徵
        """
        features = []
        feature_id = 1
        
        try:
            doc = ezdxf.readfile(dxf_file)
            msp = doc.modelspace()
            
            # 提取所有實體
            for entity in msp:
                try:
                    if entity.dxftype() == 'CIRCLE':
                        center = entity.dxf.center
                        radius = entity.dxf.radius
                        features.append(GeometricFeature(
                            f"F{feature_id:02d}",
                            "circle",
                            {'radius': float(radius), 'x': float(center.x), 'y': float(center.y)},
                            f"circle_{feature_id}"
                        ))
                        feature_id += 1
                    elif entity.dxftype() == 'LWPOLYLINE':
                        # 提取多段線的邊界框作為矩形
                        vertices = list(entity.vertices())
                        if len(vertices) >= 2:
                            x_coords = [float(v[0]) for v in vertices]
                            y_coords = [float(v[1]) for v in vertices]
                            xmin, xmax = min(x_coords), max(x_coords)
                            ymin, ymax = min(y_coords), max(y_coords)
                            width = xmax - xmin
                            height = ymax - ymin
                            center_x = (xmin + xmax) / 2
                            center_y = (ymin + ymax) / 2
                            
                            # 如果寬高比接近 1:1，可能是正方形
                            if abs(width - height) < max(width, height) * 0.1:
                                features.append(GeometricFeature(
                                    f"F{feature_id:02d}",
                                    "rect",
                                    {'w': width, 'h': height, 'x': center_x, 'y': center_y},
                                    f"polyline_rect_{feature_id}"
                                ))
                                feature_id += 1
                    elif entity.dxftype() == 'LINE':
                        # 線條暫時跳過，或可以轉換為多段線
                        pass
                except Exception as e:
                    # 跳過無法處理的實體
                    continue
            
            print(f"[CAD Kernel] Extracted {len(features)} features from DXF")
            return features
            
        except Exception as e:
            print(f"[CAD Kernel] Error reading DXF: {e}")
            return []

# ==========================================
# 2. AI 語意解析層 (The "Brain")
# ==========================================

class AIIntentParser:
    """
    負責將自然語言轉換為 CAD 操作指令
    實際應用中，這裡會呼叫 OpenAI API
    """
    def parse_instruction(self, user_prompt: str, context_features: List[GeometricFeature]) -> dict:
        print(f"[AI Agent] Analyzing prompt: '{user_prompt}'")
        
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

    def process_request(self, user_prompt: str, output_filename: Optional[str] = None):
        """
        處理用戶請求，生成 2D 工程圖
        Args:
            user_prompt: 用戶指令
            output_filename: 輸出檔案名稱，如果為 None 則根據 3D 檔案名稱自動生成
        """
        # 1. 如果載入了 3D 檔案，先顯示 3D 預覽
        if self.cad.model_file and os.path.exists(self.cad.model_file):
            print(f"[System] 顯示 3D 模型預覽...")
            EngineeringViewer.view_3d_stl(self.cad.model_file)
        
        # 2. AI 解析
        instruction = self.ai.parse_instruction(user_prompt, self.cad.features)
        
        if "error" in instruction:
            print(f"Error: {instruction['error']}")
            return

        # 3. 執行修改
        print(f"Executing: {instruction}")
        self.cad.modify_feature(
            instruction["target_id"], 
            instruction["parameter"], 
            instruction["value"], 
            instruction["operation"]
        )

        # 4. 生成 2D 視圖
        projection_data = self.cad.project_to_2d()

        # 5. 確定輸出檔案名稱
        if output_filename is None:
            if self.cad.model_file:
                # 使用原檔名，但改為 .dxf 副檔名
                base_name = os.path.splitext(os.path.basename(self.cad.model_file))[0]
                output_dir = "output"
                # 確保 output 目錄存在
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"[System] 創建輸出目錄: {output_dir}")
                output_filename = os.path.join(output_dir, f"{base_name}.dxf")
            else:
                output_filename = "output.dxf"
        
        # 6. 輸出 DXF 圖檔
        self._export_dxf(projection_data, output_filename)
        
        # 7. 顯示 2D 工程圖
        print(f"[System] 顯示 2D 工程圖...")
        self._open_file(output_filename)

    def _export_dxf(self, features: List[GeometricFeature], filename: str):
        """使用 ezdxf 產生真實的 CAD 檔案"""
        doc = ezdxf.new()
        msp = doc.modelspace()

        print(f"[DXF Gen] Drawing {len(features)} entities...")
        
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
        print(f"[Success] Saved drawing to {filename}")
    
    def _open_file(self, filename: str):
        """
        自動開啟生成的檔案
        使用 EngineeringViewer 來檢視 DXF 檔案
        其他格式使用系統預設程式開啟
        """
        if not os.path.exists(filename):
            print(f"[Warning] File not found: {filename}")
            return
        
        file_ext = os.path.splitext(filename)[1].lower()
        
        try:
            # 如果是 DXF/DWG 檔案，使用 EngineeringViewer 開啟
            if file_ext in ['.dxf', '.dwg']:
                print(f"[System] Opening {file_ext.upper()} file with EngineeringViewer: {filename}")
                EngineeringViewer.view_2d_dxf(filename)
            # 如果是 STL 檔案，使用 3D 檢視器
            elif file_ext == '.stl':
                print(f"[System] Opening STL file with EngineeringViewer: {filename}")
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
                print(f"[System] Opened file with system default: {filename}")
        except Exception as e:
            print(f"[Warning] Could not open file: {e}")

# ==========================================
# 4. 輔助函數：檔案選擇對話框
# ==========================================

def select_3d_file() -> Optional[str]:
    """
    開啟檔案選擇對話框，讓使用者選擇 3D 模型檔案
    返回選取的檔案路徑，如果取消則返回 None
    """
    root = None
    try:
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
        
        if file_path:
            return file_path
        return None
    except KeyboardInterrupt:
        print("\n[System] 使用者中斷檔案選擇")
        return None
    except Exception as e:
        print(f"[Error] 檔案選擇對話框錯誤: {e}")
        return None
    finally:
        # 確保視窗被正確關閉
        if root is not None:
            try:
                root.destroy()
            except:
                pass

# ==========================================
# 5. 執行範例
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("自動繪圖系統 (Auto Drafter System)")
    print("=" * 50)
    
    # 讓使用者選擇 3D 模型檔案
    print("\n請選擇 3D 模型檔案...")
    model_file = select_3d_file()
    
    if model_file:
        print(f"[System] 已選擇檔案: {model_file}")
    else:
        print("[System] 未選擇檔案，將使用模擬資料")
    
    # 初始化系統
    system = AutoDraftingSystem(model_file=model_file)
    
    # ==========================================
    # 修改指令部分（已註解，直接轉換 3D 到 2D）
    # ==========================================
    # 使用者輸入（可以改為從命令列參數或輸入取得）
    # print("\n請輸入修改指令（或按 Enter 使用預設指令）:")
    # try:
    #     user_input = input().strip()
    # except (EOFError, KeyboardInterrupt):
    #     # 非互動模式或使用者取消，使用預設指令
    #     user_input = ""
    #     print("\n使用預設指令")
    # 
    # if not user_input:
    #     # 預設指令
    #     user_input = "中間的孔太小了，請把它變成2倍大"
    #     print(f"使用預設指令: {user_input}")
    # 
    # # 處理請求並生成 DXF 檔案（output_filename 為 None，會自動根據 3D 檔案名稱生成）
    # system.process_request(user_input, output_filename=None)
    
    # ==========================================
    # 直接轉換 3D 到 2D（不修改原圖內容）
    # ==========================================
    if model_file:
        # 1. 先顯示 3D 模型預覽
        print(f"\n[System] 顯示 3D 模型預覽...")
        EngineeringViewer.view_3d_stl(model_file)
        
        # 2. 直接轉換 3D 模型到 2D 工程圖（不修改原圖內容）
        print(f"\n[System] 直接轉換 3D 模型到 2D 工程圖（不修改原圖）...")
        
        # 3. 確定輸出檔案名稱（儲存到 output 目錄）
        base_name = os.path.splitext(os.path.basename(model_file))[0]
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"[System] 創建輸出目錄: {output_dir}")
        output_filename = os.path.join(output_dir, f"{base_name}.dxf")
        print(f"[System] 輸出檔案: {output_filename}")
        
        # 4. 直接使用 CadQuery 導出 DXF（最快最簡單的方法）
        if CADQUERY_AVAILABLE and system.cad.cad_model is not None:
            print(f"[System] 使用 CadQuery 直接導出 DXF...")
            try:
                cq.exporters.export(system.cad.cad_model, output_filename, "DXF")
                print(f"[Success] DXF 檔案已直接導出")
                
                # 確認檔案是否成功建立
                if os.path.exists(output_filename):
                    file_size = os.path.getsize(output_filename)
                    print(f"[Success] DXF 檔案已成功儲存: {output_filename} ({file_size} bytes)")
                    
                    # 5. 顯示 2D 工程圖
                    print(f"[System] 顯示 2D 工程圖...")
                    try:
                        system._open_file(output_filename)
                        print(f"[Success] 2D 工程圖視窗已開啟")
                    except Exception as e:
                        print(f"[Error] 顯示 2D 工程圖時發生錯誤: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[Error] DXF 檔案導出失敗，檔案不存在: {output_filename}")
            except Exception as e:
                print(f"[Error] 直接導出 DXF 失敗: {e}")
                import traceback
                traceback.print_exc()
                print(f"[System] 嘗試使用特徵提取方法...")
                # 回退到特徵提取方法
                try:
                    projection_data = system.cad.project_to_2d()
                    print(f"[System] 2D 投影完成，提取到 {len(projection_data)} 個特徵")
                    
                    if projection_data and len(projection_data) > 0:
                        # 輸出 DXF 圖檔
                        print(f"[System] 正在儲存 DXF 檔案到: {output_filename}")
                        system._export_dxf(projection_data, output_filename)
                        
                        # 確認檔案是否成功建立
                        if os.path.exists(output_filename):
                            file_size = os.path.getsize(output_filename)
                            print(f"[Success] DXF 檔案已成功儲存: {output_filename} ({file_size} bytes)")
                            
                            # 顯示 2D 工程圖
                            print(f"[System] 顯示 2D 工程圖...")
                            try:
                                system._open_file(output_filename)
                                print(f"[Success] 2D 工程圖視窗已開啟")
                            except Exception as e2:
                                print(f"[Error] 顯示 2D 工程圖時發生錯誤: {e2}")
                                import traceback
                                traceback.print_exc()
                    else:
                        print(f"[Error] 無法提取 2D 特徵，投影資料為空")
                except Exception as e2:
                    print(f"[Error] 特徵提取方法也失敗: {e2}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"[Error] CadQuery 不可用或沒有載入 3D 模型")
    else:
        print("[System] 未選擇 3D 檔案，無法轉換")
    
    print("\n" + "=" * 50)
    print("處理完成！")
    print("=" * 50)