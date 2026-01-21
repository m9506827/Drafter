import cadquery as cq
import os

# ==========================================
# 3D 機械圖產生器：標準法蘭盤 (Standard Flange)
# ==========================================

class FlangeGenerator:
    """
    生成參數化的法蘭盤 3D 模型
    用途：作為 3D 轉 2D 系統的標準測試輸入 (Test Input)
    """
    def __init__(self):
        # 預設參數 (這些是您的 AI 系統未來要負責修改的數值)
        self.base_diameter = 100.0  # 底座直徑
        self.base_thickness = 10.0  # 底座厚度
        self.hub_diameter = 50.0    # 軸頸直徑
        self.hub_height = 20.0      # 軸頸高度 (從底座表面算起)
        self.bore_diameter = 25.0   # 中心孔直徑
        self.bolt_circle_dia = 75.0 # 螺栓孔節圓直徑
        self.bolt_hole_dia = 10.0   # 螺栓孔直徑
        self.num_holes = 4          # 孔數

    def build(self):
        """
        執行幾何建模邏輯
        """
        print(f"Generating Flange: D{self.base_diameter} / Hub{self.hub_diameter} / {self.num_holes} Holes")
        
        # 1. 建立底座 (Base Plate)
        # Workplane("XY") 相當於在從上往下看的平面開始繪圖
        result = (
            cq.Workplane("XY")
            .circle(self.base_diameter / 2)
            .extrude(self.base_thickness)
        )
        
        # 2. 建立軸頸 (Hub)
        # faces(">Z") 選擇物體最頂部的面作為新的繪圖平面
        result = (
            result.faces(">Z")
            .workplane()
            .circle(self.hub_diameter / 2)
            .extrude(self.hub_height)
        )
        
        # 3. 切割中心孔 (Center Bore)
        # cutThruAll() 貫穿所有物體
        result = (
            result.faces(">Z")
            .workplane()
            .circle(self.bore_diameter / 2)
            .cutThruAll()
        )
        
        # 4. 切割螺栓孔陣列 (Bolt Pattern)
        # 使用 polarArray 進行環狀陣列
        result = (
            result.faces("<Z") # 選擇底部面 (比較好定位)
            .workplane()
            .polarArray(
                radius=self.bolt_circle_dia / 2, 
                startAngle=0, 
                angle=360, 
                count=self.num_holes
            )
            .circle(self.bolt_hole_dia / 2)
            .cutThruAll()
        )
        
        return result

    def export(self, shape, filename_base="test_flange"):
        """輸出不同格式供測試"""
        # STEP: 工業標準格式 (包含幾何特徵，適合轉 2D 工程圖)
        cq.exporters.export(shape, f"{filename_base}.step")
        
        # STL: 網格格式 (適合 3D 列印或快速預覽，但不適合精確量測)
        cq.exporters.export(shape, f"{filename_base}.stl")
        
        print(f"[Success] Exported to {filename_base}.step and .stl")

# ==========================================
# 執行腳本
# ==========================================
if __name__ == "__main__":
    try:
        # 實例化產生器
        generator = FlangeGenerator()
        
        # 建立模型
        flange_model = generator.build()
        
        # 匯出檔案
        # 您的系統可以讀取這個 .step 檔來測試投影功能
        generator.export(flange_model)
        
    except ImportError:
        print("錯誤：請先安裝 CadQuery 函式庫 (pip install cadquery)")
    except Exception as e:
        print(f"發生錯誤: {e}")