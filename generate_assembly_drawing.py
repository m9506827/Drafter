# 生成組立施工圖
# Generate Assembly Drawing from STEP file

import os
import sys

# 添加當前目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_drafter_system import MockCADEngine, log_print
from simple_viewer import EngineeringViewer

def main():
    """
    主程式：載入 STEP 檔案，生成組立施工圖並顯示
    """
    # 設定輸入檔案
    # 可以修改為你的 STEP 檔案路徑
    step_file = r"D:\Google\Drafter\1-2.stp"

    # 檢查檔案是否存在
    if not os.path.exists(step_file):
        print(f"[Error] 找不到檔案: {step_file}")
        print("請修改 step_file 變數為正確的 STEP 檔案路徑")
        return

    print("=" * 60)
    print("組立施工圖生成器")
    print("Assembly Drawing Generator")
    print("=" * 60)
    print(f"\n輸入檔案: {step_file}")

    # 載入 CAD 模型
    print("\n[1/4] 載入 3D 模型...")
    cad = MockCADEngine(step_file)

    # 顯示模型資訊
    print("\n[2/4] 模型資訊:")
    info = cad.get_model_info()
    print(f"  - 產品名稱: {info.get('product_name', 'N/A')}")
    print(f"  - 來源軟體: {info.get('source_software', 'N/A')}")
    print(f"  - 實體數量: {info.get('solid_count', 0)}")
    print(f"  - 面數量: {info.get('face_count', 0)}")
    print(f"  - 邊數量: {info.get('edge_count', 0)}")

    bbox = info.get('bounding_box')
    if bbox:
        print(f"  - 尺寸: {bbox['width']:.2f} x {bbox['height']:.2f} x {bbox['depth']:.2f}")

    # 生成組立施工圖
    print("\n[3/4] 生成組立施工圖...")
    output_dir = "output"
    assembly_dxf = cad.generate_assembly_drawing(output_dir)

    if assembly_dxf:
        print(f"\n[Success] 組立施工圖已生成: {assembly_dxf}")

        # 同時生成資訊檔案
        print("\n[4/4] 儲存模型資訊...")
        info_file = cad.save_info_to_file(output_dir)
        if info_file:
            print(f"[Success] 模型資訊已儲存: {info_file}")

        # 顯示 DXF
        print("\n" + "=" * 60)
        print("正在開啟組立施工圖預覽...")
        print("=" * 60)

        try:
            EngineeringViewer.view_2d_dxf(assembly_dxf, fast_mode=True)
        except Exception as e:
            print(f"[Warning] 無法開啟預覽: {e}")
            print(f"請使用 AutoCAD 或其他 DXF 檢視器開啟: {assembly_dxf}")
    else:
        print("\n[Error] 組立施工圖生成失敗")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

    # 列出輸出檔案
    print(f"\n輸出目錄: {os.path.abspath(output_dir)}")
    if os.path.exists(output_dir):
        print("輸出檔案:")
        for f in os.listdir(output_dir):
            filepath = os.path.join(output_dir, f)
            size = os.path.getsize(filepath)
            print(f"  - {f} ({size:,} bytes)")

if __name__ == "__main__":
    main()
