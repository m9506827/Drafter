"""
子系統施工圖生成腳本
用法: python run_sub_assembly.py [STEP檔案路徑]
預設: 1-2.stp
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from auto_drafter_system import MockCADEngine, log_print
from simple_viewer import EngineeringViewer


def main():
    step_file = sys.argv[1] if len(sys.argv) > 1 else "1-2.stp"

    if not os.path.exists(step_file):
        print(f"[Error] 找不到檔案: {step_file}")
        return 1

    print(f"[System] 載入 {step_file} ...")
    engine = MockCADEngine()
    engine.load_3d_file(step_file)

    output_dir = "output"

    # 1) 儲存模型資訊
    print(f"[System] 儲存模型資訊 ...")
    info_path = engine.save_info_to_file(output_dir)
    if info_path:
        print(f"  [OK] {info_path}")

    # 2) 生成投影圖 (XY/XZ/YZ + rot)
    print(f"[System] 生成投影圖 ...")
    proj_paths = engine.export_projections_to_dxf(output_dir)
    for p in proj_paths:
        print(f"  [OK] {p}")

    # 3) 生成子系統施工圖 (3 張)
    print(f"[System] 生成子系統施工圖 ...")
    result = engine.generate_sub_assembly_drawing(output_dir)

    if result:
        print(f"\n[Done] 共 {len(result)} 張施工圖:")
        for i, dxf_path in enumerate(result, 1):
            print(f"  [{i}] DXF: {dxf_path}")

        # 使用程式互動視窗預覽每張施工圖
        for dxf_path in result:
            if os.path.exists(dxf_path):
                try:
                    print(f"  [Preview] 開啟預覽: {dxf_path}")
                    EngineeringViewer.view_2d_dxf(dxf_path, fast_mode=True)
                except Exception as pve:
                    print(f"  [Preview] 預覽失敗: {pve}")
        return 0
    else:
        print("[Error] 生成失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
