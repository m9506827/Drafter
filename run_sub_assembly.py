"""
子系統施工圖生成腳本
用法: python run_sub_assembly.py [STEP檔案路徑]
預設: 1-2.stp
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from auto_drafter_system import MockCADEngine, log_print


def main():
    step_file = sys.argv[1] if len(sys.argv) > 1 else "1-2.stp"

    if not os.path.exists(step_file):
        print(f"[Error] 找不到檔案: {step_file}")
        return 1

    print(f"[System] 載入 {step_file} ...")
    engine = MockCADEngine()
    engine.load_3d_file(step_file)

    print(f"[System] 生成子系統施工圖 ...")
    result = engine.generate_sub_assembly_drawing("output")

    if result:
        print(f"\n[Done] DXF: {result}")
        # 檢查預覽
        base = os.path.splitext(os.path.basename(step_file))[0]
        preview = os.path.join("output", f"{base}_子系統施工圖_preview.png")
        if os.path.exists(preview):
            print(f"[Done] Preview: {preview}")
        return 0
    else:
        print("[Error] 生成失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
