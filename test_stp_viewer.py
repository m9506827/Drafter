#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試 STP 檔案讀取和顯示
"""

import os
import sys
from simple_viewer import EngineeringViewer

def test_stp_file(filename):
    """
    測試 STP 檔案的讀取和顯示
    """
    if not os.path.exists(filename):
        print(f"錯誤：找不到檔案 {filename}")
        return False
    
    print("=" * 60)
    print(f"測試 STP 檔案: {filename}")
    print("=" * 60)
    
    # 檢查檔案大小
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    print(f"檔案大小: {file_size:.2f} MB")
    
    # 檢查 CadQuery 是否可用
    try:
        import cadquery as cq
        print("[OK] CadQuery 可用")
        
        # 嘗試讀取 STEP 檔案（不顯示，只驗證）
        print("正在驗證 STEP 檔案格式...")
        step_model = cq.importers.importStep(filename)
        print("[OK] STEP 檔案格式正確，可以讀取")
        
        # 獲取模型資訊
        try:
            bbox = step_model.val().BoundingBox()
            print(f"模型邊界框:")
            print(f"  X: [{bbox.xmin:.2f}, {bbox.xmax:.2f}]")
            print(f"  Y: [{bbox.ymin:.2f}, {bbox.ymax:.2f}]")
            print(f"  Z: [{bbox.zmin:.2f}, {bbox.zmax:.2f}]")
            print(f"  尺寸: {bbox.xmax - bbox.xmin:.2f} x {bbox.ymax - bbox.ymin:.2f} x {bbox.zmax - bbox.zmin:.2f}")
        except:
            print("[INFO] 無法獲取模型邊界框資訊")
        
    except ImportError:
        print("[ERROR] CadQuery 不可用，無法讀取 STEP 檔案")
        return False
    except Exception as e:
        print(f"[ERROR] 讀取 STEP 檔案時發生錯誤: {e}")
        return False
    
    # 使用 EngineeringViewer 查看
    print("\n正在開啟 3D 檢視器...")
    try:
        viewer = EngineeringViewer()
        viewer.view_3d_stl(filename)  # view_3d_stl 現在支援 STP 檔案
        print("\n[OK] 視窗已開啟")
        return True
    except Exception as e:
        print(f"\n[ERROR] 顯示錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 預設測試檔案
    test_file = "1-2.stp"
    
    # 如果命令列有提供檔案，使用該檔案
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    print(f"測試檔案: {test_file}")
    print()
    
    success = test_stp_file(test_file)
    
    if success:
        print("\n" + "=" * 60)
        print("測試完成！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("測試失敗！")
        print("=" * 60)
        sys.exit(1)
