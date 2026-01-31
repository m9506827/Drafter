"""
測試進階分析元件：管路中心線、零件分類、角度計算、取料明細
使用 1-2.stp 模型檔案進行測試
"""
import os
import sys

# 將上層目錄加入搜尋路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_drafter_system import MockCADEngine


def test_with_model():
    """使用 1-2.stp 模型測試所有進階分析元件"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "1-2.stp")
    if not os.path.exists(model_path):
        print(f"[SKIP] 找不到模型檔案: {model_path}")
        return False

    print(f"載入模型: {model_path}")
    engine = MockCADEngine(model_path)

    solid_features = [f for f in engine.features if f.type == "solid"]
    print(f"\n實體數量: {len(solid_features)}")

    # --- Component 2: Pipe Centerlines ---
    print("\n" + "=" * 60)
    print("Component 2: Pipe Centerline Extractor")
    print("=" * 60)
    pipes = engine._pipe_centerlines
    print(f"偵測到 {len(pipes)} 個管路")
    for p in pipes:
        print(f"  {p['solid_id']}: diameter={p['pipe_diameter']:.2f} mm, "
              f"total_length={p['total_length']:.1f} mm, "
              f"method={p.get('method', 'unknown')}")
        for i, seg in enumerate(p['segments']):
            if seg['type'] == 'straight':
                print(f"    seg{i+1}: straight L={seg['length']:.1f}")
            elif seg['type'] == 'arc':
                print(f"    seg{i+1}: arc angle={seg['angle_deg']}° "
                      f"R={seg.get('radius', 0):.0f}")

    # --- Component 1: Part Classifier ---
    print("\n" + "=" * 60)
    print("Component 1: Part Classifier")
    print("=" * 60)
    classifications = engine._part_classifications
    print(f"分類了 {len(classifications)} 個零件")

    class_counts = {}
    for c in classifications:
        cls = c['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
        print(f"  {c['feature_id']}: {c['class_zh']} ({c['class']}) "
              f"confidence={c['confidence']:.2f} slenderness={c['slenderness']:.1f}")

    print(f"\n分類統計: {class_counts}")

    # --- Component 3: Angle Calculator ---
    print("\n" + "=" * 60)
    print("Component 3: Angle Calculator")
    print("=" * 60)
    angles = engine._angles
    print(f"計算了 {len(angles)} 個角度")
    for a in angles:
        print(f"  {a['description']}: {a['part_a']} -> {a.get('part_b', '-')} = {a['angle_deg']:.1f}°")

    # --- Component 4: Cutting List ---
    print("\n" + "=" * 60)
    print("Component 4: Cutting List Generator")
    print("=" * 60)
    cutting = engine._cutting_list
    track_items = cutting.get('track_items', [])
    print(f"軌道項目: {len(track_items)}")
    for item in track_items:
        print(f"  {item['item']}: {item['spec']}")

    leg_items = cutting.get('leg_items', [])
    print(f"腳架項目: {len(leg_items)}")
    for item in leg_items:
        print(f"  {item['item']}. {item['name']} x{item['quantity']} - {item['spec']}")

    bracket_items = cutting.get('bracket_items', [])
    print(f"支撐架項目: {len(bracket_items)}")
    for item in bracket_items:
        print(f"  {item['item']}. {item['name']} x{item['quantity']} - {item['spec']}")

    # --- Integration: get_model_info ---
    print("\n" + "=" * 60)
    print("Integration: get_model_info()")
    print("=" * 60)
    info = engine.get_model_info()
    print(f"pipe_centerlines in info: {'pipe_centerlines' in info}")
    print(f"part_classifications in info: {'part_classifications' in info}")
    print(f"angles in info: {'angles' in info}")
    print(f"cutting_list in info: {'cutting_list' in info}")

    # --- Integration: save_info_to_file ---
    print("\n" + "=" * 60)
    print("Integration: save_info_to_file()")
    print("=" * 60)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    txt_path = engine.save_info_to_file(output_dir)
    if txt_path and os.path.exists(txt_path):
        print(f"報告已儲存: {txt_path}")
        # 驗證新區段存在
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        checks = [
            ("零件分類", "零件分類" in content),
            ("管路中心線", "管路中心線" in content),
            ("角度分析", "角度分析" in content),
            ("軌道取料明細", "軌道取料明細" in content),
        ]
        for name, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name} 區段存在")
    else:
        print("[FAIL] 報告儲存失敗")

    # --- Integration: generate_assembly_drawing ---
    print("\n" + "=" * 60)
    print("Integration: generate_assembly_drawing()")
    print("=" * 60)
    dxf_path = engine.generate_assembly_drawing(output_dir)
    if dxf_path and os.path.exists(dxf_path):
        print(f"組立圖已儲存: {dxf_path}")
        print("[PASS] DXF 生成成功")
    else:
        print("[FAIL] DXF 生成失敗")

    print("\n" + "=" * 60)
    print("測試完成")
    print("=" * 60)
    return True


def test_fallback():
    """測試無 3D 模型時的後備行為"""
    print("\n" + "=" * 60)
    print("Fallback Test: 無模型")
    print("=" * 60)

    engine = MockCADEngine()
    print(f"pipe_centerlines: {engine._pipe_centerlines}")
    print(f"part_classifications: {engine._part_classifications}")
    print(f"angles: {engine._angles}")
    print(f"cutting_list: {engine._cutting_list}")
    assert engine._pipe_centerlines == [], "Should be empty list"
    assert engine._part_classifications == [], "Should be empty list"
    assert engine._angles == [], "Should be empty list"
    assert engine._cutting_list == {}, "Should be empty dict"
    print("[PASS] 後備行為正確")


if __name__ == "__main__":
    test_fallback()
    test_with_model()
