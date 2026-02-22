#!/usr/bin/env python3
"""
自動化驗證腳本 — 驗證 auto_drafter_system.py 針對 2-4.stp 的產出
=================================================================
用法:
    py test/verify_values_24.py              # 執行完整驗證（含生成圖面）
    py test/verify_values_24.py --quick      # 只驗證數值，不生成圖面
    py test/verify_values_24.py --draw2      # 只驗 Drawing 2
    py test/verify_values_24.py --draw3      # 只驗 Drawing 3

2-4.stp 特性：由下至上軌道右彎 (bend_direction='right')
  Draw 1（第一段）：腳架×1 + 支撐架×1 / 上×1 直 / 下×1 直
  Draw 2（彎軌）  ：支撐架×5 / 上×1 彎 / 下×1 彎
  Draw 3（第二段）：腳架×2 / 上×3 直彎直 / 下×3 直彎直

每次修改 auto_drafter_system.py 後必須執行此腳本。
所有斷言失敗都會彙報，不會在第一個錯誤就中斷。
"""
import sys
import os
import math
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.environ['DRAFTER_NO_GUI'] = '1'

MODEL_FILE = os.path.join(ROOT_DIR, 'test', '2-4.stp')

# =====================================================================
# 期望值基準（來自 2-4.stp 的標準圖面）
# =====================================================================
EXPECTED = {
    # --- 弧段參數（Draw 2 核心） ---
    'arc_radius':       260,
    'arc_angle_deg':    180.0,
    'arc_height_gain':  297.3,
    'arc_cl_length':    869,       # 允許 ±1
    'arc_3d_endpoint':  599,       # 允許 ±2

    # --- 管徑 ---
    'pipe_diameter':    48.1,

    # --- 軌道間距（彎軌區段，gn=(0,0,1) 時取 Z diff） ---
    'curved_rail_spacing': 251.1,

    # --- 腳架長度（個位數四捨五入到10位） ---
    'leg_lengths':      [460, 510, 450],

    # --- 支撐架（Draw1×1 + Draw2×5 = 總計 6） ---
    'bracket_count':    6,

    # --- 彎曲方向 ---
    'bend_direction':   'right',

    # --- 仰角（進入 20°、轉換角 15°、離開 35°） ---
    'entry_elevation_deg':      20.0,
    'transition_angle_deg':     15.0,
    'exit_elevation_deg':       35.0,

    # --- 弦長（2R × sin(θ/2)，θ=180°） ---
    'chord_length':     520,
}


# =====================================================================
# 測試框架（與 verify_values.py 相同）
# =====================================================================
class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, actual, expected, tolerance=0):
        if tolerance > 0:
            ok = abs(actual - expected) <= tolerance
        else:
            ok = actual == expected
        if ok:
            self.passed += 1
            print(f"  [PASS] {name}: {actual}")
        else:
            self.failed += 1
            msg = f"  [FAIL] {name}: got {actual}, expected {expected}"
            if tolerance > 0:
                msg += f" (tol={tolerance})"
            print(msg)
            self.errors.append(msg)

    def check_in(self, name, actual, expected_list):
        if actual in expected_list:
            self.passed += 1
            print(f"  [PASS] {name}: {actual}")
        else:
            self.failed += 1
            msg = f"  [FAIL] {name}: got {actual}, expected one of {expected_list}"
            print(msg)
            self.errors.append(msg)

    def check_true(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            msg = f"  [FAIL] {name}{': ' + detail if detail else ''}"
            print(msg)
            self.errors.append(msg)

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        if self.failed == 0:
            print(f"ALL PASSED: {self.passed}/{total} checks")
        else:
            print(f"FAILED: {self.failed}/{total} checks")
            for e in self.errors:
                print(e)
        print(f"{'='*60}")
        return self.failed == 0


# =====================================================================
# 測試群組
# =====================================================================

def test_pipe_centerlines(engine, T):
    """測試管路中心線弧段擷取"""
    print("\n--- Test: Pipe Centerlines (Arc Extraction) ---")

    arc_pipes = []
    for pc in engine._pipe_centerlines:
        for seg in pc.get('segments', []):
            if seg.get('type') == 'arc' and seg.get('radius', 0) > 50:
                arc_pipes.append((pc, seg))

    T.check_true("Found arc pipes", len(arc_pipes) >= 2,
                 f"found {len(arc_pipes)}")

    for pc, seg in arc_pipes:
        fid = pc['solid_id']
        T.check(f"{fid} R", seg['radius'], EXPECTED['arc_radius'])
        T.check(f"{fid} angle", seg['angle_deg'], EXPECTED['arc_angle_deg'])
        T.check(f"{fid} h_gain", seg['height_gain'], EXPECTED['arc_height_gain'],
                tolerance=0.1)
        T.check(f"{fid} arc_length", seg['arc_length'], EXPECTED['arc_cl_length'],
                tolerance=1)

        sp = pc.get('start_point', (0, 0, 0))
        ep = pc.get('end_point', (0, 0, 0))
        d3 = math.sqrt(sum((a - b) ** 2 for a, b in zip(sp, ep)))
        T.check(f"{fid} 3D_dist", round(d3), EXPECTED['arc_3d_endpoint'],
                tolerance=2)


def test_cutting_list(engine, T):
    """測試取料明細"""
    print("\n--- Test: Cutting List ---")

    cl = engine._cutting_list
    T.check_true("Cutting list exists", cl is not None)

    arc_items = [t for t in cl.get('track_items', []) if t.get('type') == 'arc']
    T.check_true("Arc track items >= 2", len(arc_items) >= 2,
                 f"found {len(arc_items)}")

    for ai in arc_items:
        T.check(f"{ai['item']} R", ai.get('radius', 0), EXPECTED['arc_radius'])
        T.check(f"{ai['item']} angle", ai.get('angle_deg', 0),
                EXPECTED['arc_angle_deg'])
        T.check(f"{ai['item']} h_gain", ai.get('height_gain', 0),
                EXPECTED['arc_height_gain'], tolerance=0.1)

    leg_items = cl.get('leg_items', [])
    T.check("Leg count", len(leg_items), len(EXPECTED['leg_lengths']))
    for i, li in enumerate(leg_items):
        exp_len = EXPECTED['leg_lengths'][i] if i < len(EXPECTED['leg_lengths']) else 0
        T.check(f"Leg {i+1} length", li.get('line_length', 0), exp_len)

    bracket_items = cl.get('bracket_items', [])
    total_brackets = sum(b.get('quantity', 1) for b in bracket_items)
    T.check("Bracket count (Draw1×1 + Draw2×5)", total_brackets,
            EXPECTED['bracket_count'])


def test_rail_spacing(engine, T):
    """測試彎軌區段軌道間距（2-4.stp: gn=(0,0,1)，取 Z diff）"""
    print("\n--- Test: Rail Spacing (Curved Section) ---")

    info = engine.get_model_info()
    part_cls = info.get('part_classifications', [])
    class_map = {c['feature_id']: c for c in part_cls}

    # 找弧管特徵 ID
    arc_fids = []
    for pc in engine._pipe_centerlines:
        for seg in pc.get('segments', []):
            if seg.get('type') == 'arc' and seg.get('radius', 0) > 50:
                arc_fids.append(pc['solid_id'])

    T.check_true("Found 2 arc track features", len(arc_fids) >= 2,
                 f"found {len(arc_fids)}")

    if len(arc_fids) >= 2:
        c1 = class_map.get(arc_fids[0], {}).get('centroid')
        c2 = class_map.get(arc_fids[1], {}).get('centroid')

        T.check_true(f"{arc_fids[0]} centroid exists", c1 is not None)
        T.check_true(f"{arc_fids[1]} centroid exists", c2 is not None)

        if c1 and c2:
            gn = engine._ground_normal
            if gn == (0, 1, 0):
                spacing = abs(c1[1] - c2[1])
            else:
                # gn=(0,0,1): 軌道間距沿 Z 軸量測
                spacing = abs(c1[2] - c2[2])
            T.check("Curved rail spacing", round(spacing, 1),
                    EXPECTED['curved_rail_spacing'])


def test_stp_data(engine, T):
    """測試 stp_data 結構（所有繪圖數值的來源）"""
    print("\n--- Test: stp_data Assembly ---")

    info = engine.get_model_info()
    cl = info.get('cutting_list', {})
    track_items = cl.get('track_items', [])
    pipe_centerlines = info.get('pipe_centerlines', [])
    part_cls = info.get('part_classifications', [])
    angles = info.get('angles', [])

    class_map = {c['feature_id']: c for c in part_cls}
    track_pipes = [pc for pc in pipe_centerlines
                   if class_map.get(pc['solid_id'], {}).get('class') == 'track']

    # 管徑
    diameters = [pc.get('pipe_diameter', 0) for pc in track_pipes]
    pipe_d = max(diameters) if diameters else 0
    T.check("pipe_diameter", pipe_d, EXPECTED['pipe_diameter'])

    # 弧段
    arc_items = [t for t in track_items if t.get('type') == 'arc']
    if arc_items:
        T.check("arc_radius", arc_items[0].get('radius', 0),
                EXPECTED['arc_radius'])
        T.check("arc_angle_deg", arc_items[0].get('angle_deg', 0),
                EXPECTED['arc_angle_deg'])
        T.check("arc_height_gain", arc_items[0].get('height_gain', 0),
                EXPECTED['arc_height_gain'], tolerance=0.1)
        T.check("arc_cl_length", arc_items[0].get('arc_length', 0),
                EXPECTED['arc_cl_length'], tolerance=1)

    # 仰角（進入 20°、轉換 15°、離開 35°）
    track_elevs = [a.get('angle_deg', 0) for a in angles
                   if a.get('type') == 'track_elevation' and a.get('angle_deg', 0) > 0.5]
    T.check_true("Entry elevation 20° present",
                 any(abs(v - EXPECTED['entry_elevation_deg']) < 0.5 for v in track_elevs),
                 f"elevations found: {track_elevs}")
    T.check_true("Exit elevation 35° present",
                 any(abs(v - EXPECTED['exit_elevation_deg']) < 0.5 for v in track_elevs),
                 f"elevations found: {track_elevs}")

    # 弦長
    chord = 2 * EXPECTED['arc_radius'] * math.sin(
        math.radians(EXPECTED['arc_angle_deg'] / 2))
    T.check("chord_length", round(chord), EXPECTED['chord_length'])

    # 彎曲方向
    bend_dir = engine._detect_bend_direction(pipe_centerlines, part_cls)
    T.check("bend_direction", bend_dir, EXPECTED['bend_direction'])


def test_drawing_generation(engine, T):
    """測試完整圖面生成（4 張 DXF）"""
    print("\n--- Test: Drawing Generation (4 sheets) ---")

    output_dir = os.path.join(ROOT_DIR, 'output')
    result = engine.generate_sub_assembly_drawing(output_dir)

    T.check("Drawing count", len(result), 4)
    for f in result:
        T.check_true(f"File exists: {os.path.basename(f)}", os.path.exists(f))


def _find_dxf(base_name, sheet_num, output_dir):
    """尋找圖面 DXF，相容不同命名格式（- 或 _）"""
    for sep in ['-', '_']:
        p = os.path.join(output_dir, f"{base_name}{sep}{sheet_num}.dxf")
        if os.path.exists(p):
            return p
    return None


def test_dxf_content_draw1(T):
    """Drawing 1 DXF: 腳架×1 + 支撐架×1 (PSA20)"""
    print("\n--- Test: Drawing 1 DXF Content (2-4.stp) ---")
    import ezdxf
    output_dir = os.path.join(ROOT_DIR, 'output')
    dxf_path = _find_dxf('2-4', 1, output_dir)
    if not dxf_path:
        T.check_true("Draw 1 DXF exists", False, "not found in output/")
        return
    msp = ezdxf.readfile(dxf_path).modelspace()
    texts = [e.dxf.text for e in msp if e.dxftype() == 'TEXT']
    T.check_true("Draw 1: BOM contains '支撐架'",
                 any('支撐架' in t for t in texts),
                 f"texts={texts[:20]}")
    T.check_true("Draw 1: BOM contains 'PSA'",
                 any('PSA' in t for t in texts),
                 f"texts={[t for t in texts if 'PSA' in t]}")


def test_dxf_content_draw2(T):
    """測試 Drawing 2 DXF 中的實際文字數值"""
    print("\n--- Test: Drawing 2 DXF Content (2-4.stp) ---")

    import ezdxf
    output_dir = os.path.join(ROOT_DIR, 'output')
    dxf_path = _find_dxf('2-4', 2, output_dir)
    if not dxf_path:
        T.check_true("Draw 2 DXF exists", False, "not found in output/")
        return

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    all_text = []
    for e in msp:
        if e.dxftype() == 'TEXT':
            all_text.append(e.dxf.text.strip())
        elif e.dxftype() == 'MTEXT':
            all_text.append(e.text.strip())

    # 弧半徑
    T.check_true("DXF contains R260",
                 any('R260' in t or '260' in t for t in all_text),
                 f"texts sample: {' | '.join(all_text[:20])}")

    # 弧角
    T.check_true("DXF contains 180",
                 any('180' in t for t in all_text))

    # 高度增益
    T.check_true("DXF contains h_gain 297.3",
                 any('297.3' in t for t in all_text))

    # 弧長
    T.check_true("DXF contains arc_len 869",
                 any('869' in t for t in all_text))

    # 軌道間距
    T.check_true("DXF contains rail_spacing 251.1",
                 any('251.1' in t for t in all_text))

    # 3D 端點距
    T.check_true("DXF contains endpoint ~599",
                 any(t in ('599', '600') for t in all_text))

    # 支撐架圓圈（Draw 2 應有 5 個）
    circles = [e for e in msp if e.dxftype() == 'CIRCLE']
    bracket_circles = [c for c in circles if 0.3 < c.dxf.radius < 8]
    T.check_true("DXF has bracket circles (Draw2 = 5 brackets)",
                 len(bracket_circles) >= 5,
                 f"found {len(bracket_circles)} circles with 0.3<r<8")


def test_dxf_content_draw3(T):
    """測試 Drawing 3 DXF 中的腳架長度和球號標註"""
    print("\n--- Test: Drawing 3 DXF Content (2-4.stp) ---")

    import ezdxf
    output_dir = os.path.join(ROOT_DIR, 'output')
    dxf_path = _find_dxf('2-4', 3, output_dir)
    if not dxf_path:
        T.check_true("Draw 3 DXF exists", False, "not found in output/")
        return

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    all_text = [e.dxf.text.strip() for e in msp if e.dxftype() == 'TEXT']

    # Draw 3 有 2 支腳架 → 應包含 leg_lengths[1] 和 leg_lengths[2]
    for leg_len in EXPECTED['leg_lengths'][1:]:
        T.check_true(f"Draw 3: contains leg length {leg_len}",
                     any(str(leg_len) in t for t in all_text))

    # 取料球號（U1~U3, D1~D3）— 上下軌各 3 段（直彎直）
    for label in ['U1', 'U2', 'U3', 'D1', 'D2', 'D3']:
        T.check_true(f"Draw 3: cutting list label '{label}' exists",
                     any(t == label for t in all_text))

    # 取料球號應為藍色 (color=5)
    blue_labels = [
        e.dxf.text.strip() for e in msp
        if e.dxftype() == 'TEXT'
        and e.dxf.text.strip() in ['U1', 'U2', 'U3', 'D1', 'D2', 'D3']
        and e.dxf.color == 5
    ]
    T.check_true("Draw 3: cutting list labels are blue (color=5)",
                 len(blue_labels) >= 6,
                 f"found {len(blue_labels)} blue: {blue_labels}")

    # BOM 球號（腳架）
    circles = [e for e in msp if e.dxftype() == 'CIRCLE']
    T.check_true("Draw 3: BOM balloon circle exists",
                 len(circles) >= 1, f"found {len(circles)}")
    T.check_true("Draw 3: BOM balloon '1' text exists",
                 any(t == '1' for t in all_text))


# =====================================================================
# 主程式
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='Auto Drafter verification - 2-4.stp')
    parser.add_argument('--quick', action='store_true',
                        help='Only verify numeric values, skip drawing generation')
    parser.add_argument('--draw2', action='store_true',
                        help='Only verify Drawing 2')
    parser.add_argument('--draw3', action='store_true',
                        help='Only verify Drawing 3')
    args = parser.parse_args()

    T = TestRunner()

    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file not found: {MODEL_FILE}")
        sys.exit(1)

    print(f"Loading model: {MODEL_FILE}")
    from auto_drafter_system import MockCADEngine
    engine = MockCADEngine(MODEL_FILE)
    print(f"Ground normal: {engine._ground_normal}")

    only_draw = args.draw2 or args.draw3

    # ---- 數值驗證 ----
    if not only_draw:
        test_pipe_centerlines(engine, T)
        test_cutting_list(engine, T)
        test_rail_spacing(engine, T)
        test_stp_data(engine, T)

    # ---- 圖面生成 + DXF 內容驗證 ----
    if not args.quick:
        test_drawing_generation(engine, T)
        if args.draw2:
            test_dxf_content_draw2(T)
        elif args.draw3:
            test_dxf_content_draw3(T)
        else:
            test_dxf_content_draw1(T)
            test_dxf_content_draw2(T)
            test_dxf_content_draw3(T)

    success = T.summary()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
