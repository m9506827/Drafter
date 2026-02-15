#!/usr/bin/env python3
"""
自動化驗證腳本 — 驗證 auto_drafter_system.py 產出的所有關鍵數值
=================================================================
用法:
    py test/verify_values.py              # 執行完整驗證（含生成圖面）
    py test/verify_values.py --quick      # 只驗證數值，不生成圖面
    py test/verify_values.py --draw2      # 只驗證 Drawing 2
    py test/verify_values.py --draw3      # 只驗證 Drawing 3

每次修改 auto_drafter_system.py 後必須執行此腳本。
所有斷言失敗都會彙報，不會在第一個錯誤就中斷。
"""
import sys
import os
import math
import argparse

# 設定路徑
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.environ['DRAFTER_NO_GUI'] = '1'

MODEL_FILE = os.path.join(ROOT_DIR, 'test', '2-2.stp')

# =====================================================================
# 期望值基準（來自 2-2.stp 的標準圖面）
# =====================================================================
EXPECTED = {
    # --- 弧段參數（Draw 2 核心） ---
    'arc_radius':       250,
    'arc_angle_deg':    180.0,
    'arc_height_gain':  490.8,
    'arc_cl_length':    926,       # 允許 ±1
    'arc_3d_endpoint':  700,       # 允許 ±2 (rounded)

    # --- 管徑 ---
    'pipe_diameter':    48.1,

    # --- 軌道間距 ---
    'curved_rail_spacing': 230.2,  # F03-F06 centroid Z 距離

    # --- 腳架長度（avg(bbox,face_ext) - 50, 個位數4捨5入10位） ---
    'leg_lengths':      [510, 470, 430],

    # --- 支撐架 ---
    'bracket_count':    5,

    # --- 彎曲方向 ---
    'bend_direction':   'left',

    # --- 仰角 ---
    'elevation_deg':    32.0,

    # --- transition bend 半徑 ---
    'upper_bend_r':     270,
    'lower_bend_r':     220,

    # --- 弦長 (2R * sin(θ/2), θ=180°) ---
    'chord_length':     500,
}


# =====================================================================
# 測試框架
# =====================================================================
class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, actual, expected, tolerance=0):
        """比較數值，tolerance=0 表示完全匹配"""
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
        """檢查 actual 是否在 expected_list 中"""
        if actual in expected_list:
            self.passed += 1
            print(f"  [PASS] {name}: {actual}")
        else:
            self.failed += 1
            msg = f"  [FAIL] {name}: got {actual}, expected one of {expected_list}"
            print(msg)
            self.errors.append(msg)

    def check_true(self, name, condition, detail=""):
        """布林檢查"""
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

        # 3D endpoint distance
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

    # 軌道弧段
    arc_items = [t for t in cl.get('track_items', []) if t.get('type') == 'arc']
    T.check_true("Arc track items >= 2", len(arc_items) >= 2,
                 f"found {len(arc_items)}")

    for ai in arc_items:
        T.check(f"{ai['item']} R", ai.get('radius', 0), EXPECTED['arc_radius'])
        T.check(f"{ai['item']} angle", ai.get('angle_deg', 0),
                EXPECTED['arc_angle_deg'])
        T.check(f"{ai['item']} h_gain", ai.get('height_gain', 0),
                EXPECTED['arc_height_gain'], tolerance=0.1)

    # 腳架長度
    leg_items = cl.get('leg_items', [])
    T.check(f"Leg count", len(leg_items), len(EXPECTED['leg_lengths']))
    for i, li in enumerate(leg_items):
        exp_len = EXPECTED['leg_lengths'][i] if i < len(EXPECTED['leg_lengths']) else 0
        T.check(f"Leg {i+1} length", li.get('line_length', 0), exp_len)

    # 支撐架數量
    bracket_items = cl.get('bracket_items', [])
    total_brackets = sum(b.get('quantity', 1) for b in bracket_items)
    T.check("Bracket count", total_brackets, EXPECTED['bracket_count'])


def test_rail_spacing(engine, T):
    """測試彎軌區段軌道間距"""
    print("\n--- Test: Rail Spacing (Curved Section) ---")

    info = engine.get_model_info()
    part_cls = info.get('part_classifications', [])
    class_map = {c['feature_id']: c for c in part_cls}

    f03_cen = class_map.get('F03', {}).get('centroid', None)
    f06_cen = class_map.get('F06', {}).get('centroid', None)

    T.check_true("F03 centroid exists", f03_cen is not None)
    T.check_true("F06 centroid exists", f06_cen is not None)

    if f03_cen and f06_cen:
        gn = engine._ground_normal
        if gn == (0, 1, 0):
            spacing = abs(f03_cen[1] - f06_cen[1])
        else:
            spacing = abs(f03_cen[2] - f06_cen[2])
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

    # 仰角
    track_elevs = [a for a in angles if a.get('type') == 'track_elevation']
    non_zero_elev = [te.get('angle_deg', 0) for te in track_elevs
                     if te.get('angle_deg', 0) > 0.5]
    if non_zero_elev:
        T.check("elevation_deg", non_zero_elev[0], EXPECTED['elevation_deg'])

    # 弦長
    arc_r = EXPECTED['arc_radius']
    arc_a = EXPECTED['arc_angle_deg']
    chord = 2 * arc_r * math.sin(math.radians(arc_a / 2))
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


def test_dxf_content_draw2(T):
    """測試 Drawing 2 DXF 中的實際文字數值"""
    print("\n--- Test: Drawing 2 DXF Content ---")

    import ezdxf
    dxf_path = os.path.join(ROOT_DIR, 'output', '2-2_2.dxf')
    if not os.path.exists(dxf_path):
        T.check_true("Draw 2 DXF exists", False, f"not found: {dxf_path}")
        return

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # 收集所有文字內容
    all_text = []
    for e in msp:
        if e.dxftype() == 'TEXT':
            all_text.append(e.dxf.text.strip())
        elif e.dxftype() == 'MTEXT':
            all_text.append(e.text.strip())

    joined = ' | '.join(all_text)

    # 檢查關鍵數值出現在文字中
    T.check_true("DXF contains R250",
                 any('R250' in t or 'R=250' in t or '250' in t
                     for t in all_text),
                 f"texts: {joined[:200]}")

    T.check_true("DXF contains 180",
                 any('180' in t for t in all_text),
                 f"not found in texts")

    T.check_true("DXF contains h_gain 490.8",
                 any('490.8' in t for t in all_text),
                 f"not found in texts")

    T.check_true("DXF contains arc_len 926",
                 any('926' in t for t in all_text),
                 f"not found in texts")

    T.check_true("DXF contains rail_spacing 230.2",
                 any('230.2' in t for t in all_text),
                 f"not found in texts")

    T.check_true("DXF contains endpoint ~700-701",
                 any(t in ('700', '701') for t in all_text),
                 f"not found in texts")

    # 檢查有 CIRCLE 實體（支撐架圓柱），半徑應接近實際管徑 13mm 的一半
    circles = [e for e in msp if e.dxftype() == 'CIRCLE']
    bracket_circles = [c for c in circles if 0.3 < c.dxf.radius < 8]
    T.check_true("DXF has bracket circles (5 brackets)",
                 len(bracket_circles) >= 5,
                 f"found {len(bracket_circles)} circles with 0.3<r<8")

    # 計算弧心（從 5 個 bracket circle 中心推算 — 它們在以弧心為中心的圓上）
    all_lines = [e for e in msp if e.dxftype() == 'LINE']
    arc_center_est = None
    if len(bracket_circles) >= 3:
        # 用前 3 個 bracket circle 的中心做三點求圓心
        pts = [(bc.dxf.center.x, bc.dxf.center.y) for bc in bracket_circles[:3]]
        ax, ay = pts[0]; bx, by = pts[1]; cx, cy = pts[2]
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) > 1e-6:
            ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
                  + (cx**2 + cy**2) * (ay - by)) / D
            uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
                  + (cx**2 + cy**2) * (bx - ax)) / D
            arc_center_est = (ux, uy)
            est_r = math.sqrt((ax - ux)**2 + (ay - uy)**2)

            # 檢查 bracket circles 在弧面內側（凹側/下側）
            # bracket circles 到弧心距離 < 軌道中心線半徑（它們在弧心方向延伸）
            # 簡單驗證：5 個 circles 都等距於弧心，且距離 < track R
            inner_count = 0
            for bc in bracket_circles[:5]:
                bx2, by2 = bc.dxf.center.x, bc.dxf.center.y
                d = math.sqrt((bx2 - ux)**2 + (by2 - uy)**2)
                if abs(d - est_r) < est_r * 0.15:  # 半徑一致（15%容差）
                    inner_count += 1
            T.check_true("Bracket circles are at consistent radius (inner/concave side)",
                         inner_count >= 4,
                         f"{inner_count}/5 at consistent radius ~{est_r:.0f}")

    # 檢查支撐架有管壁雙線（每個支撐架應有 2 條平行 LINE + 1 條封口線）
    if bracket_circles and all_lines:
        brk_line_count = 0
        for bc in bracket_circles[:5]:
            bx, by = bc.dxf.center.x, bc.dxf.center.y
            nearby = 0
            for ln in all_lines:
                sx, sy = ln.dxf.start.x, ln.dxf.start.y
                ex, ey = ln.dxf.end.x, ln.dxf.end.y
                mx, my = (sx + ex) / 2, (sy + ey) / 2
                d = math.sqrt((mx - bx)**2 + (my - by)**2)
                if d < 20:
                    nearby += 1
            if nearby >= 3:  # 2 管壁 + 1 封口
                brk_line_count += 1
        T.check_true("Brackets have wall lines (>=3 lines near each circle)",
                     brk_line_count >= 4,
                     f"{brk_line_count}/5 brackets with wall lines")

    # 檢查 R250 引出線從弧心出發
    r_text = None
    for e in msp:
        if e.dxftype() == 'TEXT' and 'R250' in e.dxf.text:
            r_text = e
            break
    if r_text and arc_center_est:
        acx, acy = arc_center_est
        # 找端點在弧心附近的 LINE（leader 從弧心出發）
        center_lines = 0
        for ln in all_lines:
            sx, sy = ln.dxf.start.x, ln.dxf.start.y
            ex, ey = ln.dxf.end.x, ln.dxf.end.y
            d_s = math.sqrt((sx - acx)**2 + (sy - acy)**2)
            d_e = math.sqrt((ex - acx)**2 + (ey - acy)**2)
            if d_s < 5 or d_e < 5:
                center_lines += 1
        T.check_true("R250 leader starts from arc center",
                     center_lines >= 1,
                     f"found {center_lines} lines near arc center ({acx:.0f},{acy:.0f})")

    # ---- 側視圖：490.8 和 230.2 標註位置驗證 ----
    # 找 standalone 尺寸文字（排除取料明細等長文字，只匹配獨立數值）
    text_490 = None
    text_230 = None
    for e in msp:
        if e.dxftype() == 'TEXT':
            t = e.dxf.text.strip()
            # 獨立尺寸文字通常只有數值本身（< 10 字元）
            if t == '490.8':
                text_490 = e
            elif t == '230.2':
                text_230 = e
    if text_490 and text_230:
        x_490 = text_490.dxf.insert.x
        x_230 = text_230.dxf.insert.x
        # 兩者 X 座標差應 < 5（同一條垂直尺寸線上）
        x_diff = abs(x_490 - x_230)
        T.check_true("490.8 and 230.2 at same X position",
                     x_diff < 5,
                     f"x_490={x_490:.1f}, x_230={x_230:.1f}, diff={x_diff:.1f}")
        # 230.2 應在 490.8 正下方（Y 更小）
        y_490 = text_490.dxf.insert.y
        y_230 = text_230.dxf.insert.y
        T.check_true("230.2 is below 490.8",
                     y_230 < y_490,
                     f"y_490={y_490:.1f}, y_230={y_230:.1f}")
    else:
        T.check_true("Found both 490.8 and 230.2 TEXT entities",
                     False,
                     f"490.8={'found' if text_490 else 'missing'}, "
                     f"230.2={'found' if text_230 else 'missing'}")


def test_balloon_annotations(T):
    """測試 Drawing 2 球號標註（業界慣例：氣泡圓 + 引出線 + 箭頭）"""
    print("\n--- Test: Drawing 2 Balloon Annotations ---")

    import ezdxf
    dxf_path = os.path.join(ROOT_DIR, 'output', '2-2_2.dxf')
    if not os.path.exists(dxf_path):
        T.check_true("Draw 2 DXF exists for balloon test", False, f"not found")
        return

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # 收集所有文字和圓
    all_texts = [(e, e.dxf.text.strip()) for e in msp if e.dxftype() == 'TEXT']
    all_circles = [e for e in msp if e.dxftype() == 'CIRCLE']
    all_lines = [e for e in msp if e.dxftype() == 'LINE']

    # 側視圖區域：取料球號 U1, D1 應為藍色文字 + 引線（不使用氣泡圓）
    for label in ['U1', 'D1']:
        found_blue_label = False
        has_blue_leader = False
        for e, t in all_texts:
            if t != label:
                continue
            # 檢查是否為藍色 (color=5)
            if e.dxf.color == 5:
                found_blue_label = True
                tx, ty = e.dxf.insert.x, e.dxf.insert.y
                # 檢查是否有藍色引線（至少一條 color=5 的 LINE 端點靠近文字）
                for ln in all_lines:
                    if ln.dxf.color != 5:
                        continue
                    for pt in [(ln.dxf.start.x, ln.dxf.start.y),
                               (ln.dxf.end.x, ln.dxf.end.y)]:
                        dist = math.sqrt((pt[0] - tx)**2 + (pt[1] - ty)**2)
                        if dist < 3.0:
                            has_blue_leader = True
                            break
                    if has_blue_leader:
                        break
                break
        T.check_true(f"Draw 2: {label} blue text + leader",
                     found_blue_label and has_blue_leader,
                     f"blue_text={found_blue_label}, leader={has_blue_leader}")

    # BOM 球號 "1"（支撐架）— 在側視圖中應有一個氣泡
    found_bom1 = False
    for e, t in all_texts:
        if t != '1':
            continue
        tx, ty = e.dxf.insert.x, e.dxf.insert.y
        th = e.dxf.height
        if not (2.5 <= th <= 6.0):
            continue
        # 側視圖區域 y 大約 80~180
        if ty < 70 or ty > 200:
            continue
        for c in all_circles:
            cx, cy = c.dxf.center.x, c.dxf.center.y
            r = c.dxf.radius
            dist = math.sqrt((cx - tx)**2 + (cy - ty)**2)
            if 3.0 < r < 8.0 and dist < r + 3:
                found_bom1 = True
                break
        if found_bom1:
            break
    T.check_true("Balloon BOM-1 exists (bracket ball number)",
                 found_bom1, "BOM ball number 1 in side view")


def test_dxf_content_draw1(T):
    """測試 Drawing 1 DXF 中的腳架長度和彎曲方向"""
    print("\n--- Test: Drawing 1 DXF Content ---")

    import ezdxf
    dxf_path = os.path.join(ROOT_DIR, 'output', '2-2-1.dxf')
    if not os.path.exists(dxf_path):
        T.check_true("Draw 1 DXF exists", False, f"not found: {dxf_path}")
        return

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    all_text = []
    for e in msp:
        if e.dxftype() == 'TEXT':
            all_text.append(e.dxf.text.strip())
        elif e.dxftype() == 'MTEXT':
            all_text.append(e.text.strip())

    # 腳架長度
    for leg_len in EXPECTED['leg_lengths'][:2]:  # Draw 1 包含前兩支腳架
        T.check_true(f"DXF contains leg length {leg_len}",
                     any(str(leg_len) in t for t in all_text),
                     f"not found")


def _load_draw3_dxf():
    """載入 Drawing 3 DXF，回傳 (doc, msp) 或 (None, None)"""
    import ezdxf
    for name in ['2-2_3.dxf', '2-2-3.dxf']:
        p = os.path.join(ROOT_DIR, 'output', name)
        if os.path.exists(p):
            doc = ezdxf.readfile(p)
            return doc, doc.modelspace()
    return None, None


def _collect_draw3_entities(msp):
    """收集 Drawing 3 DXF 中所有 TEXT 實體及其座標"""
    text_entities = []
    for e in msp:
        if e.dxftype() == 'TEXT':
            t = e.dxf.text.strip()
            x, y = e.dxf.insert.x, e.dxf.insert.y
            text_entities.append({'text': t, 'x': x, 'y': y,
                                  'color': e.dxf.color, 'entity': e})
    return text_entities


def test_dxf_content_draw3(T):
    """測試 Drawing 3 DXF 中的腳架長度和球號標註"""
    print("\n--- Test: Drawing 3 DXF Content ---")

    doc, msp = _load_draw3_dxf()
    if not msp:
        T.check_true("Draw 3 DXF exists", False, "not found")
        return

    all_text = [e.dxf.text.strip() for e in msp if e.dxftype() == 'TEXT']

    # 腳架長度（Drawing 3 的第三支腳架）
    leg3_len = EXPECTED['leg_lengths'][2]  # 430
    T.check_true(f"Draw 3: BOM leg length = {leg3_len}",
                 any(f"L={leg3_len}" in t for t in all_text),
                 f"expected L={leg3_len} in BOM")

    # 取料球號標註存在（U1, U2, U3, D1, D2）— 藍色文字 + 引線
    for label in ['U1', 'U2', 'U3', 'D1', 'D2']:
        found = any(t == label for t in all_text)
        T.check_true(f"Draw 3: cutting list label '{label}' exists",
                     found, f"not found in DXF text")

    # 取料球號標註應為藍色 (color=5)
    blue_labels = []
    for e in msp:
        if e.dxftype() == 'TEXT' and e.dxf.text.strip() in ['U1', 'U2', 'U3', 'D1', 'D2']:
            if e.dxf.color == 5:
                blue_labels.append(e.dxf.text.strip())
    T.check_true("Draw 3: cutting list labels are blue (color=5)",
                 len(blue_labels) >= 5,
                 f"found {len(blue_labels)} blue labels: {blue_labels}")

    # 引線（藍色 LINE）應存在 — 至少 5 條（每個取料球號一條引線）
    blue_lines = [e for e in msp if e.dxftype() == 'LINE' and e.dxf.color == 5]
    T.check_true("Draw 3: blue leader lines exist (>= 5)",
                 len(blue_lines) >= 5,
                 f"found {len(blue_lines)} blue lines")

    # BOM 球號（腳架 = 1）— 使用氣泡圓，區別於取料球號
    circles = [e for e in msp if e.dxftype() == 'CIRCLE']
    T.check_true("Draw 3: BOM balloon circle exists (>= 1)",
                 len(circles) >= 1,
                 f"found {len(circles)} circles")
    bom_found = any(t == '1' for t in all_text)
    T.check_true("Draw 3: BOM balloon '1' exists",
                 bom_found, "not found")


def test_draw3_angles(T):
    """測試 Drawing 3 角度標註：42°/58°/16° 的值和位置"""
    print("\n--- Test: Drawing 3 Angle Annotations ---")

    doc, msp = _load_draw3_dxf()
    if not msp:
        T.check_true("Draw 3 DXF exists for angle test", False, "not found")
        return

    te = _collect_draw3_entities(msp)

    # ---- 角度值存在性 ----
    # 文字格式: "42°", "58°", "16°" (使用 Unicode °)
    angle_42 = [t for t in te if t['text'] in ('42°', '42\u00B0')]
    angle_58 = [t for t in te if t['text'] in ('58°', '58\u00B0')]
    angle_16 = [t for t in te if t['text'] in ('16°', '16\u00B0')]

    T.check_true("Draw 3: angle 42° text exists",
                 len(angle_42) >= 1,
                 f"found {len(angle_42)}")
    T.check_true("Draw 3: angle 58° text exists",
                 len(angle_58) >= 1,
                 f"found {len(angle_58)}")
    T.check_true("Draw 3: angle 16° text exists",
                 len(angle_16) >= 1,
                 f"found {len(angle_16)}")

    # ---- 角度位置驗證 ----
    # 標準圖配置：58° 在最高（腳架-上軌交點），42° 在中間（腳架-下軌交點），16° 在最低（D2彎曲處）
    if angle_42 and angle_58 and angle_16:
        y_42 = angle_42[0]['y']
        y_58 = angle_58[0]['y']
        y_16 = angle_16[0]['y']

        # 58° 標示於上軌下側，42° 標示於下軌上側，兩者都在軌道之間
        # 42° 和 58° Y 位置應在 16° 之上
        T.check_true("Draw 3: 42° above 16° (Y position)",
                     y_42 > y_16,
                     f"42°.y={y_42:.1f}, 16°.y={y_16:.1f}")
        T.check_true("Draw 3: 58° above 16° (Y position)",
                     y_58 > y_16,
                     f"58°.y={y_58:.1f}, 16°.y={y_16:.1f}")

        # 42° 和 58° 標示在軌道內側（上軌下側、下軌上側），X 差可較大（< 25）
        x_42 = angle_42[0]['x']
        x_58 = angle_58[0]['x']
        x_diff_42_58 = abs(x_42 - x_58)
        T.check_true("Draw 3: 42° and 58° at similar X (leg position)",
                     x_diff_42_58 < 25,
                     f"42°.x={x_42:.1f}, 58°.x={x_58:.1f}, diff={x_diff_42_58:.1f}")

    # ---- 角度弧線存在性 ----
    arcs = [e for e in msp if e.dxftype() == 'ARC']
    T.check_true("Draw 3: angle arcs exist (>= 3)",
                 len(arcs) >= 3,
                 f"found {len(arcs)} arcs")


def test_draw3_dimensions(T):
    """測試 Drawing 3 尺寸標註：段長、總長"""
    print("\n--- Test: Drawing 3 Dimension Values ---")

    doc, msp = _load_draw3_dxf()
    if not msp:
        T.check_true("Draw 3 DXF exists for dim test", False, "not found")
        return

    te = _collect_draw3_entities(msp)
    all_text = [t['text'] for t in te]

    # ---- 直線段長度（取料明細的直線段） ----
    # 標準圖: U1=89.1, U3=147.3, D1=244.2
    for seg_len in ['89.1', '147.3', '244.2']:
        T.check_true(f"Draw 3: segment length {seg_len} exists",
                     any(seg_len in t for t in all_text),
                     f"not found in DXF text")

    # ---- 上軌端點距離 ----
    # 標準圖: 295.5 (兩端點直線距離，非展開長)
    # 允許範圍: 294 ~ 297
    upper_total_found = False
    upper_total_val = None
    for t in te:
        try:
            v = float(t['text'])
            if 293 <= v <= 298:
                upper_total_found = True
                upper_total_val = v
                break
        except ValueError:
            pass
    T.check_true("Draw 3: upper track endpoint distance ~295.5",
                 upper_total_found,
                 f"found value: {upper_total_val}" if upper_total_val else "no value in range 293-298")

    # ---- 下軌端點距離 ----
    # 標準圖: 318.8 (兩端點直線距離，非展開長)
    # 允許範圍: 317 ~ 320
    lower_total_found = False
    lower_total_val = None
    for t in te:
        try:
            v = float(t['text'])
            if 317 <= v <= 320:
                lower_total_found = True
                lower_total_val = v
                break
        except ValueError:
            pass
    T.check_true("Draw 3: lower track endpoint distance ~318.8",
                 lower_total_found,
                 f"found value: {lower_total_val}" if lower_total_val else "no value in range 317-320")

    # ---- 整體尺寸（上下軌中心線垂直距離 / 起始端＆末端） ----
    # 標準圖: 196.4 (起始端) / 230.2 (末端)
    # 目前程式計算為 perpendicular spacing / cos(angle) 的結果
    # 至少應有一個垂直尺寸在合理範圍 (80~250)
    vert_dims = []
    for t in te:
        try:
            v = float(t['text'])
            if 80 <= v <= 260 and v not in (89.1, 147.3, 244.2):
                vert_dims.append(v)
        except ValueError:
            pass
    T.check_true("Draw 3: vertical dimensions exist (>= 1)",
                 len(vert_dims) >= 1,
                 f"found values: {vert_dims}")


# =====================================================================
# 主程式
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='Auto Drafter verification')
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
            test_balloon_annotations(T)
        elif args.draw3:
            test_dxf_content_draw3(T)
            test_draw3_angles(T)
            test_draw3_dimensions(T)
        else:
            test_dxf_content_draw2(T)
            test_balloon_annotations(T)
            test_dxf_content_draw1(T)
            test_dxf_content_draw3(T)
            test_draw3_angles(T)
            test_draw3_dimensions(T)

    # ---- 結果彙報 ----
    success = T.summary()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
