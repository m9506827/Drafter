"""檢查 section3 的取料明細"""
from auto_drafter_system import MockCADEngine

cad = MockCADEngine(r"test\2-2.stp")

# 取得各種資料
part_classifications = cad._part_classifications
pipe_centerlines = cad._pipe_centerlines
track_elevations = cad._angles

# 調用 _detect_track_sections
pipe_diameter = 48.1
rail_spacing = 500  # 假設值

# 建立 track_items（簡化版）
class_map = {c['feature_id']: c for c in part_classifications}
track_items = []
for pc in pipe_centerlines:
    fid = pc['solid_id']
    cls = class_map.get(fid, {})
    if cls.get('class') == 'track':
        track_items.append({
            'solid_id': fid,
            'pipe_data': pc,
            'centroid': cls.get('centroid', (0, 0, 0)),
        })

# 呼叫 _detect_track_sections
sections = cad._detect_track_sections(pipe_centerlines, part_classifications, track_items)

print(f"共 {len(sections)} 個 sections:")
for i, sec in enumerate(sections):
    t = sec['section_type']
    upper = [tr.get('solid_id', '?') for tr in sec.get('upper_tracks', [])]
    lower = [tr.get('solid_id', '?') for tr in sec.get('lower_tracks', [])]
    print(f"  [{i}] {t}: upper={upper}, lower={lower}")

# 找第二個 straight section
second_straight_idx = None
straight_count = 0
for si, sec in enumerate(sections):
    if sec['section_type'] == 'straight':
        straight_count += 1
        if straight_count == 2:
            second_straight_idx = si
            break

print(f"\nsecond_straight_idx = {second_straight_idx}")

if second_straight_idx is not None:
    section3 = sections[second_straight_idx]
    print(f"\nsection3 = {section3['section_type']}")
    print(f"  upper_tracks: {[t.get('solid_id', '?') for t in section3.get('upper_tracks', [])]}")
    print(f"  lower_tracks: {[t.get('solid_id', '?') for t in section3.get('lower_tracks', [])]}")
    
    # 計算 transition bends
    bends = cad._compute_transition_bends(
        section3, track_elevations, pipe_centerlines,
        part_classifications, pipe_diameter, rail_spacing)
    print(f"\nsection3_bends: {bends}")
    
    # 建構取料明細
    cl = cad._build_section_cutting_list(
        section3, bends, track_items,
        part_classifications, pipe_diameter)
    
    print(f"\nsection3_cutting_list ({len(cl)} items):")
    for it in cl:
        print(f"  {it.get('item')}: {it.get('type')} - {it.get('spec', it.get('length', ''))}")
