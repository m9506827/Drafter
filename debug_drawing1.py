"""診斷 Drawing 1 的 transition bend 問題"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from auto_drafter_system import MockCADEngine

cad = MockCADEngine(r"test\2-2.stp")

part_classifications = cad._part_classifications
pipe_centerlines = cad._pipe_centerlines
track_elevations = cad._angles

# 建立分類映射
class_map = {c['feature_id']: c for c in part_classifications}

# 建立 track_items
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

# 偵測 sections
sections = cad._detect_track_sections(pipe_centerlines, part_classifications, track_items)

print(f"共 {len(sections)} 個 sections:")
for i, sec in enumerate(sections):
    t = sec['section_type']
    upper = [tr.get('solid_id', '?') for tr in sec.get('upper_tracks', [])]
    lower = [tr.get('solid_id', '?') for tr in sec.get('lower_tracks', [])]
    print(f"  [{i}] {t}: upper={upper}, lower={lower}")

# 找第一個 straight section (Drawing 1)
first_straight_idx = None
for si, sec in enumerate(sections):
    if sec['section_type'] == 'straight':
        first_straight_idx = si
        break

if first_straight_idx is not None:
    section = sections[first_straight_idx]
    print(f"\n=== Drawing 1 使用 section {first_straight_idx} ===")
    
    # 檢查 track 數量
    upper_tracks = section.get('upper_tracks', [])
    lower_tracks = section.get('lower_tracks', [])
    print(f"  upper_tracks: {len(upper_tracks)}")
    print(f"  lower_tracks: {len(lower_tracks)}")
    
    # 計算 transition bends
    pipe_diameter = 48.1
    rail_spacing = 500
    
    bends = cad._compute_transition_bends(
        section, track_elevations, pipe_centerlines,
        part_classifications, pipe_diameter, rail_spacing)
    
    print(f"\n  transition_bends: {len(bends)}")
    for b in bends:
        print(f"    {b}")
    
    # 如果 bends 為空，檢查為什麼
    if not bends:
        print("\n  [分析] 為什麼沒有 transition bends?")
        print(f"  upper_tracks 數量: {len(upper_tracks)}")
        print(f"  lower_tracks 數量: {len(lower_tracks)}")
        print(f"  需要至少 2 個 tracks 才能有 transition bend")
        
        # 檢查仰角
        elev_map = {}
        for te in track_elevations:
            if te.get('part_a') in [t.get('solid_id') for t in upper_tracks + lower_tracks]:
                elev_map[te.get('part_a')] = te.get('angle_deg', 0)
        print(f"  仰角: {elev_map}")
