"""詳細分析 2-4.stp 軌道結構"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from auto_drafter_system import MockCADEngine

cad = MockCADEngine(r"2-4.stp")

pipe_centerlines = cad._pipe_centerlines
part_classifications = cad._part_classifications
class_map = {c['feature_id']: c for c in part_classifications}

print("=== 2-4.stp 軌道詳細分析 ===\n")

# 找出所有 track
tracks = []
for pc in pipe_centerlines:
    fid = pc['solid_id']
    cls = class_map.get(fid, {})
    if cls.get('class') == 'track':
        centroid = cls.get('centroid', (0, 0, 0))
        start_p = pc.get('start_point', (0, 0, 0))
        end_p = pc.get('end_point', (0, 0, 0))
        
        is_curved = False
        arc_angle = 0
        for seg in pc.get('segments', []):
            if seg.get('type') == 'arc' and seg.get('angle_deg', 0) >= 60:
                is_curved = True
                arc_angle = seg.get('angle_deg', 0)
                
        tracks.append({
            'fid': fid,
            'centroid': centroid,
            'start_point': start_p,
            'end_point': end_p,
            'is_curved': is_curved,
            'arc_angle': arc_angle,
        })

# 按 centroid 的主軸排序（可能是 Y 或 Z）
# 先檢查哪個軸變化最大
all_x = [t['centroid'][0] for t in tracks]
all_y = [t['centroid'][1] for t in tracks]
all_z = [t['centroid'][2] for t in tracks]

x_range = max(all_x) - min(all_x)
y_range = max(all_y) - min(all_y)
z_range = max(all_z) - min(all_z)

print(f"座標範圍: X={x_range:.1f}, Y={y_range:.1f}, Z={z_range:.1f}")

# 選擇最大範圍的軸作為排序依據
if z_range >= y_range and z_range >= x_range:
    sort_key = lambda t: t['centroid'][2]
    axis_name = 'Z'
elif y_range >= x_range:
    sort_key = lambda t: t['centroid'][1]
    axis_name = 'Y'
else:
    sort_key = lambda t: t['centroid'][0]
    axis_name = 'X'

tracks.sort(key=sort_key)

print(f"按 {axis_name} 座標排序:\n")
print("-" * 100)

for t in tracks:
    marker = " (CURVED)" if t['is_curved'] else ""
    cx, cy, cz = t['centroid']
    sx, sy, sz = t['start_point']
    ex, ey, ez = t['end_point']
    
    print(f"{t['fid']}{marker}:")
    print(f"  Centroid: X={cx:.1f}, Y={cy:.1f}, Z={cz:.1f}")
    print(f"  Start:    X={sx:.1f}, Y={sy:.1f}, Z={sz:.1f}")
    print(f"  End:      X={ex:.1f}, Y={ey:.1f}, Z={ez:.1f}")
    if t['is_curved']:
        print(f"  Arc: {t['arc_angle']:.0f}°")
    print()

# 分析彎曲方向
print("=" * 100)
print("彎曲方向分析:")
print("=" * 100)

curved = [t for t in tracks if t['is_curved']]
straight = [t for t in tracks if not t['is_curved']]

if curved:
    ct = curved[0]
    print(f"\n彎軌 {ct['fid']}:")
    print(f"  Start: ({ct['start_point'][0]:.1f}, {ct['start_point'][1]:.1f}, {ct['start_point'][2]:.1f})")
    print(f"  End:   ({ct['end_point'][0]:.1f}, {ct['end_point'][1]:.1f}, {ct['end_point'][2]:.1f})")
    
    # 檢查 start 和 end 的 Y 座標變化
    y_diff = ct['end_point'][1] - ct['start_point'][1]
    print(f"\n  Y 座標變化 (彎軌內): {y_diff:.1f}")
    
    if y_diff > 50:
        print(f"  → 右彎 (Y 座標增加)")
    elif y_diff < -50:
        print(f"  → 左彎 (Y 座標減少)")
    else:
        # 檢查 X 座標
        x_diff = ct['end_point'][0] - ct['start_point'][0]
        print(f"  X 座標變化 (彎軌內): {x_diff:.1f}")
        if x_diff > 50:
            print(f"  → 右彎")
        elif x_diff < -50:
            print(f"  → 左彎")
