"""分析軌道仰角數據"""
from auto_drafter_system import MockCADEngine

cad = MockCADEngine(r"test\2-2.stp")

track_elevations = cad._angles
pipe_centerlines = cad._pipe_centerlines
part_classifications = cad._part_classifications

class_map = {c['feature_id']: c for c in part_classifications}

print("=== 軌道仰角數據 ===\n")
for te in track_elevations:
    part_a = te.get('part_a', '')
    angle = te.get('angle_deg', 0)
    cls = class_map.get(part_a, {})
    part_class = cls.get('class', 'unknown')
    print(f"{part_a}: 仰角={angle}°, class={part_class}")

print("\n=== 各軌道 segments ===\n")
for pc in pipe_centerlines:
    fid = pc['solid_id']
    cls = class_map.get(fid, {})
    if cls.get('class') == 'track':
        centroid = cls.get('centroid', (0,0,0))
        print(f"\n{fid}: centroid=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
        for j, seg in enumerate(pc.get('segments', [])):
            if seg.get('type') == 'straight':
                print(f"  seg[{j}]: straight, L={seg.get('length', 0):.1f}")
            elif seg.get('type') == 'arc':
                print(f"  seg[{j}]: arc, angle={seg.get('angle_deg', 0):.1f}°, R={seg.get('radius', 0):.1f}")

# 找出彎軌和其相鄰直軌
print("\n=== 彎軌相鄰分析 ===\n")

# 收集所有 tracks 並按 Z 座標排序
tracks = []
for pc in pipe_centerlines:
    fid = pc['solid_id']
    cls = class_map.get(fid, {})
    if cls.get('class') == 'track':
        centroid = cls.get('centroid', (0,0,0))
        is_curved = any(seg.get('type') == 'arc' and seg.get('angle_deg', 0) >= 60 
                       for seg in pc.get('segments', []))
        tracks.append({
            'fid': fid,
            'z': centroid[2],
            'is_curved': is_curved,
            'segments': pc.get('segments', []),
        })

tracks.sort(key=lambda t: t['z'])

for i, t in enumerate(tracks):
    marker = " (CURVED)" if t['is_curved'] else ""
    print(f"[{i}] {t['fid']}: Z={t['z']:.1f}{marker}")
    
    # 找出其仰角
    for te in track_elevations:
        if te.get('part_a') == t['fid']:
            print(f"      仰角: {te.get('angle_deg', 0)}°")
            break
