"""檢查軌道仰角數據 - 從 start_point/end_point 計算"""
import sys
import math
sys.stdout.reconfigure(encoding='utf-8')

from auto_drafter_system import MockCADEngine

cad = MockCADEngine(r"test\2-2.stp")
info = cad.get_model_info()

pipe_centerlines = info.get('pipe_centerlines', [])
part_classifications = info.get('part_classifications', [])

class_map = {c['feature_id']: c for c in part_classifications}

print("=== 軌道仰角和 Section 結構分析 ===\n")

# 找出所有 track 並排序（包括彎軌）
all_tracks = []
for pc in pipe_centerlines:
    fid = pc['solid_id']
    cls = class_map.get(fid, {})
    if cls.get('class') == 'track':
        centroid = cls.get('centroid', (0, 0, 0))
        sp = pc.get('start_point', (0, 0, 0))
        ep = pc.get('end_point', (0, 0, 0))
        
        is_curved = any(seg.get('type') == 'arc' and seg.get('angle_deg', 0) >= 60 
                       for seg in pc.get('segments', []))
        
        # 計算仰角
        dz = abs(ep[2] - sp[2])
        dxy = math.sqrt((ep[0] - sp[0])**2 + (ep[1] - sp[1])**2)
        elev = math.degrees(math.atan2(dz, dxy)) if dxy > 1e-6 else 0
        
        all_tracks.append({
            'fid': fid,
            'z': centroid[2],
            'elev': elev,
            'is_curved': is_curved,
        })

all_tracks.sort(key=lambda t: t['z'])

print("所有軌道（按 Z 排序）：\n")
for i, t in enumerate(all_tracks):
    marker = "(彎軌)" if t['is_curved'] else "(直軌)"
    print(f"{i+1}. {t['fid']} {marker}: Z={t['z']:.1f}, 仰角={t['elev']:.1f}°")

# 分析 Drawing 1 的 section（彎軌前的直軌）
print("\n" + "=" * 60)
print("Drawing 1 區段分析（彎軌前的第一個直軌區段）：")
print("=" * 60)

# 找第一個彎軌的索引
curved_idx = None
for i, t in enumerate(all_tracks):
    if t['is_curved']:
        curved_idx = i
        break

if curved_idx:
    # 彎軌前的直軌
    before_curved = [t for t in all_tracks[:curved_idx] if not t['is_curved']]
    if before_curved:
        print(f"\n彎軌前的直軌: {[t['fid'] for t in before_curved]}")
        if len(before_curved) >= 2:
            elev_diff = abs(before_curved[-1]['elev'] - before_curved[0]['elev'])
            print(f"仰角差（計算出的 transition bend 角度）: {elev_diff:.1f}°")
        elif len(before_curved) == 1:
            # 單軌：需要與彎軌後的直軌比較
            after_curved = [t for t in all_tracks[curved_idx+1:] if not t['is_curved']]
            if after_curved:
                elev_diff = abs(after_curved[0]['elev'] - before_curved[0]['elev'])
                print(f"與彎軌後第一軌的仰角差: {elev_diff:.1f}°")
                print(f"(這就是 Drawing 1 應該使用的 transition bend 角度)")
