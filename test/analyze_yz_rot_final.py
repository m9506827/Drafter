import ezdxf
import math

DXF_PATH = r"D:\Google\Drafter\test\output\1-2_YZ_rot.dxf"
TARGET = 850.8

doc = ezdxf.readfile(DXF_PATH)
msp = doc.modelspace()
polylines = list(msp.query("LWPOLYLINE"))

# Collect all unique structural endpoints (start/end of each polyline)
endpoints = []
for idx, pl in enumerate(polylines):
    pts = list(pl.get_points(format="xy"))
    if len(pts) >= 2:
        endpoints.append((pts[0][0], pts[0][1], idx, "start"))
        endpoints.append((pts[-1][0], pts[-1][1], idx, "end"))

print("=" * 80)
print(f"ENDPOINT-ONLY SEARCH FOR DISTANCE ~ {TARGET}")
print(f"Total endpoints: {len(endpoints)}")
print("=" * 80)

# Deduplicate endpoints
ep_map = {}
for x, y, idx, pos in endpoints:
    key = (round(x, 2), round(y, 2))
    if key not in ep_map:
        ep_map[key] = (x, y, idx, pos)
unique_eps = list(ep_map.values())
print(f"Unique endpoints: {len(unique_eps)}")

matches = []
for i in range(len(unique_eps)):
    x1, y1, pi1, pos1 = unique_eps[i]
    for j in range(i + 1, len(unique_eps)):
        x2, y2, pi2, pos2 = unique_eps[j]
        d = math.hypot(x2 - x1, y2 - y1)
        if abs(d - TARGET) < 1.0:
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            matches.append(((x1, y1, pi1, pos1), (x2, y2, pi2, pos2), d, dx, dy))

matches.sort(key=lambda m: abs(m[2] - TARGET))
print(f"\nEndpoint pairs within 1.0 of {TARGET}: {len(matches)}")
for k, (p1, p2, d, dx, dy) in enumerate(matches[:25]):
    angle = math.degrees(math.atan2(dy, dx))
    print(f"  #{k+1:3d}  dist={d:.4f}  dx={dx:.2f}  dy={dy:.2f}  angle={angle:.1f}deg"
          f"  P1=({p1[0]:.2f},{p1[1]:.2f})[PL-{p1[2]:03d}.{p1[3]}]"
          f"  P2=({p2[0]:.2f},{p2[1]:.2f})[PL-{p2[2]:03d}.{p2[3]}]")

# Now check: the plan mentions 850.8 as a diagonal dimension on YZ_rot view
# In a structural steel drawing, this is likely the length of a diagonal brace member
# or the distance between two connection points
# Let us look at what structural members span that distance

print()
print("=" * 80)
print("POLYLINE START-TO-END DISTANCES (checking for 850.8 member length)")
print("=" * 80)

member_lengths = []
for idx, pl in enumerate(polylines):
    pts = list(pl.get_points(format="xy"))
    if len(pts) < 2:
        continue
    s2e = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    member_lengths.append((idx, s2e, pts[0], pts[-1]))

# Sort by closeness to target
member_lengths.sort(key=lambda m: abs(m[1] - TARGET))
print(f"\nPolylines closest to {TARGET} start-to-end distance:")
for idx, s2e, start, end in member_lengths[:20]:
    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    angle = math.degrees(math.atan2(dy, dx)) if dx > 0.001 else 90.0
    print(f"  [PL-{idx:03d}]  s2e={s2e:.4f}  diff={s2e-TARGET:.4f}  dx={dx:.2f}  dy={dy:.2f}  angle={angle:.1f}deg"
          f"  ({start[0]:.2f},{start[1]:.2f})->({end[0]:.2f},{end[1]:.2f})")

# Check sums of connected vertical members
print()
print("=" * 80)
print("COMBINED VERTICAL MEMBER SPANS AT KEY X-POSITIONS")
print("=" * 80)

# For each X position where we have vertical members, find all vertical polylines
# and compute the total covered span
from collections import defaultdict

# Get all polylines with their bounding info
pl_data = []
for idx, pl in enumerate(polylines):
    pts = list(pl.get_points(format="xy"))
    if len(pts) < 2:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    pl_data.append((idx, min(xs), max(xs), min(ys), max(ys), w, h, pts[0], pts[-1]))

# Identify near-vertical polylines (aspect ratio > 3:1 height-dominant or width < 5)
vert_members = [(idx, xmin, xmax, ymin, ymax, w, h, s, e) 
                for idx, xmin, xmax, ymin, ymax, w, h, s, e in pl_data 
                if w < 5 and h > 50]

# Group by X position
x_groups = defaultdict(list)
for idx, xmin, xmax, ymin, ymax, w, h, s, e in vert_members:
    x_mid = round((xmin + xmax) / 2, 0)
    x_groups[x_mid].append((idx, ymin, ymax, h))

for x_key in sorted(x_groups.keys()):
    group = sorted(x_groups[x_key], key=lambda g: g[1])
    # Compute total extent
    total_ymin = min(g[1] for g in group)
    total_ymax = max(g[2] for g in group)
    total_span = total_ymax - total_ymin
    
    # Also try contiguous spans (connect touching segments)
    segments = [(g[1], g[2]) for g in group]
    # Merge overlapping/touching segments
    merged_segs = [segments[0]]
    for ymin, ymax in segments[1:]:
        if ymin <= merged_segs[-1][1] + 5:  # within 5 units gap
            merged_segs[-1] = (merged_segs[-1][0], max(merged_segs[-1][1], ymax))
        else:
            merged_segs.append((ymin, ymax))
    
    for seg_ymin, seg_ymax in merged_segs:
        seg_span = seg_ymax - seg_ymin
        if abs(seg_span - TARGET) < 15:
            print(f"  X ~ {x_key:.0f}: merged segment Y: {seg_ymin:.2f}..{seg_ymax:.2f}  span={seg_span:.2f}  *** CLOSE TO {TARGET} ***")
            for idx, ymin, ymax, h in group:
                if ymin >= seg_ymin - 5 and ymax <= seg_ymax + 5:
                    print(f"    [PL-{idx:03d}]  y: {ymin:.2f}..{ymax:.2f}  h={h:.2f}")

# Also check: maybe 850.8 is between two HORIZONTAL structural levels
print()
print("=" * 80)
print("Y-LEVEL DIFFERENCES CLOSE TO 850.8")
print("=" * 80)

# Gather all Y-positions that appear at endpoints of polylines
y_endpoints = set()
for x, y, idx, pos in endpoints:
    y_endpoints.add(round(y, 2))

y_list = sorted(y_endpoints)
for i in range(len(y_list)):
    for j in range(i + 1, len(y_list)):
        diff = abs(y_list[j] - y_list[i])
        if abs(diff - TARGET) < 1.0:
            print(f"  Y1={y_list[i]:.2f}  Y2={y_list[j]:.2f}  diff={diff:.4f}")

print()
print("DONE.")
