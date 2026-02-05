import ezdxf
import math

DXF_PATH = r"D:\Google\Drafter\test\output\1-2_YZ_rot.dxf"
TARGET = 850.8

doc = ezdxf.readfile(DXF_PATH)
msp = doc.modelspace()
polylines = list(msp.query("LWPOLYLINE"))

all_points = []
for idx, pl in enumerate(polylines):
    for p in pl.get_points(format="xy"):
        all_points.append((p[0], p[1], idx))

# The #1 closest match found earlier:
# dist=850.8001  P1=(-653.3647,-2072.8729)[PL-046]  P2=(-956.2268,-1277.8034)[PL-344]
# But these are INTERIOR curve vertices, not structural corners.

# The NEAR-VERTICAL matches are more likely annotation endpoints:
# #1  dist=850.8055  dx=2.74  dy=850.80  P1=(-770.79,-809.07)[PL-053]  P2=(-773.53,-1659.87)[PL-306]
# #2  dist=850.8380  dx=0.70  dy=850.84  P1=(-812.79,-1607.77)[PL-002]  P2=(-812.09,-756.94)[PL-059]
# #5  dist=850.9817  dx=0.95  dy=850.98  P1=(-714.96,-654.89)[PL-029]  P2=(-714.01,-1505.87)[PL-033]

# Let us examine these polylines to understand what they represent
print("=" * 80)
print("ANALYSIS OF CANDIDATE VERTICAL-SPAN POLYLINES FOR 850.8 DIMENSION")
print("=" * 80)

candidates = [
    (53, 306, "Near-vertical pair #1: dx=2.74"),
    (2, 59,   "Near-vertical pair #2: dx=0.70"),
    (29, 33,  "Near-vertical pair #5: dx=0.95"),
    (32, 35,  "Vertical lines PL-032/035 (height=374.7)"),
]

for pla_id, plb_id, desc in candidates:
    print(f"\n--- {desc} ---")
    for plid in [pla_id, plb_id]:
        pl = polylines[plid]
        pts = list(pl.get_points(format="xy"))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        s2e = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        print(f"  [PL-{plid:03d}]  pts={len(pts)}  closed={pl.closed}  "
              f"start=({pts[0][0]:.2f},{pts[0][1]:.2f})  end=({pts[-1][0]:.2f},{pts[-1][1]:.2f})  "
              f"s2e_dist={s2e:.2f}  size=({w:.2f}x{h:.2f})")

# Now identify the overall structure shape
print()
print("=" * 80)
print("STRUCTURAL SHAPE IDENTIFICATION")
print("=" * 80)

# Find polylines that are essentially vertical lines (width < 1)
vert_lines = []
for idx, pl in enumerate(polylines):
    pts = list(pl.get_points(format="xy"))
    if len(pts) < 2:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if w < 1.0 and h > 100:
        vert_lines.append((idx, min(xs), max(xs), min(ys), max(ys), h))

vert_lines.sort(key=lambda v: v[5], reverse=True)
print(f"\nVertical line-like polylines (width<1, height>100): {len(vert_lines)}")
for idx, xmin, xmax, ymin, ymax, h in vert_lines[:15]:
    print(f"  [PL-{idx:03d}]  x~{(xmin+xmax)/2:.2f}  y: {ymin:.2f}..{ymax:.2f}  height={h:.2f}")

# Find polylines that are essentially horizontal lines (height < 1)
horiz_lines = []
for idx, pl in enumerate(polylines):
    pts = list(pl.get_points(format="xy"))
    if len(pts) < 2:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if h < 1.0 and w > 100:
        horiz_lines.append((idx, min(xs), max(xs), min(ys), max(ys), w))

horiz_lines.sort(key=lambda v: v[5], reverse=True)
print(f"\nHorizontal line-like polylines (height<1, width>100): {len(horiz_lines)}")
for idx, xmin, xmax, ymin, ymax, w in horiz_lines[:15]:
    print(f"  [PL-{idx:03d}]  y~{(ymin+ymax)/2:.2f}  x: {xmin:.2f}..{xmax:.2f}  width={w:.2f}")

# Find the top and bottom Y of each vertical line to understand structure height spans
print()
print("=" * 80)
print("VERTICAL SPAN ANALYSIS - pairs of vertical lines at same X")
print("=" * 80)

# Group vertical lines by approximate X
from collections import defaultdict
x_groups = defaultdict(list)
for idx, xmin, xmax, ymin, ymax, h in vert_lines:
    x_key = round((xmin + xmax) / 2, 0)
    x_groups[x_key].append((idx, ymin, ymax, h))

for x_key in sorted(x_groups.keys()):
    group = x_groups[x_key]
    if len(group) < 2:
        continue
    # Find total vertical span
    all_ymin = min(g[1] for g in group)
    all_ymax = max(g[2] for g in group)
    total_span = all_ymax - all_ymin
    if abs(total_span - TARGET) < 20:
        print(f"\n  X ~ {x_key:.0f}:  total Y span = {total_span:.4f}  {'*** CLOSE TO 850.8 ***' if abs(total_span - TARGET) < 5 else ''}")
        for idx, ymin, ymax, h in sorted(group, key=lambda g: g[1]):
            print(f"    [PL-{idx:03d}]  y: {ymin:.2f}..{ymax:.2f}  h={h:.2f}")

# Also check combined spans
print()
print("=" * 80)
print("ALL VERTICAL LINE SPANS (total height per X-group)")
for x_key in sorted(x_groups.keys()):
    group = x_groups[x_key]
    all_ymin = min(g[1] for g in group)
    all_ymax = max(g[2] for g in group)
    total_span = all_ymax - all_ymin
    n = len(group)
    marker = " <--- CLOSE TO 850.8!" if abs(total_span - TARGET) < 5 else ""
    print(f"  X ~ {x_key:8.0f}:  n={n}  Y: {all_ymin:.2f}..{all_ymax:.2f}  span={total_span:.2f}{marker}")

print()
print("DONE.")
