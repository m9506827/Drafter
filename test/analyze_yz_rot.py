import ezdxf
import math
from collections import Counter

DXF_PATH = r"D:\Google\Drafter\test\output\1-2_YZ_rot.dxf"
TARGET = 850.8

doc = ezdxf.readfile(DXF_PATH)
msp = doc.modelspace()

polylines = list(msp.query("LWPOLYLINE"))
print(f"Total LWPOLYLINE entities: {len(polylines)}")
print("=" * 80)

all_points = []
all_points_flat = []

gxmin = float("inf")
gymin = float("inf")
gxmax = float("-inf")
gymax = float("-inf")

for idx, pl in enumerate(polylines):
    pts = list(pl.get_points(format="xy"))
    n = len(pts)
    if n == 0:
        print(f"[PL-{idx:03d}]  (empty)")
        continue

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = xmax - xmin
    h = ymax - ymin

    if xmin < gxmin: gxmin = xmin
    if ymin < gymin: gymin = ymin
    if xmax > gxmax: gxmax = xmax
    if ymax > gymax: gymax = ymax

    for p in pts:
        all_points.append((p[0], p[1], idx))
        all_points_flat.append((p[0], p[1]))

    closed = pl.closed
    print(f"[PL-{idx:03d}]  points={n:4d}  closed={closed}  bbox=({xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f})  size=({w:.4f} x {h:.4f})")

print()
print("=" * 80)
print("GLOBAL BOUNDING BOX")
print(f"  X: {gxmin:.4f}  ..  {gxmax:.4f}   (width  = {gxmax - gxmin:.4f})")
print(f"  Y: {gymin:.4f}  ..  {gymax:.4f}   (height = {gymax - gymin:.4f})")
diag = math.hypot(gxmax - gxmin, gymax - gymin)
print(f"  Diagonal of global bbox: {diag:.4f}")

print()
print("=" * 80)
print("EXTREME POINTS")

topmost    = max(all_points, key=lambda p: p[1])
bottommost = min(all_points, key=lambda p: p[1])
leftmost   = min(all_points, key=lambda p: p[0])
rightmost  = max(all_points, key=lambda p: p[0])

print(f"  Topmost    : ({topmost[0]:.4f}, {topmost[1]:.4f})   [PL-{topmost[2]:03d}]")
print(f"  Bottommost : ({bottommost[0]:.4f}, {bottommost[1]:.4f})   [PL-{bottommost[2]:03d}]")
print(f"  Leftmost   : ({leftmost[0]:.4f}, {leftmost[1]:.4f})   [PL-{leftmost[2]:03d}]")
print(f"  Rightmost  : ({rightmost[0]:.4f}, {rightmost[1]:.4f})   [PL-{rightmost[2]:03d}]")

print()
print("=" * 80)
print("KEY STRUCTURAL CORNERS  (vertex closest to each global bbox corner)")

corners = {
    "Top-Left":     (gxmin, gymax),
    "Top-Right":    (gxmax, gymax),
    "Bottom-Left":  (gxmin, gymin),
    "Bottom-Right": (gxmax, gymin),
}

for label, (cx, cy) in corners.items():
    best = min(all_points, key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
    dist = math.hypot(best[0] - cx, best[1] - cy)
    print(f"  {label:13s}: ({best[0]:.4f}, {best[1]:.4f})  [PL-{best[2]:03d}]  dist_to_corner={dist:.4f}")

print()
print("=" * 80)
print(f"SEARCHING FOR POINT PAIRS WITH DIAGONAL DISTANCE ~ {TARGET}")

unique_map = {}
for x, y, i in all_points:
    key = (round(x, 4), round(y, 4))
    if key not in unique_map:
        unique_map[key] = (x, y, i)
unique_pts = list(unique_map.values())
print(f"  Unique vertices: {len(unique_pts)}")

for TOL in [1.0, 2.0, 5.0, 10.0, 20.0]:
    matches = []
    for i in range(len(unique_pts)):
        x1, y1, pi1 = unique_pts[i]
        for j in range(i + 1, len(unique_pts)):
            x2, y2, pi2 = unique_pts[j]
            dx = abs(x2 - x1)
            if dx > TARGET + TOL:
                continue
            dy = abs(y2 - y1)
            if dy > TARGET + TOL:
                continue
            d = math.hypot(dx, dy)
            if abs(d - TARGET) < TOL:
                matches.append(((x1, y1, pi1), (x2, y2, pi2), d))
    matches.sort(key=lambda m: abs(m[2] - TARGET))
    print(f"\n  Tolerance={TOL}: {len(matches)} pairs found")
    for k, (p1, p2, d) in enumerate(matches[:20]):
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        print(f"    #{k+1:3d}  dist={d:.4f}  delta=({dx:.4f}, {dy:.4f})"
              f"  P1=({p1[0]:.4f},{p1[1]:.4f})[PL-{p1[2]:03d}]"
              f"  P2=({p2[0]:.4f},{p2[1]:.4f})[PL-{p2[2]:03d}]")
    if len(matches) > 0:
        break

print()
print("=" * 80)
print("ALL ENTITY TYPES IN MODELSPACE")
types = Counter(e.dxftype() for e in msp)
for t, c in types.most_common():
    print(f"  {t:20s}: {c}")

lines = list(msp.query("LINE"))
print(f"\nLINE entities: {len(lines)}")
for idx2, ln in enumerate(lines):
    s = ln.dxf.start
    e = ln.dxf.end
    d = math.hypot(e.x - s.x, e.y - s.y)
    print(f"  [LINE-{idx2:03d}]  len={d:.4f}  ({s.x:.4f},{s.y:.4f}) -> ({e.x:.4f},{e.y:.4f})")
    if abs(d - TARGET) < 20.0:
        print(f"     *** CLOSE TO TARGET {TARGET} (diff={d - TARGET:.4f}) ***")

print()
print("=" * 80)
print("UNIQUE Y LEVELS (rounded to 1 decimal)")
y_vals = sorted(set(round(p[1], 1) for p in all_points_flat))
for y in y_vals:
    count = sum(1 for p in all_points_flat if abs(p[1] - y) < 0.15)
    print(f"  Y = {y:12.4f}   ({count} vertices)")

print()
print("UNIQUE X LEVELS (rounded to 1 decimal)")
x_vals = sorted(set(round(p[0], 1) for p in all_points_flat))
for x in x_vals:
    count = sum(1 for p in all_points_flat if abs(p[0] - x) < 0.15)
    print(f"  X = {x:12.4f}   ({count} vertices)")

print()
print("=" * 80)
print("PAIRWISE DISTANCES BETWEEN ALL 4 EXTREME POINTS")
extremes = {
    "Topmost": (topmost[0], topmost[1]),
    "Bottommost": (bottommost[0], bottommost[1]),
    "Leftmost": (leftmost[0], leftmost[1]),
    "Rightmost": (rightmost[0], rightmost[1]),
}
for n1, p1 in extremes.items():
    for n2, p2 in extremes.items():
        if n1 >= n2:
            continue
        d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        print(f"  {n1:12s} <-> {n2:12s}  dist={d:.4f}")

print()
print("DONE.")
