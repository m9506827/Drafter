import ezdxf
import math
from collections import Counter

DXF_PATH = r"D:\Google\Drafter\test\output\1-2_YZ_rot.dxf"
TARGET = 850.8

doc = ezdxf.readfile(DXF_PATH)
msp = doc.modelspace()

polylines = list(msp.query("LWPOLYLINE"))

all_points = []
for idx, pl in enumerate(polylines):
    pts = list(pl.get_points(format="xy"))
    for p in pts:
        all_points.append((p[0], p[1], idx))

# Deduplicate by rounding to 2 decimal places
unique_map = {}
for x, y, i in all_points:
    key = (round(x, 2), round(y, 2))
    if key not in unique_map:
        unique_map[key] = (x, y, i)
unique_pts = list(unique_map.values())

print(f"Total polylines: {len(polylines)}, Total vertices: {len(all_points)}, Unique (2dp): {len(unique_pts)}")

# --- Cluster Y levels with high vertex count (structural levels) ---
print("\n" + "=" * 80)
print("STRUCTURAL Y-LEVELS (clusters with >= 20 vertices, 5-unit window)")

y_all = sorted([p[1] for p in all_points])
clusters = []
used = set()
for y in y_all:
    yr = round(y, 0)
    if yr in used:
        continue
    count = sum(1 for yy in y_all if abs(yy - yr) < 5)
    if count >= 20:
        clusters.append((yr, count))
        used.add(yr)
# Merge close clusters
merged = []
for yr, cnt in sorted(clusters):
    if merged and abs(yr - merged[-1][0]) < 10:
        merged[-1] = ((merged[-1][0] * merged[-1][1] + yr * cnt) / (merged[-1][1] + cnt), merged[-1][1] + cnt)
    else:
        merged.append((yr, cnt))

print(f"Found {len(merged)} major Y-level clusters:")
for y, cnt in sorted(merged, reverse=True):
    print(f"  Y ~ {y:10.1f}   ({cnt} vertices)")

# --- Cluster X levels ---
print("\n" + "=" * 80)
print("STRUCTURAL X-LEVELS (clusters with >= 20 vertices, 5-unit window)")

x_all = sorted([p[0] for p in all_points])
clusters_x = []
used_x = set()
for x in x_all:
    xr = round(x, 0)
    if xr in used_x:
        continue
    count = sum(1 for xx in x_all if abs(xx - xr) < 5)
    if count >= 20:
        clusters_x.append((xr, count))
        used_x.add(xr)
merged_x = []
for xr, cnt in sorted(clusters_x):
    if merged_x and abs(xr - merged_x[-1][0]) < 10:
        merged_x[-1] = ((merged_x[-1][0] * merged_x[-1][1] + xr * cnt) / (merged_x[-1][1] + cnt), merged_x[-1][1] + cnt)
    else:
        merged_x.append((xr, cnt))

print(f"Found {len(merged_x)} major X-level clusters:")
for x, cnt in sorted(merged_x, reverse=True):
    print(f"  X ~ {x:10.1f}   ({cnt} vertices)")

# --- Find structure corners based on major clusters ---
print("\n" + "=" * 80)
print("CORNER ANALYSIS")
print("Trying to find the structural outline corner points...")

# Get the extreme Y clusters (top and bottom of structure)
y_sorted = sorted(merged, key=lambda c: c[0], reverse=True)
y_top = y_sorted[0][0]
y_bot = y_sorted[-1][0]

# Get the extreme X clusters 
x_sorted = sorted(merged_x, key=lambda c: c[0])
x_left = x_sorted[0][0]
x_right = x_sorted[-1][0]

print(f"\n  Topmost cluster Y  ~ {y_top:.1f}")
print(f"  Bottommost cluster Y ~ {y_bot:.1f}")
print(f"  Leftmost cluster X  ~ {x_left:.1f}")
print(f"  Rightmost cluster X ~ {x_right:.1f}")

# For each corner region, find the actual vertices
def find_vertices_near(x_target, y_target, tol_x=30, tol_y=30):
    nearby = [(p[0], p[1], p[2]) for p in all_points 
              if abs(p[0] - x_target) < tol_x and abs(p[1] - y_target) < tol_y]
    return nearby

corners_names = ["TopRight", "TopLeft", "BotRight", "BotLeft"]
corners_targets = [
    (x_right, y_top),
    (x_left, y_top),
    (x_right, y_bot),
    (x_left, y_bot),
]

corner_pts = {}
for name, (xt, yt) in zip(corners_names, corners_targets):
    nearby = find_vertices_near(xt, yt, 50, 50)
    if nearby:
        # Find the one closest to the target
        best = min(nearby, key=lambda p: math.hypot(p[0] - xt, p[1] - yt))
        corner_pts[name] = best
        print(f"  {name:12s}: ({best[0]:.4f}, {best[1]:.4f})  [PL-{best[2]:03d}]")
    else:
        print(f"  {name:12s}: NO VERTEX FOUND near ({xt:.1f}, {yt:.1f})")

# --- Diagonal distances between corners ---
print("\n" + "=" * 80)
print("CORNER-TO-CORNER DISTANCES")
for n1 in corner_pts:
    for n2 in corner_pts:
        if n1 >= n2:
            continue
        p1 = corner_pts[n1]
        p2 = corner_pts[n2]
        d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        marker = " *** CLOSE TO 850.8 ***" if abs(d - TARGET) < 20 else ""
        print(f"  {n1:12s} <-> {n2:12s}  dist={d:.4f}  dx={dx:.4f}  dy={dy:.4f}{marker}")

# --- Focused: find the BEST pair for 850.8 among structural/corner vertices ---
print("\n" + "=" * 80)
print("FOCUSED SEARCH: 850.8 among vertices near structural Y-levels")

# Gather vertices near major Y-levels
structural_verts = []
for yc, _ in merged:
    for p in unique_pts:
        if abs(p[1] - yc) < 8:
            structural_verts.append(p)

# Deduplicate
sv_map = {}
for x, y, i in structural_verts:
    key = (round(x, 1), round(y, 1))
    if key not in sv_map:
        sv_map[key] = (x, y, i)
structural_verts = list(sv_map.values())
print(f"  Structural vertices near major Y-levels: {len(structural_verts)}")

matches = []
for i in range(len(structural_verts)):
    x1, y1, pi1 = structural_verts[i]
    for j in range(i + 1, len(structural_verts)):
        x2, y2, pi2 = structural_verts[j]
        d = math.hypot(x2 - x1, y2 - y1)
        if abs(d - TARGET) < 0.5:
            matches.append(((x1, y1, pi1), (x2, y2, pi2), d))

matches.sort(key=lambda m: abs(m[2] - TARGET))
print(f"  Pairs with dist within 0.5 of {TARGET}: {len(matches)}")
for k, (p1, p2, d) in enumerate(matches[:15]):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    print(f"    #{k+1:3d}  dist={d:.4f}  dx={dx:.4f}  dy={dy:.4f}"
          f"  P1=({p1[0]:.4f},{p1[1]:.4f})[PL-{p1[2]:03d}]"
          f"  P2=({p2[0]:.4f},{p2[1]:.4f})[PL-{p2[2]:03d}]")

# --- Also check pure vertical and horizontal distances ---
print("\n" + "=" * 80)
print("CHECKING PURE VERTICAL/HORIZONTAL DISTANCES ~ 850.8 AMONG ALL UNIQUE POINTS")

vert_matches = []
horiz_matches = []
for i in range(len(unique_pts)):
    x1, y1, pi1 = unique_pts[i]
    for j in range(i + 1, len(unique_pts)):
        x2, y2, pi2 = unique_pts[j]
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        if abs(dy - TARGET) < 0.5 and dx < 1.0:
            vert_matches.append(((x1, y1, pi1), (x2, y2, pi2), dy, dx))
        if abs(dx - TARGET) < 0.5 and dy < 1.0:
            horiz_matches.append(((x1, y1, pi1), (x2, y2, pi2), dx, dy))

print(f"\n  Vertical (same X, dy~{TARGET}): {len(vert_matches)}")
for k, (p1, p2, dy, dx) in enumerate(vert_matches[:10]):
    print(f"    #{k+1}  dy={dy:.4f}  dx={dx:.4f}  P1=({p1[0]:.4f},{p1[1]:.4f})  P2=({p2[0]:.4f},{p2[1]:.4f})")

print(f"\n  Horizontal (same Y, dx~{TARGET}): {len(horiz_matches)}")
for k, (p1, p2, dx, dy) in enumerate(horiz_matches[:10]):
    print(f"    #{k+1}  dx={dx:.4f}  dy={dy:.4f}  P1=({p1[0]:.4f},{p1[1]:.4f})  P2=({p2[0]:.4f},{p2[1]:.4f})")

# --- Print all polylines sorted by size (largest first) for structural overview ---
print("\n" + "=" * 80)
print("LARGEST POLYLINES (by bounding box diagonal, top 20)")

pl_info = []
for idx, pl in enumerate(polylines):
    pts = list(pl.get_points(format="xy"))
    if len(pts) == 0:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    diag = math.hypot(xmax - xmin, ymax - ymin)
    pl_info.append((idx, len(pts), xmin, ymin, xmax, ymax, diag, pl.closed))

pl_info.sort(key=lambda x: x[6], reverse=True)
for idx, npts, xmin, ymin, xmax, ymax, diag, closed in pl_info[:20]:
    w = xmax - xmin
    h = ymax - ymin
    print(f"  [PL-{idx:03d}]  diag={diag:10.4f}  pts={npts:4d}  closed={closed}  "
          f"bbox=({xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f})  size=({w:.2f}x{h:.2f})")

print("\nDONE.")
