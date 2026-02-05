import ezdxf
import math

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

print("=" * 80)
print("NEAR-VERTICAL DISTANCES CLOSE TO 850.8  (dx < 5)")
print("=" * 80)

# Deduplicate more aggressively for this search
unique_map = {}
for x, y, i in all_points:
    key = (round(x, 1), round(y, 1))
    if key not in unique_map:
        unique_map[key] = (x, y, i)
unique_pts = list(unique_map.values())
print(f"Unique vertices (1dp): {len(unique_pts)}")

vert_matches = []
for i in range(len(unique_pts)):
    x1, y1, pi1 = unique_pts[i]
    for j in range(i + 1, len(unique_pts)):
        x2, y2, pi2 = unique_pts[j]
        dx = abs(x2 - x1)
        if dx > 5:
            continue
        dy = abs(y2 - y1)
        if abs(dy - TARGET) > 5:
            continue
        d = math.hypot(dx, dy)
        vert_matches.append(((x1, y1, pi1), (x2, y2, pi2), d, dx, dy))

vert_matches.sort(key=lambda m: abs(m[2] - TARGET))
print(f"\nNear-vertical pairs (dx<5, dy~850.8): {len(vert_matches)}")
for k, (p1, p2, d, dx, dy) in enumerate(vert_matches[:30]):
    print(f"  #{k+1:3d}  dist={d:.4f}  dx={dx:.4f}  dy={dy:.4f}"
          f"  P1=({p1[0]:.4f},{p1[1]:.4f})[PL-{p1[2]:03d}]"
          f"  P2=({p2[0]:.4f},{p2[1]:.4f})[PL-{p2[2]:03d}]")

print()
print("=" * 80)
print("TRUE DIAGONAL INTERPRETATION")
print("=" * 80)
print("The 850.8 dimension is described as a DIAGONAL span.")
print("Looking at closest matches (dist exactly 850.8):")
print()

# Get the very best matches (closest to exactly 850.8)
all_matches = []
for i in range(len(unique_pts)):
    x1, y1, pi1 = unique_pts[i]
    for j in range(i + 1, len(unique_pts)):
        x2, y2, pi2 = unique_pts[j]
        dx = abs(x2 - x1)
        if dx > TARGET + 1:
            continue
        dy = abs(y2 - y1)
        if dy > TARGET + 1:
            continue
        d = math.hypot(dx, dy)
        if abs(d - TARGET) < 0.01:
            all_matches.append(((x1, y1, pi1), (x2, y2, pi2), d, dx, dy))

all_matches.sort(key=lambda m: abs(m[2] - TARGET))
print(f"Pairs within 0.01 of {TARGET}: {len(all_matches)}")
for k, (p1, p2, d, dx, dy) in enumerate(all_matches[:20]):
    angle = math.degrees(math.atan2(dy, dx))
    print(f"  #{k+1:3d}  dist={d:.6f}  dx={dx:.4f}  dy={dy:.4f}  angle={angle:.1f}deg"
          f"  P1=({p1[0]:.4f},{p1[1]:.4f})[PL-{p1[2]:03d}]"
          f"  P2=({p2[0]:.4f},{p2[1]:.4f})[PL-{p2[2]:03d}]")

print()
print("=" * 80)
print("POLYLINES PL-009 and PL-010 DETAIL  (two largest diag polylines in structure)")
print("=" * 80)
for plid in [9, 10]:
    pl = polylines[plid]
    pts = list(pl.get_points(format="xy"))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    print(f"\n[PL-{plid:03d}]  points={len(pts)}  closed={pl.closed}")
    print(f"  X range: {min(xs):.4f} .. {max(xs):.4f}")
    print(f"  Y range: {min(ys):.4f} .. {max(ys):.4f}")
    print(f"  Start: ({pts[0][0]:.4f}, {pts[0][1]:.4f})")
    print(f"  End:   ({pts[-1][0]:.4f}, {pts[-1][1]:.4f})")
    start_to_end = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    print(f"  Start-to-End distance: {start_to_end:.4f}")
    print(f"  All points:")
    for i, p in enumerate(pts):
        print(f"    [{i:2d}] ({p[0]:.4f}, {p[1]:.4f})")

# Also check PL-025 and PL-026 (the triangle top)
print()
print("=" * 80)
print("POLYLINES PL-025 and PL-026  (top triangle area)")
for plid in [25, 26]:
    pl = polylines[plid]
    pts = list(pl.get_points(format="xy"))
    print(f"\n[PL-{plid:03d}]  points={len(pts)}  closed={pl.closed}")
    print(f"  Start: ({pts[0][0]:.4f}, {pts[0][1]:.4f})")
    print(f"  End:   ({pts[-1][0]:.4f}, {pts[-1][1]:.4f})")
    start_to_end = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    print(f"  Start-to-End distance: {start_to_end:.4f}")

# Check what the angle of PL-009 diagonal is
print()
print("=" * 80)
print("DIAGONAL ANGLE ANALYSIS FOR PL-009 (main diagonal member)")
pl9 = polylines[9]
pts9 = list(pl9.get_points(format="xy"))
sx, sy = pts9[0]
ex, ey = pts9[-1]
dx = ex - sx
dy = ey - sy
angle = math.degrees(math.atan2(abs(dy), abs(dx)))
diag = math.hypot(dx, dy)
print(f"  Start: ({sx:.4f}, {sy:.4f})")
print(f"  End:   ({ex:.4f}, {ey:.4f})")
print(f"  dx={dx:.4f}  dy={dy:.4f}")
print(f"  Diagonal length: {diag:.4f}")
print(f"  Angle from horizontal: {angle:.1f} degrees")
print(f"  This polyline is a CURVE approximating a diagonal member.")
print(f"  The straight-line distance between its endpoints = {diag:.4f}")

# Check distances using PL-009/010 endpoints
print()
print("=" * 80)
print("ENDPOINT-TO-ENDPOINT DISTANCES FOR DIAGONAL MEMBERS")
for pla_id, plb_id in [(9, 10), (25, 26), (9, 25), (9, 26), (10, 25), (10, 26)]:
    pla = polylines[pla_id]
    plb = polylines[plb_id]
    ptsa = list(pla.get_points(format="xy"))
    ptsb = list(plb.get_points(format="xy"))
    for label_a, pa in [("start", ptsa[0]), ("end", ptsa[-1])]:
        for label_b, pb in [("start", ptsb[0]), ("end", ptsb[-1])]:
            d = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            if abs(d - TARGET) < 50:
                print(f"  PL-{pla_id:03d}.{label_a} <-> PL-{plb_id:03d}.{label_b}: dist={d:.4f}"
                      f"  ({pa[0]:.2f},{pa[1]:.2f}) <-> ({pb[0]:.2f},{pb[1]:.2f})")

print("\nDONE.")
