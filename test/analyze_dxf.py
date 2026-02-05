"""
Analyze LWPOLYLINE entities in a DXF file.

For each polyline: print point count, bounding box, first/last point.
Global statistics:
  - Global bounding box across all polylines
  - Circle fitting for large arc-like polylines (xmax > 700, vertical span > 200)
  - Horizontal bars (Y variation < 5, X span > 100) and their average Y
  - Global minimum Y
"""

import ezdxf
import numpy as np
from pathlib import Path

DXF_PATH = Path(r"D:\Google\Drafter\test\output\1-2_XY_rot.dxf")


def fit_circle_least_squares(points: np.ndarray):
    """
    Fit a circle to 2D points using algebraic least-squares (Kasa method).

    Minimises  ||Ax - b||^2  where the circle equation is rewritten as:
        x^2 + y^2 = Dx + Ey + F
    so  centre = (D/2, E/2),  radius = sqrt(F + D^2/4 + E^2/4).

    Returns (cx, cy, radius).
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    result, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = result
    cx = D / 2.0
    cy = E / 2.0
    radius = np.sqrt(F + cx ** 2 + cy ** 2)
    return cx, cy, radius


def main():
    print(f"Reading: {DXF_PATH}\n")
    doc = ezdxf.readfile(str(DXF_PATH))
    msp = doc.modelspace()

    polylines = list(msp.query("LWPOLYLINE"))
    print(f"Total LWPOLYLINE entities found: {len(polylines)}\n")
    print("=" * 90)

    # Accumulators for global stats
    global_xmin = float("inf")
    global_ymin = float("inf")
    global_xmax = float("-inf")
    global_ymax = float("-inf")

    arc_candidates = []      # (index, points_array, bbox)
    horizontal_bars = []     # (index, avg_y, xspan, bbox)

    for idx, poly in enumerate(polylines):
        # get_points returns tuples of (x, y [, start_width, end_width, bulge])
        # format='xy' gives only x, y
        pts = list(poly.get_points(format="xy"))
        n = len(pts)
        arr = np.array(pts)

        xmin, ymin = arr.min(axis=0)
        xmax, ymax = arr.max(axis=0)
        xspan = xmax - xmin
        yspan = ymax - ymin

        # Update global bbox
        global_xmin = min(global_xmin, xmin)
        global_ymin = min(global_ymin, ymin)
        global_xmax = max(global_xmax, xmax)
        global_ymax = max(global_ymax, ymax)

        # Print per-polyline info
        print(f"Polyline #{idx:3d}  |  Points: {n:5d}  |  "
              f"BBox: ({xmin:10.3f}, {ymin:10.3f}, {xmax:10.3f}, {ymax:10.3f})  |  "
              f"Span: ({xspan:.3f} x {yspan:.3f})")
        print(f"    First: ({pts[0][0]:.4f}, {pts[0][1]:.4f})   "
              f"Last: ({pts[-1][0]:.4f}, {pts[-1][1]:.4f})")

        # Check arc candidate: xmax > 700 and vertical span > 200
        if xmax > 700 and yspan > 200:
            arc_candidates.append((idx, arr, (xmin, ymin, xmax, ymax)))

        # Check horizontal bar: Y variation < 5 and X span > 100
        if yspan < 5 and xspan > 100:
            avg_y = arr[:, 1].mean()
            horizontal_bars.append((idx, avg_y, xspan, (xmin, ymin, xmax, ymax)))

    print("=" * 90)

    # ---- Global Bounding Box ----
    print(f"\n{'=' * 90}")
    print("GLOBAL STATISTICS")
    print(f"{'=' * 90}")
    print(f"\nGlobal Bounding Box:")
    print(f"  xmin = {global_xmin:.4f}")
    print(f"  ymin = {global_ymin:.4f}")
    print(f"  xmax = {global_xmax:.4f}")
    print(f"  ymax = {global_ymax:.4f}")
    print(f"  Total span: {global_xmax - global_xmin:.4f} x {global_ymax - global_ymin:.4f}")

    # ---- Circle Fitting for Arc Candidates ----
    print(f"\n{'-' * 90}")
    print(f"ARC CANDIDATES (xmax > 700 and vertical span > 200): {len(arc_candidates)} found")
    print(f"{'-' * 90}")
    if arc_candidates:
        for idx, arr, bbox in arc_candidates:
            cx, cy, radius = fit_circle_least_squares(arr)
            # Compute fitting residuals
            distances = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
            residual_mean = np.abs(distances - radius).mean()
            residual_max = np.abs(distances - radius).max()
            print(f"\n  Polyline #{idx}  ({len(arr)} points)")
            print(f"    BBox:   ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})")
            print(f"    Circle centre:  ({cx:.4f}, {cy:.4f})")
            print(f"    Circle radius:  {radius:.4f}")
            print(f"    Fit residual:   mean={residual_mean:.6f}  max={residual_max:.6f}")
    else:
        print("  (none)")

    # ---- Horizontal Bars ----
    print(f"\n{'-' * 90}")
    print(f"HORIZONTAL BARS (Y variation < 5, X span > 100): {len(horizontal_bars)} found")
    print(f"{'-' * 90}")
    if horizontal_bars:
        # Sort by average Y descending so top bar is first
        horizontal_bars.sort(key=lambda t: t[1], reverse=True)
        for idx, avg_y, xspan, bbox in horizontal_bars:
            print(f"  Polyline #{idx}  |  Avg Y = {avg_y:.4f}  |  X span = {xspan:.3f}  |  "
                  f"BBox: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})")

        top_bar_y = horizontal_bars[0][1]
        bottom_bar_y = horizontal_bars[-1][1]
        print(f"\n  Top    horizontal bar avg Y: {top_bar_y:.4f}")
        print(f"  Bottom horizontal bar avg Y: {bottom_bar_y:.4f}")
    else:
        print("  (none)")

    # ---- Global Minimum Y ----
    print(f"\n{'-' * 90}")
    print(f"GLOBAL MINIMUM Y: {global_ymin:.4f}")
    print(f"{'-' * 90}")

    # ---- Summary for Dimension Annotations ----
    print(f"\n{'=' * 90}")
    print("SUMMARY FOR DIMENSION ANNOTATIONS")
    print(f"{'=' * 90}")
    if arc_candidates:
        for i, (idx, arr, bbox) in enumerate(arc_candidates):
            cx, cy, radius = fit_circle_least_squares(arr)
            print(f"  Arc #{i}: centre=({cx:.4f}, {cy:.4f}), radius={radius:.4f}")
    if horizontal_bars:
        print(f"  Top horizontal bar Y:    {horizontal_bars[0][1]:.4f}")
        print(f"  Bottom horizontal bar Y: {horizontal_bars[-1][1]:.4f}")
    print(f"  Global min Y:            {global_ymin:.4f}")
    print()


if __name__ == "__main__":
    main()
