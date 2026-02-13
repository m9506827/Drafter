"""Check Drawing 2 for specific issue details."""
import ezdxf, os

dxf_path = r'd:\Google\Drafter\output\2-2_2.dxf'
doc = ezdxf.readfile(dxf_path)
msp = doc.modelspace()

# Get ALL text values
print("=== ALL TEXT content ===")
all_text = []
for e in msp:
    if e.dxftype() == 'TEXT':
        t = e.dxf.text.strip()
        pos = (e.dxf.insert[0], e.dxf.insert[1]) if hasattr(e.dxf, 'insert') else (0,0)
        all_text.append((t, pos))
    elif e.dxftype() == 'MTEXT':
        t = e.text.strip()
        pos = (e.dxf.insert[0], e.dxf.insert[1]) if hasattr(e.dxf, 'insert') else (0,0)
        all_text.append((t, pos))

# Sort by position to understand layout
for t, pos in sorted(all_text, key=lambda x: (-x[1][1], x[1][0])):
    print(f"  ({pos[0]:7.1f}, {pos[1]:7.1f}) {t}")

# Check if 267.2 appears anywhere
print("\n=== Search for '267' ===")
for t, pos in all_text:
    if '267' in t:
        print(f"  FOUND: ({pos[0]:.1f}, {pos[1]:.1f}) {t}")

# Check dimension lines near 230.2
print("\n=== Dimension lines with vertical span ===")
lines = [e for e in msp if e.dxftype() == 'LINE']
for l in lines:
    sx, sy = l.dxf.start[0], l.dxf.start[1]
    ex, ey = l.dxf.end[0], l.dxf.end[1]
    dx = abs(sx - ex)
    dy = abs(sy - ey)
    # Vertical dimension lines (small dx, large dy)
    if dx < 3 and dy > 10:
        print(f"  x={sx:.1f}, y=[{min(sy,ey):.1f}, {max(sy,ey):.1f}], span={dy:.1f}")

# Find bracket circles in top view (larger radius)
print("\n=== Top view circles (brackets, r > 2) ===")
circles = [e for e in msp if e.dxftype() == 'CIRCLE']
for c in sorted(circles, key=lambda c: -c.dxf.radius):
    if c.dxf.radius > 2:
        cx = c.dxf.center
        print(f"  center=({cx[0]:.1f}, {cx[1]:.1f}), r={c.dxf.radius:.1f}")
