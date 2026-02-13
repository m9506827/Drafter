"""
preview_dxf.py
Reads DXF files from output/ and renders them to PNG previews using ezdxf + matplotlib.
"""

import os
import sys
import math

import ezdxf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle, Arc as MplArc
import matplotlib.font_manager as fm


def find_cjk_font():
    """Try to find a CJK font available on the system."""
    for font_name in ['Microsoft JhengHei', 'Microsoft YaHei',
                       'SimHei', 'Noto Sans CJK TC']:
        result = fm.findfont(fm.FontProperties(family=font_name),
                             fallback_to_default=False)
        if result and 'fallback' not in str(result).lower():
            return font_name
    return None


def render_dxf_to_png(dxf_path: str, png_path: str):
    """
    Render a DXF file to a PNG image.
    - White background, black lines
    - Figure size (16, 11) for A3 ratio
    - Handles: LINE, LWPOLYLINE, CIRCLE, ARC, TEXT, MTEXT, POLYLINE
    """
    print(f"Reading DXF: {dxf_path}")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    cjk_font = find_cjk_font()
    if cjk_font:
        plt.rcParams['font.sans-serif'] = [cjk_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(16, 11), dpi=150)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    default_color = 'black'

    entity_count = 0

    for entity in msp:
        etype = entity.dxftype()
        lw = 0.5

        if etype == 'LINE':
            ax.plot(
                [entity.dxf.start.x, entity.dxf.end.x],
                [entity.dxf.start.y, entity.dxf.end.y],
                color=default_color, linewidth=lw)
            entity_count += 1

        elif etype == 'LWPOLYLINE':
            pts = list(entity.get_points(format='xyb'))
            if len(pts) < 2:
                continue
            # Handle bulge (arc segments between vertices)
            is_closed = entity.closed
            n = len(pts)
            for i in range(n - (0 if is_closed else 1)):
                x1, y1, bulge = pts[i]
                x2, y2, _ = pts[(i + 1) % n]
                if abs(bulge) < 1e-6:
                    # Straight segment
                    ax.plot([x1, x2], [y1, y2],
                            color=default_color, linewidth=lw)
                else:
                    # Arc segment defined by bulge
                    _draw_bulge_arc(ax, x1, y1, x2, y2, bulge,
                                    color=default_color, linewidth=lw)
            entity_count += 1

        elif etype == 'POLYLINE':
            pts = [(v.dxf.location.x, v.dxf.location.y)
                   for v in entity.vertices]
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                if entity.is_closed:
                    xs.append(xs[0])
                    ys.append(ys[0])
                ax.plot(xs, ys, color=default_color, linewidth=lw)
            entity_count += 1

        elif etype == 'CIRCLE':
            cx = entity.dxf.center.x
            cy = entity.dxf.center.y
            r = entity.dxf.radius
            circle_patch = MplCircle(
                (cx, cy), r, fill=False,
                edgecolor=default_color, linewidth=lw)
            ax.add_patch(circle_patch)
            entity_count += 1

        elif etype == 'ARC':
            cx = entity.dxf.center.x
            cy = entity.dxf.center.y
            r = entity.dxf.radius
            start = entity.dxf.start_angle
            end = entity.dxf.end_angle
            arc_patch = MplArc(
                (cx, cy), r * 2, r * 2, angle=0,
                theta1=start, theta2=end,
                edgecolor=default_color, linewidth=lw, fill=False)
            ax.add_patch(arc_patch)
            entity_count += 1

        elif etype == 'TEXT':
            insert = entity.dxf.insert
            text_val = entity.dxf.text
            h = entity.dxf.height
            font_size = max(3, min(h * 1.2, 8))
            font_props = {}
            if cjk_font:
                font_props['fontfamily'] = cjk_font
            else:
                font_props['fontfamily'] = 'monospace'
            ax.text(insert.x, insert.y, text_val,
                    fontsize=font_size, color=default_color,
                    verticalalignment='bottom', **font_props)
            entity_count += 1

        elif etype == 'MTEXT':
            insert = entity.dxf.insert
            text_val = entity.plain_text()
            h = entity.dxf.char_height
            font_size = max(3, min(h * 1.2, 8))
            font_props = {}
            if cjk_font:
                font_props['fontfamily'] = cjk_font
            else:
                font_props['fontfamily'] = 'monospace'
            ax.text(insert.x, insert.y, text_val,
                    fontsize=font_size, color=default_color,
                    verticalalignment='top', **font_props)
            entity_count += 1

    ax.autoscale()
    ax.margins(0.02)
    ax.tick_params(colors='#888888', labelsize=6)

    # Add a thin border around the axes
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.5)

    basename = os.path.splitext(os.path.basename(dxf_path))[0]
    ax.set_title(basename, fontsize=10, color='black', pad=8)

    fig.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    print(f"  -> Saved PNG: {png_path}  ({entity_count} entities rendered)")


def _draw_bulge_arc(ax, x1, y1, x2, y2, bulge, color='black', linewidth=0.5):
    """
    Draw an arc segment defined by two endpoints and a bulge value.
    Bulge = tan(included_angle / 4). Positive = CCW, negative = CW.
    """
    dx = x2 - x1
    dy = y2 - y1
    chord = math.hypot(dx, dy)
    if chord < 1e-12:
        return

    # Sagitta and radius
    sagitta = abs(bulge) * chord / 2.0
    radius = ((chord / 2.0) ** 2 + sagitta ** 2) / (2.0 * sagitta)

    # Midpoint of chord
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0

    # Unit normal to chord (pointing left of direction P1->P2)
    nx = -dy / chord
    ny = dx / chord

    # Distance from midpoint to center
    d = radius - sagitta

    # Center: offset from midpoint along normal
    if bulge > 0:
        cx = mx + d * nx
        cy = my + d * ny
    else:
        cx = mx - d * nx
        cy = my - d * ny

    # Calculate start and end angles
    start_angle = math.degrees(math.atan2(y1 - cy, x1 - cx))
    end_angle = math.degrees(math.atan2(y2 - cy, x2 - cx))

    # For positive bulge (CCW), start_angle < end_angle
    # For negative bulge (CW), start_angle > end_angle
    if bulge > 0:
        if end_angle < start_angle:
            end_angle += 360.0
    else:
        if start_angle < end_angle:
            start_angle += 360.0
        start_angle, end_angle = end_angle, start_angle

    arc_patch = MplArc(
        (cx, cy), radius * 2, radius * 2, angle=0,
        theta1=start_angle, theta2=end_angle,
        edgecolor=color, linewidth=linewidth, fill=False)
    ax.add_patch(arc_patch)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")

    files = [
        ("2-2-0.dxf", "2-2-0-preview.png"),
        ("2-2_2.dxf", "2-2-2-preview.png"),
    ]

    for dxf_name, png_name in files:
        dxf_path = os.path.join(output_dir, dxf_name)
        png_path = os.path.join(output_dir, png_name)

        if not os.path.isfile(dxf_path):
            print(f"WARNING: DXF file not found: {dxf_path}")
            continue

        render_dxf_to_png(dxf_path, png_path)

    print("Done.")


if __name__ == "__main__":
    main()
