"""Generate PNG preview of Drawing 3 DXF."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['DRAFTER_NO_GUI'] = '1'

import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dxf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_d3_test', '2-2_3.dxf')
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2-2-3-seg-only.png')

doc = ezdxf.readfile(dxf_path)
msp = doc.modelspace()

fig = plt.figure(figsize=(16, 12))
ax = fig.add_axes([0, 0, 1, 1])
ctx = RenderContext(doc)
out = MatplotlibBackend(ax)
Frontend(ctx, out).draw_layout(msp)
ax.set_aspect('equal')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Preview saved: {out_png}")
