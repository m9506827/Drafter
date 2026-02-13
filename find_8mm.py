"""深入分析 F09 腳架的圓柱面分佈，找出 8mm 腳架座"""
import sys, os
sys.path.insert(0, '.')
os.environ['DISPLAY'] = ''

from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp as brepgprop

from auto_drafter_system import MockCADEngine

stp_path = 'test/2-2.stp'
engine = MockCADEngine(model_file=stp_path)

# 從 _solid_shapes 取得 shape
target_fid = 'F09'
target_shape = engine._solid_shapes.get(target_fid)

if target_shape is None:
    print(f"Shape not found for {target_fid} in _solid_shapes")
    print(f"Available keys: {list(engine._solid_shapes.keys())}")
    sys.exit(1)

print(f"Found {target_fid} shape in _solid_shapes")

# 主方向 = Z 軸 (腳架垂直)
dominant_radius = 20.65  # pipe_d/2 = 41.3/2

# 遍歷所有面
face_data = []
explorer = TopExp_Explorer(target_shape, TopAbs_FACE)
face_idx = 0
while explorer.More():
    face_shape = explorer.Current()
    face = TopoDS.Face_s(face_shape)
    adaptor = BRepAdaptor_Surface(face)
    surf_type = adaptor.GetType()
    
    bb = Bnd_Box()
    BRepBndLib.Add_s(face, bb)
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    
    props = GProp_GProps()
    brepgprop.SurfaceProperties_s(face, props)
    area = props.Mass()
    
    type_name = "other"
    radius = 0
    if surf_type == GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        radius = cyl.Radius()
        type_name = "cylinder"
    elif surf_type == GeomAbs_Plane:
        type_name = "plane"
    
    face_data.append({
        'idx': face_idx,
        'type': type_name,
        'radius': radius,
        'area': area,
        'z_min': zmin,
        'z_max': zmax,
        'z_span': zmax - zmin,
        'x_range': (round(xmin, 1), round(xmax, 1)),
        'y_range': (round(ymin, 1), round(ymax, 1)),
    })
    
    face_idx += 1
    explorer.Next()

print(f"\n總共 {len(face_data)} 個面")

# 主要圓柱面
print(f"\n=== 主要圓柱面 (r ~{dominant_radius:.1f}) ===")
main_cyls = [f for f in face_data if f['type'] == 'cylinder' and abs(f['radius'] - dominant_radius) < 2]
for f in sorted(main_cyls, key=lambda f: f['z_min']):
    print(f"  face[{f['idx']:2d}]: r={f['radius']:.2f}, z=[{f['z_min']:.1f} ~ {f['z_max']:.1f}], span={f['z_span']:.1f}, area={f['area']:.0f}")

if main_cyls:
    all_z_min = min(f['z_min'] for f in main_cyls)
    all_z_max = max(f['z_max'] for f in main_cyls)
    print(f"\n主要圓柱面 Z 範圍: [{all_z_min:.1f} ~ {all_z_max:.1f}], span = {all_z_max - all_z_min:.1f}")
    print(f"face_extent = {all_z_max - all_z_min:.1f}")
    print(f"line_length = {all_z_max - all_z_min:.1f} - {dominant_radius*2:.1f} = {all_z_max - all_z_min - dominant_radius*2:.1f}")

# 其他圓柱面
print(f"\n=== 其他半徑圓柱面 ===")
other_cyls = [f for f in face_data if f['type'] == 'cylinder' and abs(f['radius'] - dominant_radius) >= 2]
for f in sorted(other_cyls, key=lambda f: f['z_min']):
    print(f"  face[{f['idx']:2d}]: r={f['radius']:.2f}, z=[{f['z_min']:.1f} ~ {f['z_max']:.1f}], span={f['z_span']:.1f}, area={f['area']:.0f}")

# 平面
print(f"\n=== 平面 ===")
planes = sorted([f for f in face_data if f['type'] == 'plane'], key=lambda f: f['z_min'])
for f in planes:
    print(f"  face[{f['idx']:2d}]: z=[{f['z_min']:.1f} ~ {f['z_max']:.1f}], span={f['z_span']:.1f}, area={f['area']:.0f}, x={f['x_range']}, y={f['y_range']}")

# 其他面
print(f"\n=== 其他類型面 ===")
others = [f for f in face_data if f['type'] == 'other']
for f in sorted(others, key=lambda f: f['z_min']):
    print(f"  face[{f['idx']:2d}]: z=[{f['z_min']:.1f} ~ {f['z_max']:.1f}], span={f['z_span']:.1f}, area={f['area']:.0f}")

# Z 結構分析
print(f"\n=== Z 方向結構分析 ===")
all_z_vals = set()
for f in face_data:
    all_z_vals.add(round(f['z_min'], 1))
    all_z_vals.add(round(f['z_max'], 1))
z_sorted = sorted(all_z_vals)
print(f"所有 Z 節點值 ({len(z_sorted)} 個): {z_sorted}")

z_bottom = min(f['z_min'] for f in face_data)
z_top = max(f['z_max'] for f in face_data)
print(f"\n整體 Z 範圍: [{z_bottom:.1f} ~ {z_top:.1f}], 跨距={z_top - z_bottom:.1f}")

# 底部區域面
print(f"\n底部區域面（z < {z_bottom + 20:.1f}）:")
bottom_faces = [f for f in face_data if f['z_min'] < z_bottom + 20]
for f in sorted(bottom_faces, key=lambda f: f['z_min']):
    print(f"  face[{f['idx']:2d}]: type={f['type']}, r={f['radius']:.2f}, z=[{f['z_min']:.1f} ~ {f['z_max']:.1f}], span={f['z_span']:.1f}, area={f['area']:.0f}")

# 頂部區域面
print(f"\n頂部區域面（z > {z_top - 20:.1f}）:")
top_faces = [f for f in face_data if f['z_max'] > z_top - 20]
for f in sorted(top_faces, key=lambda f: f['z_max'], reverse=True):
    print(f"  face[{f['idx']:2d}]: type={f['type']}, r={f['radius']:.2f}, z=[{f['z_min']:.1f} ~ {f['z_max']:.1f}], span={f['z_span']:.1f}, area={f['area']:.0f}")
