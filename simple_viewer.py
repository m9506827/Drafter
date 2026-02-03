# 檔名: simple_viewer.py (優化版)
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import ezdxf
from ezdxf import bbox
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import pyvista as pv

# 強制使用白色背景主題
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = 'white'

# DXF color index → matplotlib color (white bg: 0 and 7 map to black)
_DXF_COLOR_MAP = {
    0: 'black', 1: 'red', 2: 'yellow', 3: 'green',
    4: 'cyan', 5: 'blue', 6: 'magenta', 7: 'black',
}


def _dxf_color(entity):
    """Return matplotlib color string from DXF entity color index."""
    c = entity.dxf.get('color', 7)
    return _DXF_COLOR_MAP.get(c, 'black')


def _find_cjk_font():
    """Find a CJK-capable font available on the system."""
    candidates = ['Microsoft JhengHei', 'SimHei', 'Noto Sans CJK TC',
                  'Noto Sans CJK SC', 'MS Gothic', 'Arial Unicode MS']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False

class EngineeringViewer:
    """工程圖檢視器：負責 3D (STEP/STL) 與 2D (DXF) 的視覺化"""

    @staticmethod
    def get_rotation_matrix(axis, angle_deg):
        """
        建立繞指定軸的旋轉矩陣 (3x3)
        axis: 'x', 'y', 'z'
        angle_deg: 旋轉角度 (度)
        """
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        if axis == 'x':
            return np.array([[1, 0, 0],
                             [0, c, -s],
                             [0, s,  c]])
        elif axis == 'y':
            return np.array([[ c, 0, s],
                             [ 0, 1, 0],
                             [-s, 0, c]])
        elif axis == 'z':
            return np.array([[c, -s, 0],
                             [s,  c, 0],
                             [0,  0, 1]])
        else:
            raise ValueError(f"未知的旋轉軸: {axis}")

    @staticmethod
    def project_3d_to_2d(points, view_type='front'):
        """
        將 3D 點雲投影至 2D 施工圖平面

        Parameters:
            points: numpy array, shape (N, 3) - 3D 點座標
            view_type:
                直接投影: 'top' | 'front' | 'right'
                旋轉投影: 'top_rot' | 'front_rot' | 'right_rot'
                等角視圖: 'iso'

        Returns:
            numpy array, shape (N, 2) - 投影後的 2D 座標
        """
        points = np.asarray(points, dtype=float)
        get_rot = EngineeringViewer.get_rotation_matrix

        # ===== Mode 1: 直接投影（不旋轉） =====
        if view_type == 'top':
            # 俯視圖：直接取 (x, y)，忽略 z
            return points[:, [0, 1]]

        elif view_type == 'front':
            # 前視圖：直接取 (x, z)，忽略 y
            return points[:, [0, 2]]

        elif view_type == 'right':
            # 右視圖：直接取 (y, z)，忽略 x
            return points[:, [1, 2]]

        # ===== Mode 2: 反向視角投影 =====
        elif view_type == 'top_rot':
            # 俯視圖（Y 翻轉）: (x, -y)
            return np.column_stack([points[:, 0], -points[:, 1]])

        elif view_type == 'front_rot':
            # 前視圖: (x, z)
            return points[:, [0, 2]]

        elif view_type == 'right_rot':
            # 側視圖（沿 Y 軸看，X 鏡射）: (-x, z)
            return np.column_stack([-points[:, 0], points[:, 2]])

        elif view_type == 'iso':
            # 等角視圖 (Isometric)：Z 軸轉 45°，再繞 X 軸轉 35.264°
            rot_z = get_rot('z', 45)
            rot_x = get_rot('x', 35.264)
            projected = points @ rot_z.T @ rot_x.T
            return projected[:, [0, 1]]

        else:
            raise ValueError(f"未知的視圖類型: {view_type}，支援: top, front, right, top_rot, front_rot, right_rot, iso")

    @staticmethod
    def view_projected_2d(filename, view_type='front', save_path=None):
        """
        從 3D 模型 (STEP/STL) 提取點雲並投影為 2D 施工圖

        Parameters:
            filename: 3D 模型檔案路徑
            view_type: 'top' | 'front' | 'right' | 'iso' | 'all'
        """
        if not os.path.exists(filename):
            print(f"[Viewer] 找不到檔案: {filename}")
            return

        print(f"[Viewer] 啟動 3D→2D 投影: {filename} (視圖: {view_type})")

        try:
            # 讀取 3D 模型
            mesh = None
            if filename.lower().endswith(('.step', '.stp')):
                if not CADQUERY_AVAILABLE:
                    print("[Error] 需要安裝 CadQuery 才能讀取 STEP 檔")
                    return
                step_model = cq.importers.importStep(filename)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                    cq.exporters.export(step_model, tmp.name)
                    mesh = pv.read(tmp.name)
                try:
                    os.unlink(tmp.name)
                except:
                    pass
            else:
                mesh = pv.read(filename)

            # 提取頂點及邊線
            points_3d = np.array(mesh.points)
            edges = mesh.extract_all_edges()

            if view_type == 'all':
                # 四合一視圖
                views = ['top', 'front', 'right', 'iso']
                titles = ['Top View (俯視圖)', 'Front View (前視圖)',
                          'Right View (右視圖)', 'Isometric (等角視圖)']
                fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
                fig.suptitle(f"2D 投影: {os.path.basename(filename)}", fontsize=14)

                for idx, (vt, title) in enumerate(zip(views, titles)):
                    ax = axes[idx // 2][idx % 2]
                    ax.set_facecolor('white')
                    projected = EngineeringViewer.project_3d_to_2d(points_3d, vt)

                    # 繪製邊線
                    edge_lines = edges.lines.reshape(-1, 3)
                    for line in edge_lines:
                        i0, i1 = line[1], line[2]
                        ax.plot([projected[i0, 0], projected[i1, 0]],
                                [projected[i0, 1], projected[i1, 1]],
                                'k-', linewidth=0.5)

                    ax.set_aspect('equal')
                    ax.grid(True, color='#cccccc', linestyle='--', linewidth=0.5)
                    ax.set_title(title)

            else:
                # 單一視圖
                fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
                ax.set_facecolor('white')
                projected = EngineeringViewer.project_3d_to_2d(points_3d, view_type)

                edge_lines = edges.lines.reshape(-1, 3)
                for line in edge_lines:
                    i0, i1 = line[1], line[2]
                    ax.plot([projected[i0, 0], projected[i1, 0]],
                            [projected[i0, 1], projected[i1, 1]],
                            'k-', linewidth=0.5)

                ax.set_aspect('equal')
                ax.grid(True, color='#cccccc', linestyle='--', linewidth=0.5)
                view_names = {'top': '俯視圖', 'front': '前視圖',
                              'right': '右視圖', 'iso': '等角視圖'}
                ax.set_title(f"{view_names.get(view_type, view_type)}: "
                             f"{os.path.basename(filename)}")

            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                print(f"[Viewer] Saved: {save_path}")
            else:
                print("[Viewer] 視窗開啟中...")
                plt.show(block=True)

        except Exception as e:
            print(f"[Viewer] 投影顯示錯誤: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def view_3d_stl(filename):
        """3D 模型預覽 - 四視圖"""
        if not os.path.exists(filename): 
            return
        print(f"[Viewer] 啟動工程 3D 預覽 (四視圖): {filename} ...")
        
        try:
            mesh = None
            if filename.lower().endswith(('.step', '.stp')):
                if not CADQUERY_AVAILABLE:
                    print("[Error] 需要安裝 CadQuery 才能預覽 STEP 檔")
                    return
                step_model = cq.importers.importStep(filename)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                    cq.exporters.export(step_model, tmp.name)
                    mesh = pv.read(tmp.name)
                try: 
                    os.unlink(tmp.name)
                except: 
                    pass
            else:
                mesh = pv.read(filename)

            p = pv.Plotter(shape=(2, 2), border=True)

            def add_model_to_subplot(plotter, row, col, title, view_func):
                plotter.subplot(row, col)
                plotter.add_mesh(mesh, color='#eeeeee', show_edges=True, edge_color='#333333')
                plotter.add_text(title, font_size=10)
                plotter.show_grid()
                plotter.add_axes()
                if view_func:
                    view_func()

            add_model_to_subplot(p, 0, 0, "1. Isometric (Free View)", p.view_isometric)
            add_model_to_subplot(p, 0, 1, "2. Top View (XY Plane)", p.view_xy)
            add_model_to_subplot(p, 1, 0, "3. Front View (XZ Plane)", p.view_xz)
            add_model_to_subplot(p, 1, 1, "4. Right View (YZ Plane)", p.view_yz)

            print("[Viewer] 視窗開啟中...")
            p.show()
            
        except Exception as e:
            print(f"[Viewer] 3D 顯示錯誤: {e}")

    @staticmethod
    def view_2d_dxf(filename, fast_mode=True, save_path=None):
        """
        [核心修正版] 顯示 2D DXF 工程圖
        fast_mode: True = 快速模式（簡化渲染），False = 完整模式
        """
        if not os.path.exists(filename):
            print(f"[Viewer] 找不到檔案: {filename}")
            return

        print(f"[Viewer] 啟動 2D 檢視: {filename} ...")
        
        try:
            # 1. 讀取 DXF
            doc = ezdxf.readfile(filename)
            msp = doc.modelspace()

            # 2. 計算邊界
            cache = bbox.Cache()
            bb = bbox.extents(msp, cache=cache)
            
            xmin, ymin, xmax, ymax = -50, -50, 50, 50
            
            if bb.has_data:
                xmin = bb.extmin[0]
                ymin = bb.extmin[1]
                xmax = bb.extmax[0]
                ymax = bb.extmax[1]
                
                w, h = xmax - xmin, ymax - ymin
                margin = max(w, h) * 0.1
                if margin == 0: 
                    margin = 10
                
                xmin -= margin
                xmax += margin
                ymin -= margin
                ymax += margin
                print(f"[Viewer] 偵測範圍: X[{xmin:.1f}, {xmax:.1f}], Y[{ymin:.1f}, {ymax:.1f}]")
            else:
                print("[Viewer] 警告: 檔案似乎是空的")

            # 3. 快速模式：直接繪製線條，跳過 ezdxf 渲染引擎
            if fast_mode:
                print("[Viewer] 使用快速渲染模式...")
                fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
                ax.set_facecolor('white')

                # CJK font for Chinese text
                cjk_font = _find_cjk_font()
                font_props = {'family': cjk_font} if cjk_font else {}
                if cjk_font:
                    print(f"[Viewer] 使用 CJK 字型: {cjk_font}")

                # 直接讀取所有 LINE 實體
                line_count = 0
                for entity in msp.query('LINE'):
                    start = entity.dxf.start
                    end = entity.dxf.end
                    ax.plot([start[0], end[0]], [start[1], end[1]],
                           color=_dxf_color(entity), linewidth=0.5)
                    line_count += 1

                print(f"[Viewer] 已繪製 {line_count} 條線段")

                # LWPOLYLINE
                for entity in msp.query('LWPOLYLINE'):
                    points = list(entity.get_points('xy'))
                    if points:
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        ax.plot(xs, ys, color=_dxf_color(entity), linewidth=0.5)

                # CIRCLE
                for entity in msp.query('CIRCLE'):
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    circle = plt.Circle((center[0], center[1]), radius,
                                       fill=False, edgecolor=_dxf_color(entity),
                                       linewidth=0.5)
                    ax.add_patch(circle)

                # ARC
                for entity in msp.query('ARC'):
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    start_angle = entity.dxf.get('start_angle', 0)
                    end_angle = entity.dxf.get('end_angle', 360)
                    arc = mpatches.Arc((center[0], center[1]),
                                      2 * radius, 2 * radius,
                                      angle=0,
                                      theta1=start_angle, theta2=end_angle,
                                      edgecolor=_dxf_color(entity),
                                      linewidth=0.5)
                    ax.add_patch(arc)

                # TEXT
                for entity in msp.query('TEXT'):
                    insert = entity.dxf.insert
                    text = entity.dxf.text
                    height = entity.dxf.get('height', 10)
                    rotation = entity.dxf.get('rotation', 0)
                    ax.text(insert[0], insert[1], text,
                            fontsize=max(6, height * 0.8),
                            rotation=rotation,
                            color=_dxf_color(entity),
                            ha='left', va='bottom',
                            **font_props)

                # MTEXT
                for entity in msp.query('MTEXT'):
                    insert = entity.dxf.insert
                    raw = entity.text  # plain text content
                    height = entity.dxf.get('char_height', 10)
                    rotation = entity.dxf.get('rotation', 0)
                    ax.text(insert[0], insert[1], raw,
                            fontsize=max(6, height * 0.8),
                            rotation=rotation,
                            color=_dxf_color(entity),
                            ha='left', va='bottom',
                            **font_props)

                # 設定視圖
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect('equal')
                ax.grid(True, color='#cccccc', linestyle='--', linewidth=0.5)
                ax.set_title(f"DXF: {os.path.basename(filename)}", color='black')
                
            else:
                # 4. 完整模式：使用 ezdxf 渲染引擎（較慢但支援所有實體）
                print("[Viewer] 使用完整渲染模式...")
                fig = plt.figure(figsize=(10, 8), facecolor='white')
                ax = fig.add_subplot(111, facecolor='white')
                
                ctx = RenderContext(doc)
                out = MatplotlibBackend(ax)
                Frontend(ctx, out).draw_layout(msp, finalize=True)
                
                # 強制線條為黑色
                for artist in ax.get_children():
                    if hasattr(artist, 'get_color') and hasattr(artist, 'set_color'):
                        c = artist.get_color()
                        if c in ['white', '#ffffff', 'none'] or \
                           (isinstance(c, tuple) and len(c)>=3 and sum(c[:3]) > 2.5):
                            artist.set_color('black')
                    
                    if hasattr(artist, 'set_linewidth'):
                        artist.set_linewidth(0.5)
                
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect('equal')
                ax.grid(True, color='#cccccc', linestyle='--', linewidth=0.5)
                ax.set_title(f"DXF: {os.path.basename(filename)}", color='black')
            
            # 5. 顯示或儲存
            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                print(f"[Viewer] Saved: {save_path}")
            else:
                print("[Viewer] 視窗開啟中...")
                plt.show(block=True)

        except Exception as e:
            print(f"[Viewer] 2D 顯示錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]

        # 檢查是否為 3D→2D 投影模式
        if '--project' in sys.argv:
            # 取得視圖類型，預設 'all'
            view = 'all'
            for arg in sys.argv[2:]:
                if arg in ('top', 'front', 'right', 'iso', 'all'):
                    view = arg
                    break
            EngineeringViewer.view_projected_2d(filename, view_type=view)
        elif filename.lower().endswith(('.dxf', '.dwg')):
            fast = '--full' not in sys.argv
            EngineeringViewer.view_2d_dxf(filename, fast_mode=fast)
        else:
            EngineeringViewer.view_3d_stl(filename)
    else:
        print("使用方式:")
        print("  python simple_viewer.py model.dxf                  # 快速模式")
        print("  python simple_viewer.py model.dxf --full           # 完整模式")
        print("  python simple_viewer.py model.stp                  # 3D 預覽")
        print("  python simple_viewer.py model.stp --project        # 3D→2D 投影 (四視圖)")
        print("  python simple_viewer.py model.stp --project front  # 3D→2D 投影 (指定視圖)")
        print("  可用視圖: top, front, right, iso, all")