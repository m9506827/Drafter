import sys
import os
import subprocess
import time
import platform
import matplotlib.pyplot as plt
import matplotlib
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import pyvista as pv

# 延遲導入 cadquery，用於讀取 STEP 檔案
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False

# 延遲導入 odafc，用於讀取 DWG 檔案
try:
    from ezdxf.addons import odafc
    ODAFC_AVAILABLE = True
except ImportError:
    ODAFC_AVAILABLE = False

def find_oda_file_converter():
    """
    搜索 ODA File Converter 的安裝位置
    返回可執行檔案的完整路徑，如果找不到則返回 None
    
    檢查順序：
    1. 環境變數 ODA_FILE_CONVERTER_PATH
    2. 標準安裝位置
    3. 系統搜索
    """
    if not ODAFC_AVAILABLE:
        return None
    
    # 1. 檢查環境變數（允許用戶手動指定）
    env_path = os.environ.get('ODA_FILE_CONVERTER_PATH')
    if env_path and os.path.exists(env_path):
        print(f"[Viewer] 使用環境變數指定的 ODA File Converter: {env_path}")
        return env_path
    
    # 2. 標準安裝位置
    standard_paths = [
        r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"C:\Program Files (x86)\ODA\ODAFileConverter\ODAFileConverter.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\ODA\ODAFileConverter\ODAFileConverter.exe"),
        r"D:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"D:\Program Files (x86)\ODA\ODAFileConverter\ODAFileConverter.exe",
    ]
    
    # 檢查標準位置
    for path in standard_paths:
        if os.path.exists(path):
            return path
    
    # 3. 嘗試使用 odafc 的檢測方法
    try:
        expected_path = odafc.get_win_exec_path()
        if os.path.exists(expected_path):
            return expected_path
    except:
        pass
    
    # 4. 搜索可能的安裝目錄（限制深度以提高速度）
    search_dirs = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
        r"D:\Program Files",
        r"D:\Program Files (x86)",
        os.path.expanduser(r"~\AppData\Local\Programs"),
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            try:
                # 只搜索前兩層，避免太慢
                for root, dirs, files in os.walk(search_dir):
                    depth = root[len(search_dir):].count(os.sep)
                    if depth > 2:  # 限制深度
                        dirs[:] = []  # 不繼續深入
                        continue
                    if "ODAFileConverter.exe" in files:
                        found_path = os.path.join(root, "ODAFileConverter.exe")
                        return found_path
            except (PermissionError, OSError):
                continue
    
    return None

def prompt_oda_file_converter_path():
    """
    提示用戶手動輸入 ODA File Converter 的路徑
    返回路徑字串，如果取消則返回 None
    """
    try:
        from tkinter import filedialog, Tk, messagebox
        
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        messagebox.showinfo(
            "ODA File Converter 路徑",
            "請選擇 ODAFileConverter.exe 的位置\n\n"
            "如果找不到，可以：\n"
            "1. 下載安裝：https://www.opendesign.com/guestfiles/oda_file_converter\n"
            "2. 或將 DWG 轉換為 DXF 格式"
        )
        
        file_path = filedialog.askopenfilename(
            title="選擇 ODAFileConverter.exe",
            filetypes=[
                ("執行檔", "*.exe"),
                ("所有檔案", "*.*")
            ],
            initialdir=r"C:\Program Files"
        )
        
        root.destroy()
        
        if file_path and os.path.exists(file_path):
            # 設置環境變數（僅本次會話有效）
            os.environ['ODA_FILE_CONVERTER_PATH'] = file_path
            print(f"[Viewer] 已設置 ODA File Converter 路徑: {file_path}")
            return file_path
        
        return None
    except Exception as e:
        print(f"[Error] 無法顯示檔案選擇對話框: {e}")
        return None

# ==========================================
# 繪圖師的瑞士信刀：簡易檔案檢視器
# ==========================================

class EngineeringViewer:
    """
    提供快速查看 3D (STL, STEP/STP) 與 2D (DXF/DWG) 檔案的功能
    不需要打開大型 CAD 軟體即可驗證程式產出
    
    支援的 3D 格式：
    - STL: 網格格式，直接使用 PyVista 讀取
    - STEP/STP: 參數化 CAD 格式，使用 CadQuery 讀取並轉換為網格
    
    支援的 2D 格式：
    - DXF: 繪圖交換格式，使用 ezdxf 讀取
    - DWG: AutoCAD 原生格式，使用 ezdxf 讀取
    """
    
    @staticmethod
    def view_3d_stl(filename):
        """
        使用 PyVista 查看 3D 檔案 (.stl, .step, .stp)
        支援 STL 網格檔案和 STEP 參數化 CAD 檔案
        """
        if not os.path.exists(filename):
            print(f"Error: 找不到檔案 {filename}")
            return

        file_ext = os.path.splitext(filename)[1].lower()
        
        # 檢查是否為 2D 檔案格式（不應該用 3D 檢視器開啟）
        if file_ext in ['.dxf', '.dwg']:
            print(f"[Error] {file_ext.upper()} 是 2D 檔案格式，請使用 view_2d_dxf() 方法")
            print(f"[Info] 自動切換到 2D 檢視器...")
            EngineeringViewer.view_2d_dxf(filename)
            return
        
        print(f"Opening 3D Viewer for: {filename} ({file_ext}) ...")
        
        try:
            mesh = None
            
            # 處理 STEP/STP 檔案
            if file_ext in ['.step', '.stp']:
                if not CADQUERY_AVAILABLE:
                    print(f"Error: 需要 CadQuery 來讀取 STEP 檔案，但 CadQuery 不可用")
                    print(f"請安裝: pip install cadquery")
                    return
                
                print(f"Reading STEP file with CadQuery...")
                # 使用 cadquery 讀取 STEP 檔案
                step_model = cq.importers.importStep(filename)
                
                # 將 STEP 模型轉換為網格（使用 tessellation）
                # 導出為臨時 STL 檔案，然後用 PyVista 讀取
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
                    tmp_stl = tmp_file.name
                
                try:
                    # 將 STEP 模型導出為 STL
                    cq.exporters.export(step_model, tmp_stl)
                    # 用 PyVista 讀取 STL
                    mesh = pv.read(tmp_stl)
                    # 清理臨時檔案
                    os.unlink(tmp_stl)
                except Exception as e:
                    # 如果導出失敗，嘗試清理臨時檔案
                    if os.path.exists(tmp_stl):
                        os.unlink(tmp_stl)
                    raise e
            
            # 處理 STL 和其他網格格式
            else:
                # 直接使用 PyVista 讀取網格檔案
                mesh = pv.read(filename)
            
            if mesh is None:
                print(f"Error: 無法讀取檔案 {filename}")
                return
            
            # 設定繪圖器
            plotter = pv.Plotter()
            # 加入物體 (顏色設為金屬灰，開啟邊線顯示以利檢查結構)
            plotter.add_mesh(mesh, color='lightgrey', show_edges=True, edge_color='black')
            
            plotter.add_axes()  # 顯示座標軸
            plotter.add_text(f"PREVIEW: {os.path.basename(filename)}", position='upper_left')
            plotter.show()
            
        except Exception as e:
            print(f"3D View Error: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def view_3d_and_2d(stp_file: str, dxf_file: str):
        """
        同時顯示 3D (STEP/STP) 和 2D (DXF/DWG) 視圖
        用於對比 3D 模型和 2D 工程圖
        支援的 2D 格式：.dxf, .dwg
        """
        if not os.path.exists(stp_file):
            print(f"Error: 找不到 3D 檔案 {stp_file}")
            return
        if not os.path.exists(dxf_file):
            print(f"Error: 找不到 2D 檔案 {dxf_file}")
            return
        
        file_ext = os.path.splitext(dxf_file)[1].lower()
        if file_ext not in ['.dxf', '.dwg']:
            print(f"Error: 不支援的 2D 檔案格式 {file_ext}，請使用 .dxf 或 .dwg")
            return
        
        print(f"Opening combined 3D/2D viewer...")
        print(f"  3D: {stp_file}")
        print(f"  2D: {dxf_file}")
        
        # 檢查檔案類型，避免在非主線程中使用 Matplotlib
        stp_ext = os.path.splitext(stp_file)[1].lower()
        dxf_ext = os.path.splitext(dxf_file)[1].lower()
        
        # 如果兩個檔案都是 2D 格式，直接在主線程中顯示
        if stp_ext in ['.dxf', '.dwg'] and dxf_ext in ['.dxf', '.dwg']:
            print(f"[Viewer] 兩個檔案都是 2D 格式，依次顯示...")
            EngineeringViewer.view_2d_dxf(stp_file)
            # 等待第一個視窗關閉或使用非阻塞模式
            import time
            time.sleep(0.5)  # 短暫延遲
            EngineeringViewer.view_2d_dxf(dxf_file)
            return
        
        # 如果第一個檔案是 2D，第二個是 3D，交換順序
        if stp_ext in ['.dxf', '.dwg'] and dxf_ext in ['.stl', '.step', '.stp']:
            stp_file, dxf_file = dxf_file, stp_file
            stp_ext, dxf_ext = dxf_ext, stp_ext
        
        # 使用多線程同時顯示 3D 和 2D 視圖
        # 注意：Matplotlib 必須在主線程中使用
        import threading
        
        def show_3d():
            try:
                # 3D 視圖（PyVista）可以在背景線程中運行
                EngineeringViewer.view_3d_stl(stp_file)
            except Exception as e:
                print(f"3D viewer error: {e}")
        
        def show_2d():
            try:
                # 2D 視圖（Matplotlib）必須在主線程中運行
                EngineeringViewer.view_2d_dxf(dxf_file)
            except Exception as e:
                print(f"2D viewer error: {e}")
        
        # 如果第一個檔案是 3D，第二個是 2D
        if stp_ext in ['.stl', '.step', '.stp'] and dxf_ext in ['.dxf', '.dwg']:
            # 在背景線程中顯示 3D 視圖（PyVista 支援多線程）
            thread_3d = threading.Thread(target=show_3d, daemon=True)
            thread_3d.start()
            
            # 在主線程中顯示 2D 視圖（Matplotlib 必須在主線程）
            show_2d()
            
            # 等待 3D 視圖線程結束
            thread_3d.join(timeout=1.0)
        else:
            # 其他情況，都在主線程中依次顯示
            show_3d()
            import time
            time.sleep(0.5)
            show_2d()
    
    @staticmethod
    def view_2d_dxf(filename):
        """
        使用 Matplotlib 渲染 DXF/DWG 檔案
        這能讓你像看圖表一樣看工程圖
        支援格式：.dxf (DXF), .dwg (AutoCAD DWG)
        """
        if not os.path.exists(filename):
            print(f"Error: 找不到檔案 {filename}")
            return

        file_ext = os.path.splitext(filename)[1].lower()
        print(f"Opening 2D Viewer for: {filename} ({file_ext}) ...")
        
        try:
            # 讀取 DXF 或 DWG
            if file_ext == '.dwg':
                print(f"[Viewer] 讀取 DWG 檔案（AutoCAD 格式）...")
                # DWG 需要使用 odafc 來讀取
                if not ODAFC_AVAILABLE:
                    print(f"[Error] 無法讀取 DWG 檔案：需要 ezdxf[odafc] 支援")
                    print(f"[Error] 請安裝: pip install 'ezdxf[odafc]'")
                    print(f"[Error] 或使用 ODA File Converter 工具")
                    return
                
                # 嘗試使用 odafc 讀取 DWG 檔案
                # 即使 is_installed() 返回 False，也可能安裝在非標準位置
                # 直接嘗試讀取，如果失敗會拋出異常
                try:
                    print(f"[Viewer] 使用 ODA File Converter 讀取 DWG...")
                    
                    # 先搜索 ODA File Converter 的位置
                    oda_path = find_oda_file_converter()
                    original_path = os.environ.get('PATH', '')
                    
                    if oda_path:
                        print(f"[Viewer] 找到 ODA File Converter: {oda_path}")
                        # 將 ODAFileConverter 的目錄添加到 PATH，讓 odafc 能找到它
                        oda_dir = os.path.dirname(oda_path)
                        if oda_dir not in original_path:
                            os.environ['PATH'] = oda_dir + os.pathsep + original_path
                            print(f"[Viewer] 已將 ODA File Converter 目錄添加到 PATH: {oda_dir}")
                    else:
                        print(f"[Viewer] 未在標準位置找到 ODA File Converter")
                        print(f"[Viewer] 預期位置: {odafc.get_win_exec_path()}")
                        print(f"[Viewer] 嘗試直接讀取（可能會自動找到）...")
                    
                    # 嘗試讀取 DWG 檔案
                    doc = odafc.readfile(filename)
                    print(f"[Viewer] DWG 檔案讀取成功")
                    
                    # 恢復原始 PATH（可選，因為是臨時修改）
                    # os.environ['PATH'] = original_path
                except odafc.ODAFCNotInstalledError as e:
                    print(f"[Error] ODA File Converter 未正確安裝或無法找到")
                    print(f"[Error] 詳細訊息: {e}")
                    print(f"[Info] 預期安裝位置: {odafc.get_win_exec_path()}")
                    
                    # 嘗試讓用戶手動選擇路徑
                    print(f"\n[提示] 如果您已安裝 ODA File Converter，可以手動指定路徑")
                    try:
                        user_input = input(f"是否要手動選擇 ODAFileConverter.exe 的位置？(y/n): ").strip().lower()
                        if user_input in ['y', 'yes', '是']:
                            oda_path = prompt_oda_file_converter_path()
                            if oda_path:
                                # 將 ODAFileConverter 的目錄添加到 PATH
                                oda_dir = os.path.dirname(oda_path)
                                original_path = os.environ.get('PATH', '')
                                if oda_dir not in original_path:
                                    os.environ['PATH'] = oda_dir + os.pathsep + original_path
                                    print(f"[Viewer] 已將 ODA File Converter 目錄添加到 PATH: {oda_dir}")
                                
                                # 再次嘗試讀取
                                print(f"[Viewer] 使用手動指定的路徑讀取 DWG...")
                                try:
                                    doc = odafc.readfile(filename)
                                    print(f"[Viewer] DWG 檔案讀取成功")
                                    # 繼續後續處理，不要 return
                                except Exception as e2:
                                    print(f"[Error] 使用指定路徑仍然失敗: {e2}")
                                    print(f"[Info] 建議：將 DWG 檔案轉換為 DXF 格式後再讀取")
                                    return
                            else:
                                print(f"[Info] 未選擇路徑，取消讀取")
                                return
                        else:
                            print(f"[Info] 請確認 ODA File Converter 已正確安裝:")
                            print(f"  1. 下載並安裝: https://www.opendesign.com/guestfiles/oda_file_converter")
                            print(f"  2. 確認安裝在標準位置: C:\\Program Files\\ODA\\ODAFileConverter\\")
                            print(f"  3. 或設置環境變數 ODA_FILE_CONVERTER_PATH 指向執行檔")
                            print(f"[Info] 或將 DWG 檔案轉換為 DXF 格式後再讀取")
                            return
                    except (EOFError, KeyboardInterrupt):
                        # 非互動式環境或用戶取消
                        print(f"\n[Info] 請確認 ODA File Converter 已正確安裝:")
                        print(f"  1. 下載並安裝: https://www.opendesign.com/guestfiles/oda_file_converter")
                        print(f"  2. 或設置環境變數: set ODA_FILE_CONVERTER_PATH=<路徑>")
                        print(f"[Info] 或將 DWG 檔案轉換為 DXF 格式後再讀取")
                        return
                except odafc.UnsupportedFileFormat as e:
                    print(f"[Error] 不支援的 DWG 檔案格式: {e}")
                    print(f"[Info] 建議：將 DWG 檔案轉換為 DXF 格式後再讀取")
                    return
                except Exception as e:
                    print(f"[Error] 讀取 DWG 檔案失敗: {e}")
                    print(f"[Error] 可能原因：")
                    print(f"  1. ODA File Converter 未正確安裝或配置")
                    print(f"  2. DWG 檔案格式不支援或損壞")
                    print(f"  3. DWG 檔案版本過新或過舊")
                    print(f"[Info] 建議：將 DWG 檔案轉換為 DXF 格式後再讀取")
                    import traceback
                    traceback.print_exc()
                    return
            else:
                print(f"[Viewer] 讀取 DXF 檔案...")
                doc = ezdxf.readfile(filename)
            msp = doc.modelspace()

            # 計算繪圖範圍（邊界框）
            try:
                # 遍歷所有實體來計算範圍
                x_coords = []
                y_coords = []
                
                for entity in msp:
                    try:
                        # 嘗試獲取實體的邊界框
                        if hasattr(entity, 'bbox'):
                            bbox = entity.bbox()
                            if bbox:
                                x_coords.extend([bbox.extmin.x, bbox.extmax.x])
                                y_coords.extend([bbox.extmin.y, bbox.extmax.y])
                        # 對於圓形，使用中心和半徑
                        elif entity.dxftype() == 'CIRCLE':
                            center = entity.dxf.center
                            radius = entity.dxf.radius
                            x_coords.extend([float(center.x) - float(radius), float(center.x) + float(radius)])
                            y_coords.extend([float(center.y) - float(radius), float(center.y) + float(radius)])
                        # 對於線條和多段線，獲取頂點
                        elif entity.dxftype() in ['LINE', 'LWPOLYLINE', 'POLYLINE']:
                            if entity.dxftype() == 'LWPOLYLINE':
                                # LWPOLYLINE 使用 vertices() 方法
                                try:
                                    for vertex in entity.vertices():
                                        # 頂點可能是元組 (x, y) 或 DXF 屬性
                                        if isinstance(vertex, tuple) or isinstance(vertex, list):
                                            x_coords.append(float(vertex[0]))
                                            y_coords.append(float(vertex[1]))
                                        elif hasattr(vertex, 'dxf'):
                                            x_coords.append(float(vertex.dxf.location.x))
                                            y_coords.append(float(vertex.dxf.location.y))
                                        else:
                                            # 嘗試直接轉換
                                            x_coords.append(float(vertex[0]))
                                            y_coords.append(float(vertex[1]))
                                except:
                                    # 如果 vertices() 失敗，嘗試其他方法
                                    pass
                            elif entity.dxftype() == 'LINE':
                                # LINE 有 start 和 end 點
                                if hasattr(entity, 'dxf'):
                                    if hasattr(entity.dxf, 'start'):
                                        x_coords.append(float(entity.dxf.start.x))
                                        y_coords.append(float(entity.dxf.start.y))
                                    if hasattr(entity.dxf, 'end'):
                                        x_coords.append(float(entity.dxf.end.x))
                                        y_coords.append(float(entity.dxf.end.y))
                            elif hasattr(entity, 'vertices'):
                                for vertex in entity.vertices():
                                    if hasattr(vertex, 'dxf'):
                                        x_coords.append(float(vertex.dxf.location.x))
                                        y_coords.append(float(vertex.dxf.location.y))
                                    else:
                                        x_coords.append(float(vertex[0]))
                                        y_coords.append(float(vertex[1]))
                    except Exception as e:
                        # 跳過無法處理的實體
                        continue
                
                if x_coords and y_coords:
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    # 添加一些邊距
                    margin = max((xmax - xmin), (ymax - ymin)) * 0.1
                    if margin == 0:
                        margin = 10  # 如果範圍為0，使用固定邊距
                    xmin -= margin
                    ymin -= margin
                    xmax += margin
                    ymax += margin
                    print(f"[Viewer] Drawing extents: X[{xmin:.2f}, {xmax:.2f}], Y[{ymin:.2f}, {ymax:.2f}]")
                else:
                    # 如果無法計算範圍，使用預設值
                    xmin, ymin, xmax, ymax = -100, -100, 100, 100
                    print(f"[Viewer] Could not calculate extents from entities, using default range")
            except Exception as extents_error:
                print(f"[Viewer] Warning: Could not calculate extents: {extents_error}")
                xmin, ymin, xmax, ymax = -100, -100, 100, 100

            # 設定 Matplotlib 繪圖區
            # 使用明確的白色背景
            fig = plt.figure(figsize=(12, 12), facecolor='white', edgecolor='white')
            ax = fig.add_subplot(111, facecolor='white')
            
            # 確保背景是白色（多重設置確保生效）
            ax.set_facecolor('white')
            ax.patch.set_facecolor('white')
            ax.patch.set_edgecolor('white')
            fig.patch.set_facecolor('white')
            fig.patch.set_edgecolor('white')
            
            # 設置 Matplotlib 的樣式為白色背景
            plt.style.use('default')  # 使用預設樣式（白色背景）
            
            # 設置座標軸範圍和比例（在渲染前先設置，讓渲染器知道範圍）
            if xmin < xmax and ymin < ymax:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')  # 保持縱橫比
            ax.grid(True, alpha=0.3, color='lightgray', linestyle='--')  # 添加淺灰色網格
            ax.set_title(f"DXF Preview: {os.path.basename(filename)}", fontsize=14, pad=20, color='black')
            
            # 設置座標軸顏色為黑色
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.tick_params(colors='black')
            
            # 設置 RenderContext，確保線條顏色和寬度可見
            ctx = RenderContext(doc)
            # 設置預設線寬和顏色
            ctx.set_current_layout(msp)
            
            # 使用 ezdxf 的 Matplotlib 後端進行渲染
            out = MatplotlibBackend(ax)
            
            Frontend(ctx, out).draw_layout(msp, finalize=True)
            
            # 處理所有線條：確保在白色背景上可見
            for line in ax.lines:
                # 調整線寬
                current_width = line.get_linewidth()
                if current_width < 0.5:
                    line.set_linewidth(0.8)
                
                # 處理顏色：白色或接近白色的改為黑色
                color = line.get_color()
                if isinstance(color, str):
                    if color.lower() in ['white', 'w', '#ffffff', '#FFFFFF', '#fff', '#FFF']:
                        line.set_color('black')
                elif isinstance(color, (tuple, list)):
                    # RGB 或 RGBA 格式
                    if len(color) >= 3:
                        r, g, b = color[0], color[1], color[2]
                        # 如果 RGB 值都接近 1.0（白色），改為黑色
                        if all(isinstance(c, (int, float)) and c > 0.85 for c in [r, g, b]):
                            line.set_color('black')
                        # 如果 RGB 值都接近 0.0（黑色），確保是黑色
                        elif all(isinstance(c, (int, float)) and c < 0.2 for c in [r, g, b]):
                            line.set_color('black')
                
                # 確保不透明度
                if line.get_alpha() is None or (line.get_alpha() is not None and line.get_alpha() < 0.5):
                    line.set_alpha(1.0)
            
            # 處理所有集合（圓形、多邊形等）
            for collection in ax.collections:
                # 調整線寬
                current_width = collection.get_linewidth()
                if isinstance(current_width, (list, tuple)):
                    if len(current_width) > 0 and min(current_width) < 0.5:
                        collection.set_linewidth(0.8)
                elif isinstance(current_width, (int, float)) and current_width < 0.5:
                    collection.set_linewidth(0.8)
                else:
                    collection.set_linewidth(0.8)  # 預設線寬
                
                # 處理邊緣顏色
                edge_color = collection.get_edgecolor()
                if isinstance(edge_color, str):
                    if edge_color.lower() in ['white', 'w', '#ffffff', '#FFFFFF', '#fff', '#FFF', 'none']:
                        collection.set_edgecolor('black')
                elif isinstance(edge_color, (list, tuple)) and len(edge_color) > 0:
                    # 檢查第一個顏色（如果是陣列）
                    first_color = edge_color[0] if isinstance(edge_color[0], (list, tuple)) else edge_color
                    if isinstance(first_color, (list, tuple)) and len(first_color) >= 3:
                        r, g, b = first_color[0], first_color[1], first_color[2]
                        if all(isinstance(c, (int, float)) and c > 0.85 for c in [r, g, b]):
                            collection.set_edgecolor('black')
                
                # 填充顏色設為透明
                collection.set_facecolor('none')
            
            # 渲染後重新計算實際範圍並調整視圖
            # 手動計算所有實體的範圍（ezdxf 的 Modelspace 沒有 extents 方法）
            try:
                all_x = []
                all_y = []
                
                # 重新遍歷所有實體計算實際範圍
                for entity in msp:
                    try:
                        if entity.dxftype() == 'CIRCLE':
                            center = entity.dxf.center
                            radius = entity.dxf.radius
                            all_x.extend([float(center.x) - float(radius), float(center.x) + float(radius)])
                            all_y.extend([float(center.y) - float(radius), float(center.y) + float(radius)])
                        elif entity.dxftype() == 'LWPOLYLINE':
                            for vertex in entity.vertices():
                                if isinstance(vertex, (tuple, list)):
                                    all_x.append(float(vertex[0]))
                                    all_y.append(float(vertex[1]))
                                else:
                                    # 處理 numpy 類型
                                    try:
                                        all_x.append(float(vertex[0]))
                                        all_y.append(float(vertex[1]))
                                    except:
                                        pass
                        elif entity.dxftype() == 'LINE':
                            if hasattr(entity.dxf, 'start'):
                                all_x.append(float(entity.dxf.start.x))
                                all_y.append(float(entity.dxf.start.y))
                            if hasattr(entity.dxf, 'end'):
                                all_x.append(float(entity.dxf.end.x))
                                all_y.append(float(entity.dxf.end.y))
                    except Exception as e:
                        # 跳過無法處理的實體
                        continue
                
                if all_x and all_y:
                    actual_xmin, actual_xmax = min(all_x), max(all_x)
                    actual_ymin, actual_ymax = min(all_y), max(all_y)
                    margin_x = (actual_xmax - actual_xmin) * 0.1
                    margin_y = (actual_ymax - actual_ymin) * 0.1
                    if margin_x == 0:
                        margin_x = abs(actual_xmax - actual_xmin) * 0.05 if actual_xmax != actual_xmin else 10
                    if margin_y == 0:
                        margin_y = abs(actual_ymax - actual_ymin) * 0.05 if actual_ymax != actual_ymin else 10
                    
                    ax.set_xlim(actual_xmin - margin_x, actual_xmax + margin_x)
                    ax.set_ylim(actual_ymin - margin_y, actual_ymax + margin_y)
                    print(f"[Viewer] Final view range: X[{actual_xmin - margin_x:.2f}, {actual_xmax + margin_x:.2f}], Y[{actual_ymin - margin_y:.2f}, {actual_ymax + margin_y:.2f}]")
                    print(f"[Viewer] Content range: X[{actual_xmin:.2f}, {actual_xmax:.2f}], Y[{actual_ymin:.2f}, {actual_ymax:.2f}]")
                else:
                    # 使用原始計算的範圍
                    if xmin < xmax and ymin < ymax:
                        ax.set_xlim(xmin, xmax)
                        ax.set_ylim(ymin, ymax)
                        print(f"[Viewer] Using pre-calculated range: X[{xmin:.2f}, {xmax:.2f}], Y[{ymin:.2f}, {ymax:.2f}]")
            except Exception as adjust_error:
                print(f"[Viewer] Could not adjust view: {adjust_error}")
                import traceback
                traceback.print_exc()
                # 使用原始範圍
                if xmin < xmax and ymin < ymax:
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
            
            ax.set_aspect('equal', adjustable='box')
            
            # 最後檢查：確保所有繪圖元素在白色背景上可見
            # 將白色或接近白色的元素改為黑色
            for artist in ax.get_children():
                # 跳過座標軸、標籤、標題等非繪圖元素
                if artist in list(ax.spines.values()) or artist in ax.texts or artist == ax.title:
                    continue
                
                # 處理線條顏色
                if hasattr(artist, 'get_color') and hasattr(artist, 'set_color'):
                    try:
                        color = artist.get_color()
                        if isinstance(color, str):
                            if color.lower() in ['white', 'w', '#ffffff', '#FFFFFF', '#fff', '#FFF']:
                                artist.set_color('black')
                        elif isinstance(color, (tuple, list)) and len(color) >= 3:
                            r, g, b = color[0], color[1], color[2]
                            if all(isinstance(c, (int, float)) and c > 0.85 for c in [r, g, b]):
                                artist.set_color('black')
                    except:
                        pass
                
                # 處理邊緣顏色
                if hasattr(artist, 'get_edgecolor') and hasattr(artist, 'set_edgecolor'):
                    try:
                        edge_color = artist.get_edgecolor()
                        if isinstance(edge_color, str):
                            if edge_color.lower() in ['white', 'w', '#ffffff', '#FFFFFF', '#fff', '#FFF']:
                                artist.set_edgecolor('black')
                    except:
                        pass
                
                # 確保線寬足夠
                if hasattr(artist, 'set_linewidth'):
                    try:
                        current_width = artist.get_linewidth()
                        if isinstance(current_width, (int, float)) and current_width < 0.5:
                            artist.set_linewidth(0.8)
                    except:
                        pass
            
            # 添加標籤
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            
            # 統計渲染的實體
            entity_count = len(list(msp))
            print(f"[Viewer] Rendered {entity_count} entities")
            print(f"[Viewer] View range: X[{xmin:.2f}, {xmax:.2f}], Y[{ymin:.2f}, {ymax:.2f}]")
            
            # 如果沒有實體，顯示警告
            if entity_count == 0:
                print(f"[Viewer] WARNING: No entities found in DXF file!")
            
            # 檢查 matplotlib 後端
            backend = matplotlib.get_backend()
            print(f"[Viewer] Matplotlib backend: {backend}")
            
            # 方法 1: 嘗試使用阻塞模式顯示（最可靠）
            try:
                print(f"[Viewer] Attempting to display window (blocking mode)...")
                print(f"[Viewer] Note: Close the window to continue")
                # 使用阻塞模式，確保視窗顯示
                plt.show(block=True)
                print(f"[Viewer] Window closed by user")
                plt.close(fig)
                return
            except KeyboardInterrupt:
                # 用戶按 Ctrl+C 中斷，優雅退出
                print(f"\n[Viewer] Interrupted by user (Ctrl+C)")
                try:
                    plt.close(fig)
                except:
                    pass
                return
            except Exception as block_error:
                print(f"[Viewer] Blocking mode failed: {block_error}")
                try:
                    plt.close(fig)
                except:
                    pass
            
            # 方法 2: 嘗試非阻塞模式
            try:
                print(f"[Viewer] Attempting non-blocking display...")
                plt.ion()  # 開啟互動模式
                plt.show(block=False)
                
                # 強制刷新和顯示
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                # 嘗試顯示視窗管理器
                try:
                    if hasattr(fig.canvas, 'manager') and fig.canvas.manager:
                        fig.canvas.manager.show()
                        # 如果是 Qt 後端，確保視窗置頂
                        if 'Qt' in backend:
                            window = fig.canvas.manager.window
                            if hasattr(window, 'raise_'):
                                window.raise_()
                            if hasattr(window, 'activateWindow'):
                                window.activateWindow()
                except:
                    pass
                
                time.sleep(1.0)  # 給視窗更多時間顯示
                
                # 檢查視窗是否真的打開了
                if plt.get_fignums():
                    print(f"[Viewer] Window opened successfully (non-blocking)")
                    print(f"[Viewer] Note: Close the window manually when done viewing.")
                    # 保持視窗打開
                    try:
                        while plt.get_fignums():
                            plt.pause(0.5)
                    except KeyboardInterrupt:
                        print(f"\n[Viewer] Interrupted by user (Ctrl+C)")
                        plt.close(fig)
                        return
                    except:
                        pass
                else:
                    raise Exception("Window did not open")
                    
            except Exception as nonblock_error:
                print(f"[Viewer] Non-blocking display also failed: {nonblock_error}")
                # 方法 3: 保存為圖片並自動開啟
                raise Exception("Both display methods failed, will save as image")
                    
            except Exception as show_error:
                # 如果所有顯示方法都失敗，保存為圖片並自動開啟
                print(f"[Viewer] Could not display window: {show_error}")
                print(f"[Viewer] Saving as image and opening with system default viewer...")
                try:
                    image_filename = filename.replace('.dxf', '_preview.png')
                    plt.savefig(image_filename, dpi=150, bbox_inches='tight', facecolor='white')
                    print(f"[Viewer] ✓ Saved preview image to: {image_filename}")
                    
                    # 用系統預設程式開啟圖片
                    try:
                        system = platform.system()
                        if system == "Windows":
                            os.startfile(image_filename)
                            print(f"[Viewer] ✓ Opened preview image with Windows default viewer")
                        elif system == "Darwin":  # macOS
                            subprocess.run(["open", image_filename])
                            print(f"[Viewer] ✓ Opened preview image with macOS default viewer")
                        else:  # Linux
                            subprocess.run(["xdg-open", image_filename])
                            print(f"[Viewer] ✓ Opened preview image with Linux default viewer")
                    except Exception as open_error:
                        print(f"[Viewer] Could not open image automatically: {open_error}")
                        print(f"[Viewer] Please manually open: {image_filename}")
                    
                    plt.close(fig)
                except Exception as save_error:
                    print(f"[Viewer] ✗ Could not save image: {save_error}")
                    import traceback
                    traceback.print_exc()
                    plt.close(fig)
            
        except Exception as e:
            print(f"2D View Error: {e}")
            import traceback
            traceback.print_exc()

# ==========================================
# 單獨執行時的檔案選擇和處理
# ==========================================

def select_file_to_view():
    """
    開啟檔案選擇對話框，讓用戶選擇要查看的檔案
    返回選取的檔案路徑，如果取消則返回 None
    """
    try:
        from tkinter import filedialog, Tk
        
        # 隱藏主視窗（只顯示對話框）
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # 開啟檔案選擇對話框
        file_path = filedialog.askopenfilename(
            title="選擇要查看的檔案",
            filetypes=[
                ("3D 檔案", "*.stl *.step *.stp"),
                ("2D 檔案", "*.dxf *.dwg"),
                ("STEP 檔案", "*.step *.stp"),
                ("STL 檔案", "*.stl"),
                ("DXF 檔案", "*.dxf"),
                ("DWG 檔案", "*.dwg"),
                ("所有檔案", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        root.destroy()
        
        if file_path:
            return file_path
        return None
    except Exception as e:
        print(f"[Error] 無法開啟檔案選擇對話框: {e}")
        return None

def select_files_to_view():
    """
    開啟檔案選擇對話框，讓用戶選擇 3D 和 2D 檔案
    返回 (stp_file, dxf_file) 元組，如果取消則返回 (None, None)
    """
    try:
        from tkinter import filedialog, Tk, messagebox
        
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # 選擇 3D 檔案
        stp_file = filedialog.askopenfilename(
            title="選擇 3D 模型檔案 (STEP/STP)",
            filetypes=[
                ("STEP 檔案", "*.step *.stp"),
                ("STL 檔案", "*.stl"),
                ("所有檔案", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        if not stp_file:
            root.destroy()
            return None, None
        
        # 選擇 2D 檔案
        dxf_file = filedialog.askopenfilename(
            title="選擇 2D 工程圖檔案 (DXF/DWG)",
            filetypes=[
                ("DXF/DWG 檔案", "*.dxf *.dwg"),
                ("DXF 檔案", "*.dxf"),
                ("DWG 檔案", "*.dwg"),
                ("所有檔案", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        root.destroy()
        
        return stp_file, dxf_file
    except Exception as e:
        print(f"[Error] 無法開啟檔案選擇對話框: {e}")
        return None, None

def main():
    """
    主函數：處理命令列參數或檔案選擇
    支援同時顯示 3D 和 2D 視圖
    """
    import sys
    
    viewer = EngineeringViewer()
    
    # 檢查命令列參數
    if len(sys.argv) > 1:
        # 有命令列參數
        if len(sys.argv) == 3:
            # 兩個參數：3D 和 2D 檔案
            stp_file = sys.argv[1]
            dxf_file = sys.argv[2]
            
            if not os.path.exists(stp_file):
                print(f"錯誤：找不到 3D 檔案 {stp_file}")
                return
            if not os.path.exists(dxf_file):
                print(f"錯誤：找不到 2D 檔案 {dxf_file}")
                return
            
            print(f"同時顯示 3D 和 2D 視圖:")
            print(f"  3D: {stp_file}")
            print(f"  2D: {dxf_file}")
            viewer.view_3d_and_2d(stp_file, dxf_file)
        else:
            # 單個參數：單一檔案
            filename = sys.argv[1]
            if not os.path.exists(filename):
                print(f"錯誤：找不到檔案 {filename}")
                return
            
            file_ext = os.path.splitext(filename)[1].lower()
            
            # 根據副檔名選擇查看方式
            if file_ext in ['.stl', '.step', '.stp']:
                print(f"正在查看 3D 檔案: {filename}")
                viewer.view_3d_stl(filename)
            elif file_ext in ['.dxf', '.dwg']:
                print(f"正在查看 2D 檔案: {filename}")
                viewer.view_2d_dxf(filename)
            else:
                print(f"不支援的檔案格式: {file_ext}")
                print("支援的格式: .stl, .step, .stp (3D), .dxf, .dwg (2D)")
                print("或提供兩個參數同時顯示 3D 和 2D: python simple_viewer.py <3d_file> <2d_file>")
    else:
        # 沒有命令列參數，使用檔案選擇對話框
        print("=" * 60)
        print("工程檔案檢視器 (Engineering File Viewer)")
        print("=" * 60)
        print("\n選項:")
        print("  1. 查看單一檔案 (3D 或 2D)")
        print("  2. 同時查看 3D 和 2D 檔案")
        print()
        
        try:
            from tkinter import messagebox, Tk
            root = Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            choice = messagebox.askyesno(
                "選擇模式",
                "是否要同時查看 3D 和 2D 檔案？\n\n是 = 同時查看 3D 和 2D\n否 = 只查看單一檔案"
            )
            root.destroy()
            
            if choice:
                # 同時查看 3D 和 2D
                stp_file, dxf_file = select_files_to_view()
                if stp_file and dxf_file:
                    print(f"\n已選擇檔案:")
                    print(f"  3D: {stp_file}")
                    print(f"  2D: {dxf_file}")
                    viewer.view_3d_and_2d(stp_file, dxf_file)
                else:
                    print("未選擇檔案，程式結束。")
            else:
                # 單一檔案
                filename = select_file_to_view()
                if filename:
                    file_ext = os.path.splitext(filename)[1].lower()
                    print(f"\n已選擇檔案: {filename}")
                    
                    if file_ext in ['.stl', '.step', '.stp']:
                        print("正在開啟 3D 檢視器...")
                        viewer.view_3d_stl(filename)
                    elif file_ext in ['.dxf', '.dwg']:
                        print("正在開啟 2D 檢視器...")
                        viewer.view_2d_dxf(filename)
                    else:
                        print(f"不支援的檔案格式: {file_ext}")
                        print("支援的格式: .stl, .step, .stp (3D), .dxf, .dwg (2D)")
                else:
                    print("未選擇檔案，程式結束。")
        except Exception as e:
            print(f"無法顯示選擇對話框: {e}")
            # 回退到單一檔案選擇
            filename = select_file_to_view()
            if filename:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in ['.stl', '.step', '.stp']:
                    viewer.view_3d_stl(filename)
                elif file_ext in ['.dxf', '.dwg']:
                    viewer.view_2d_dxf(filename)

# ==========================================
# 使用範例
# ==========================================
if __name__ == "__main__":
    main()