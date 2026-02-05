# 系統架構

## 架構概覽

```
┌─────────────────────────────────────────────────────────────────┐
│                      AutoDrafter System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Input     │    │   Engine    │    │      Output         │ │
│  │             │    │             │    │                     │ │
│  │  STEP File  │───▶│ MockCAD     │───▶│  DXF (Drawing 1-3)  │ │
│  │  (.stp)     │    │ Engine      │    │  PNG (Preview)      │ │
│  │             │    │             │    │  TXT (Info)         │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Processing Pipeline                    │  │
│  │                                                          │  │
│  │  1. Load STEP    2. Extract      3. Classify   4. Calc   │  │
│  │     (XCAF)          Features        Parts        Angles  │  │
│  │        │               │              │            │      │  │
│  │        ▼               ▼              ▼            ▼      │  │
│  │  5. Detect       6. Compute     7. Build     8. Generate │  │
│  │     Sections        Bends         Cutting      Drawings  │  │
│  │                                    List                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 核心模組

### 1. MockCADEngine

主要的 CAD 引擎類別，負責所有幾何運算與繪圖生成。

```python
class MockCADEngine:
    def __init__(self, model_file: str)
    def load_3d_file(self, filepath: str)
    def generate_sub_assembly_drawing(self, output_dir: str) -> List[str]
    def export_projections_to_dxf(self, output_dir: str) -> List[str]
```

### 2. EngineeringViewer

GUI 檢視器，提供 3D 模型檢視與 2D 圖面預覽。

```python
class EngineeringViewer:
    def view_3d_model(self, filepath: str)
    def view_2d_dxf(cls, dxf_path: str, save_path: str = None)
```

### 3. GeometricFeature

幾何特徵資料結構。

```python
@dataclass
class GeometricFeature:
    id: str           # 特徵 ID (F01, F02...)
    type: str         # 類型 (solid, circle, edge...)
    props: dict       # 屬性 (尺寸、位置...)
    description: str  # 描述
```

## 處理流程

### Stage 1: 載入 STEP 檔案

```
STEP File ──▶ XCAF Parser ──▶ TopoDS_Shape ──▶ Solid Features
```

- 使用 OpenCASCADE XCAF 模組載入
- 提取組件樹結構
- 建立 Solid → Shape 映射

### Stage 2: 特徵提取

```
Solid ──▶ BRepGProp ──▶ Volume, Centroid, BBox
      ──▶ TopExp_Explorer ──▶ Faces, Edges
      ──▶ BRepAdaptor ──▶ Circles, Cylinders
```

提取的特徵：
- **Solid**: 體積、質心、邊界框
- **Circle**: 圓心、半徑、法向量
- **Edge**: 類型（直線/弧線）、長度

### Stage 3: 零件分類

```python
def _classify_parts(features) -> List[Dict]:
    # 根據幾何特徵分類
    # - 長條形 + 管狀 → track
    # - 垂直 + 中等長度 → leg
    # - 小體積 → bracket
```

分類依據：
- 邊界框比例
- 體積大小
- 方向向量
- 幾何特徵組合

### Stage 4: 角度計算

```python
def _calculate_angles() -> List[Dict]:
    # 計算各種角度關係
    # - track_elevation: 軌道仰角
    # - leg_to_ground: 腳架安裝角
    # - track_bend: 彎管角度
```

### Stage 5: 軌道分段

```python
def _detect_track_sections() -> List[Dict]:
    # 將軌道分為 straight 和 curved 區段
    # straight: 仰角 < 60°
    # curved: 包含大角度弧形
```

### Stage 6: 轉折彎計算

```python
def _compute_transition_bends() -> List[Dict]:
    # 計算直軌段間的虛擬轉折彎
    # - 從相鄰軌道的仰角差推算
    # - 提取實際弧半徑
```

### Stage 7: 取料明細

```python
def _generate_cutting_list() -> Dict:
    # 生成取料明細
    # - track_items: 軌道段 (straight/arc)
    # - leg_items: 腳架
    # - bracket_items: 支撐架
```

### Stage 8: 繪圖生成

```python
def generate_sub_assembly_drawing() -> List[str]:
    # 生成 3 張施工圖
    # Drawing 1: 直線段施工圖
    # Drawing 2: 彎軌施工圖
    # Drawing 3: 完整組合施工圖
```

## 資料流

```
┌──────────────┐
│  STEP File   │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌────────────────────┐
│   features   │────▶│ part_classifications│
│   (List)     │     │      (List)        │
└──────┬───────┘     └─────────┬──────────┘
       │                       │
       ▼                       ▼
┌──────────────┐     ┌────────────────────┐
│pipe_centerlines│    │      angles        │
│   (List)     │     │      (List)        │
└──────┬───────┘     └─────────┬──────────┘
       │                       │
       └───────────┬───────────┘
                   ▼
          ┌───────────────┐
          │   sections    │
          │    (List)     │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │ cutting_list  │
          │    (Dict)     │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │  DXF Files    │
          └───────────────┘
```

## 關鍵資料結構

### Part Classification

```python
{
    'feature_id': 'F02',
    'class': 'track',
    'centroid': (x, y, z),
    'bbox': (length, width, height),
    'volume': 12345.6,
}
```

### Pipe Centerline

```python
{
    'solid_id': 'F02',
    'pipe_diameter': 48.1,
    'total_length': 221.3,
    'segments': [
        {'type': 'straight', 'length': 221.3},
        {'type': 'arc', 'angle_deg': 12, 'radius': 270},
    ],
    'start_point': (x1, y1, z1),
    'end_point': (x2, y2, z2),
}
```

### Track Section

```python
{
    'section_type': 'straight',  # or 'curved'
    'upper_tracks': [...],
    'lower_tracks': [...],
    'legs': [...],
}
```

### Cutting List Item

```python
# 直管
{
    'item': 'U1',
    'type': 'straight',
    'rail': 'upper',
    'diameter': 48.1,
    'length': 221.3,
}

# 彎管
{
    'item': 'U2',
    'type': 'arc',
    'rail': 'upper',
    'diameter': 48.1,
    'angle_deg': 12,
    'radius': 270,
    'outer_arc_length': 62,
}
```

## 依賴關係

```
auto_drafter_system.py
├── cadquery-ocp (OpenCASCADE)
│   ├── OCP.TopoDS - 拓撲資料結構
│   ├── OCP.BRepGProp - 幾何屬性計算
│   ├── OCP.TopExp - 拓撲探索
│   └── OCP.STEPCAFControl - STEP 讀取
├── ezdxf - DXF 檔案生成
├── numpy - 數值計算
└── matplotlib - 圖形預覽

simple_viewer.py
├── auto_drafter_system.py
├── matplotlib - GUI 顯示
└── PIL - 影像處理
```

## 另請參閱

- [API 參考](api/MockCADEngine.md)
- [繪圖說明](drawings/)
