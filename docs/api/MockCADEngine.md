# MockCADEngine API 參考

## 概述

`MockCADEngine` 是 AutoDrafter 系統的核心引擎，負責 STEP 檔案載入、幾何分析與施工圖生成。

## 類別定義

```python
class MockCADEngine:
    """
    CAD 引擎，負責幾何運算與 3D→2D 投影
    支援從 STEP 檔案讀取 3D 模型
    """
```

## 建構函式

```python
def __init__(self, model_file: Optional[str] = None)
```

### 參數
| 參數 | 類型 | 說明 |
|-----|------|------|
| `model_file` | `str` | STEP 檔案路徑（可選） |

### 範例
```python
# 建立引擎並載入模型
cad = MockCADEngine("model.stp")

# 建立空引擎（使用模擬資料）
cad = MockCADEngine()
```

## 模組常數

```python
__version__ = '1.0.0'   # 程式版本，標題欄「版次」來源
```

## 類別常數

```python
_TB_DEFAULTS = {
    'company':   'iDrafter股份有限公司',
    'drawer':    'Drafter',
    'units':     'mm',
    'scale':     '1:10',
    'material':  'STK-400',
    'finish':    '裁切及焊接',
    'version':   __version__,
    'quantity':  '1',
}
```

## 屬性

| 屬性 | 類型 | 說明 |
|-----|------|------|
| `model_file` | `str` | 載入的模型檔案路徑 |
| `cad_model` | `object` | CadQuery 模型物件 |
| `features` | `List[GeometricFeature]` | 提取的幾何特徵 |
| `_drafter_config` | `Dict` | 從 `drafter_config.json` 載入的設定 |
| `_pipe_centerlines` | `List[Dict]` | 管路中心線資料 |
| `_part_classifications` | `List[Dict]` | 零件分類結果 |
| `_angles` | `List[Dict]` | 角度計算結果 |
| `_cutting_list` | `Dict` | 取料明細 |

## 公開方法

### load_3d_file

載入 3D 模型檔案。

```python
def load_3d_file(self, filepath: str) -> bool
```

#### 參數
| 參數 | 類型 | 說明 |
|-----|------|------|
| `filepath` | `str` | STEP 檔案路徑 |

#### 回傳
`bool` - 載入成功回傳 `True`

#### 範例
```python
cad = MockCADEngine()
success = cad.load_3d_file("model.stp")
```

---

### display_info

顯示模型資訊到主控台。

```python
def display_info(self) -> bool
```

#### 回傳
`bool` - 有資訊顯示回傳 `True`

---

### save_info_to_file

儲存模型資訊到文字檔。

```python
def save_info_to_file(self, output_dir: str = "output") -> str
```

#### 參數
| 參數 | 類型 | 說明 |
|-----|------|------|
| `output_dir` | `str` | 輸出目錄（預設 "output"） |

#### 回傳
`str` - 輸出檔案的完整路徑

---

### generate_sub_assembly_drawing

生成子系統施工圖（4 張）。

```python
def generate_sub_assembly_drawing(self, output_dir: str = "output") -> List[str]
```

#### 參數
| 參數 | 類型 | 說明 |
|-----|------|------|
| `output_dir` | `str` | 輸出目錄（預設 "output"） |

#### 回傳
`List[str]` - 生成的 DXF 檔案路徑列表

#### 生成的檔案
| 檔案 | 說明 |
|-----|------|
| `{name}-0.dxf` | Drawing 0: 組件總覽圖（等角 + 俯視） |
| `{name}-1.dxf` | Drawing 1: 直線段施工圖 |
| `{name}_2.dxf` | Drawing 2: 彎軌施工圖 |
| `{name}_3.dxf` | Drawing 3: 完整組合施工圖 |
| `{name}_*_preview.png` | PNG 預覽圖 |

#### 範例
```python
cad = MockCADEngine("model.stp")
dxf_files = cad.generate_sub_assembly_drawing("output/")
print(f"生成了 {len(dxf_files)} 個檔案")
```

---

### export_projections_to_dxf

生成 2D 投影圖（XY, XZ, YZ 平面）。

```python
def export_projections_to_dxf(self, output_dir: str = "output") -> List[str]
```

#### 參數
| 參數 | 類型 | 說明 |
|-----|------|------|
| `output_dir` | `str` | 輸出目錄（預設 "output"） |

#### 回傳
`List[str]` - 生成的 DXF 檔案路徑列表

#### 生成的檔案
| 檔案 | 說明 |
|-----|------|
| `{name}_XY.dxf` | XY 平面投影（俯視圖） |
| `{name}_XZ.dxf` | XZ 平面投影（正視圖） |
| `{name}_YZ.dxf` | YZ 平面投影（側視圖） |

---

---

### _load_drafter_config

載入標題欄設定檔。

```python
def _load_drafter_config(self, model_file: str) -> None
```

#### 說明
搜尋 `drafter_config.json`，順序為：
1. `model_file` 同目錄
2. 目前工作目錄 (CWD)

找到後載入至 `self._drafter_config`。

---

### _build_tb_info

建構標題欄資訊字典（所有 Drawing 共用）。

```python
def _build_tb_info(self, info: Dict, base_name: str,
                   drawing_name: str, drawing_number: str,
                   today: str, **overrides) -> Dict
```

#### 參數
| 參數 | 類型 | 說明 |
|-----|------|------|
| `info` | `Dict` | `get_model_info()` 的回傳值（含 STP 中繼資料） |
| `base_name` | `str` | 檔案基本名稱（如 '2-2'） |
| `drawing_name` | `str` | 圖面名稱（如 '彎軌軌道總圖'） |
| `drawing_number` | `str` | 圖號（如 '2-2', 'LM-11'） |
| `today` | `str` | 日期字串 YYYY/MM/DD |
| `**overrides` | | 額外覆蓋值 |

#### 回傳
`Dict` - 包含以下欄位：`company`, `project`, `drawing_name`, `drawer`, `date`, `units`, `scale`, `material`, `finish`, `drawing_number`, `version`, `quantity`

#### 優先順序
`overrides` > `drafter_config.json` > STP 中繼資料 > `_TB_DEFAULTS`

> **注意**：`version` 欄位固定取用 `__version__`，不受 config 影響。

---

## 內部方法

以下方法為內部使用，但可供進階使用者參考。

### _extract_step_metadata

從 STEP 標頭擷取中繼資料。

```python
def _extract_step_metadata(self, step_path: str) -> Dict
```

#### 回傳的字典結構
```python
{
    'product_name': str,         # 產品名稱（PRODUCT id 或 name）
    'source_software': str,      # 來源軟體
    'author': str,               # 作者
    'organization': str,         # 組織
    'creation_date': str,        # 建立日期 (YYYY/MM/DD)
    'file_description': str,     # 檔案描述
}
```

---

### _extract_geometric_features

提取幾何特徵。

```python
def _extract_geometric_features(self) -> List[GeometricFeature]
```

---

### _classify_parts

分類零件。

```python
def _classify_parts(self, features: List) -> List[Dict]
```

#### 回傳的字典結構
```python
{
    'feature_id': str,      # 特徵 ID
    'class': str,           # 分類 (track/leg/bracket/fitting)
    'centroid': tuple,      # 質心座標
    'bbox': tuple,          # 邊界框尺寸
    'volume': float,        # 體積
}
```

---

### _extract_pipe_centerlines

提取管路中心線。

```python
def _extract_pipe_centerlines(self, features, part_classifications) -> List[Dict]
```

#### 回傳的字典結構
```python
{
    'solid_id': str,        # Solid ID
    'pipe_diameter': float, # 管徑
    'total_length': float,  # 總長度
    'segments': List[Dict], # 管段列表
    'start_point': tuple,   # 起點座標
    'end_point': tuple,     # 終點座標
}
```

---

### _calculate_angles

計算角度關係。

```python
def _calculate_angles(self) -> List[Dict]
```

#### 回傳的字典結構
```python
{
    'type': str,        # 角度類型
    'part_a': str,      # 零件 A
    'part_b': str,      # 零件 B
    'angle_deg': float, # 角度（度）
    'description': str, # 描述
}
```

角度類型：
- `track_elevation` - 軌道仰角
- `leg_to_ground` - 腳架安裝角
- `track_bend` - 彎管角度
- `leg_to_track` - 腳架與軌道夾角

---

### _detect_track_sections

偵測軌道區段。

```python
def _detect_track_sections(self, pipe_centerlines, part_classifications, track_items) -> List[Dict]
```

#### 回傳的字典結構
```python
{
    'section_type': str,     # 'straight' 或 'curved'
    'upper_tracks': List,    # 上軌列表
    'lower_tracks': List,    # 下軌列表
}
```

---

### _compute_transition_bends

計算轉折彎。

```python
def _compute_transition_bends(self, section, track_elevations, 
                              pipe_centerlines, part_classifications,
                              pipe_diameter, rail_spacing) -> List[Dict]
```

#### 回傳的字典結構
```python
{
    'angle_deg': float,       # 統一角度
    'upper_bend_deg': float,  # 上軌角度
    'lower_bend_deg': float,  # 下軌角度
    'upper_r': float,         # 上軌半徑
    'lower_r': float,         # 下軌半徑
    'upper_arc': float,       # 上軌弧長
    'lower_arc': float,       # 下軌弧長
}
```

---

### _generate_cutting_list

生成取料明細。

```python
def _generate_cutting_list(self) -> Dict
```

#### 回傳的字典結構
```python
{
    'track_items': List[Dict],    # 軌道項目
    'leg_items': List[Dict],      # 腳架項目
    'bracket_items': List[Dict],  # 支撐架項目
}
```

## 錯誤處理

所有方法在發生錯誤時會：
1. 記錄錯誤到日誌
2. 回傳空值或 `False`
3. 不拋出例外（除非是致命錯誤）

## 另請參閱

- [EngineeringViewer](EngineeringViewer.md)
- [系統架構](../architecture.md)
