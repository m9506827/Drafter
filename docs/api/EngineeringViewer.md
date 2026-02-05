# EngineeringViewer API 參考

## 概述

`EngineeringViewer` 提供 3D 模型檢視與 2D 圖面預覽功能。

## 類別定義

```python
class EngineeringViewer:
    """
    工程 3D/2D 檢視器
    支援 STEP 模型檢視與 DXF 圖面預覽
    """
```

## 建構函式

```python
def __init__(self)
```

## 類別方法

### view_3d_model

開啟 3D 模型檢視器。

```python
@classmethod
def view_3d_model(cls, filepath: str, fast_mode: bool = False) -> bool
```

#### 參數
| 參數 | 類型 | 說明 |
|-----|------|------|
| `filepath` | `str` | STEP 檔案路徑 |
| `fast_mode` | `bool` | 快速模式（降低品質提升速度） |

#### 回傳
`bool` - 成功開啟回傳 `True`

#### 範例
```python
EngineeringViewer.view_3d_model("model.stp")
```

---

### view_2d_dxf

檢視或儲存 2D DXF 圖面。

```python
@classmethod
def view_2d_dxf(cls, dxf_path: str, 
                fast_mode: bool = False,
                save_path: str = None) -> bool
```

#### 參數
| 參數 | 類型 | 說明 |
|-----|------|------|
| `dxf_path` | `str` | DXF 檔案路徑 |
| `fast_mode` | `bool` | 快速模式 |
| `save_path` | `str` | PNG 儲存路徑（可選） |

#### 回傳
`bool` - 成功執行回傳 `True`

#### 範例
```python
# 顯示 DXF
EngineeringViewer.view_2d_dxf("drawing.dxf")

# 儲存為 PNG
EngineeringViewer.view_2d_dxf("drawing.dxf", save_path="preview.png")
```

---

## 使用範例

### 完整工作流程

```python
from auto_drafter_system import MockCADEngine
from simple_viewer import EngineeringViewer

# 1. 載入模型
cad = MockCADEngine("model.stp")

# 2. 檢視 3D 模型
EngineeringViewer.view_3d_model("model.stp")

# 3. 生成施工圖
dxf_files = cad.generate_sub_assembly_drawing("output/")

# 4. 檢視每張圖
for dxf in dxf_files:
    EngineeringViewer.view_2d_dxf(dxf)

# 5. 儲存預覽圖
for dxf in dxf_files:
    png = dxf.replace('.dxf', '.png')
    EngineeringViewer.view_2d_dxf(dxf, save_path=png)
```

### 批次處理

```python
import os
from glob import glob

# 處理所有 DXF 檔案
for dxf in glob("output/*.dxf"):
    png = dxf.replace('.dxf', '_preview.png')
    EngineeringViewer.view_2d_dxf(dxf, fast_mode=True, save_path=png)
```

## 注意事項

1. **GUI 環境**：`view_3d_model` 需要 GUI 環境
2. **無頭模式**：使用 `save_path` 可在無 GUI 環境下儲存圖片
3. **matplotlib**：需要安裝 matplotlib 套件

## 另請參閱

- [MockCADEngine](MockCADEngine.md)
- [使用者手冊](../user-guide.md)
