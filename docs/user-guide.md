# 使用者手冊

## 概述

AutoDrafter 是一套自動化工程製圖系統，專為彎管軌道系統設計。系統可從 3D STEP 檔案自動：

1. 提取幾何特徵（軌道、腳架、支撐架）
2. 分類零件並計算角度
3. 生成標準化施工圖（DXF 格式）

## 支援的檔案格式

### 輸入
- `.stp` / `.step` - STEP 3D 模型檔案

### 輸出
- `.dxf` - AutoCAD 相容的 2D 工程圖
- `.png` - 圖面預覽
- `.txt` - 模型資訊

## 命令列使用

### 基本語法

```bash
python simple_viewer.py <STEP檔案> [選項]
```

### 選項說明

| 選項 | 說明 |
|-----|------|
| `--dxf` | 生成 DXF 施工圖 |
| `--no-gui` | 不顯示 GUI 視窗 |
| `--output <目錄>` | 指定輸出目錄（預設: output/） |

### 使用範例

```bash
# 檢視 3D 模型
python simple_viewer.py model.stp

# 生成施工圖（含 GUI）
python simple_viewer.py model.stp --dxf

# 批次生成（無 GUI）
python simple_viewer.py model.stp --dxf --no-gui

# 指定輸出目錄
python simple_viewer.py model.stp --dxf --output my_drawings/
```

## 程式化使用

### 基本用法

```python
from auto_drafter_system import MockCADEngine

# 載入 STEP 檔案
cad = MockCADEngine("model.stp")

# 生成施工圖
dxf_files = cad.generate_sub_assembly_drawing("output/")

print(f"生成了 {len(dxf_files)} 個 DXF 檔案")
```

### 取得模型資訊

```python
# 顯示模型資訊
cad.display_info()

# 儲存資訊到檔案
info_path = cad.save_info_to_file("output/")

# 取得取料明細
cutting_list = cad._cutting_list
```

### 生成 2D 投影

```python
# 生成 XY, XZ, YZ 投影
dxf_files = cad.export_projections_to_dxf("output/")
```

## 零件分類

系統自動將 3D 模型中的零件分類為：

| 類型 | 說明 | 特徵 |
|-----|------|------|
| `track` | 軌道 | 長條形管件，長度 > 100mm |
| `leg` | 腳架 | 垂直支撐，長度 > 200mm |
| `bracket` | 支撐架 | 連接件，體積較小 |
| `fitting` | 接頭 | 管路配件 |

## 角度計算

系統自動計算以下角度：

- **軌道仰角** - 軌道相對於水平面的角度
- **腳架安裝角** - 腳架相對於地面的角度
- **彎管角度** - 管路彎折角度
- **轉折角度** - 相鄰軌道段的仰角差

## 取料明細

系統自動生成取料明細，包含：

### 直管段
- 球號（U1, U2... / D1, D2...）
- 管徑
- 長度

### 彎管段
- 球號
- 管徑
- 角度
- 半徑
- 外弧長

### 腳架
- 線長
- 安裝角度

## 輸出圖面規格

### Drawing 0: 組件總覽圖
- 等角視圖（HLR 投影，從右前上方觀看）
- 俯視圖（HLR 投影，從上往下觀看）
- 上軌端點內側距離標註
- 軌道外側跨距標註
- 弧半徑標註 (R250)

### Drawing 1: 直線段施工圖
- 側視圖：上下軌 + 腳架
- 取料明細表
- BOM 表
- 標題欄

### Drawing 2: 彎軌施工圖
- 正面弧形視圖
- 等角視圖
- 取料明細表
- BOM 表

### Drawing 3: 完整組合施工圖
- 完整軌道路徑
- 所有腳架位置
- 完整取料明細
- 完整 BOM

## 設定檔 (`drafter_config.json`)

系統支援透過 JSON 設定檔自訂標題欄內容。將 `drafter_config.json` 放在
STEP 檔案同目錄或專案根目錄即可自動載入。

```json
{
  "company": "iDrafter股份有限公司",
  "drawer": "Drafter",
  "material": "STK-400",
  "finish": "裁切及焊接",
  "scale": "1:10",
  "quantity": "1"
}
```

可設定欄位：`company`, `drawer`, `material`, `finish`, `scale`, `quantity`, `project`, `date`

> **注意**：`version`（版次）由程式版本 `__version__` 自動帶入，不在設定檔中設定。

### 優先順序

1. 函式參數覆蓋 (overrides)
2. 設定檔 (`drafter_config.json`)
3. STEP 中繼資料（案名、日期、單位）
4. 程式預設值 (`_TB_DEFAULTS`)

## 圖面規範

- 圖紙大小：A3 橫向（420 x 297 mm）
- 比例：1:10（可透過設定檔修改）
- 單位：mm
- 材料：STK-400（可透過設定檔修改）
- 標準：依據工程製圖規範
- 版次：自動取用程式版本 (`__version__`)

## 故障排除

### 常見問題

**Q: STEP 檔案載入失敗**
```
確認檔案路徑正確，且檔案為有效的 STEP 格式
```

**Q: DXF 檔案無法開啟**
```
確認使用支援 DXF R2010 格式的 CAD 軟體
```

**Q: 預覽圖生成失敗**
```
確認已安裝 matplotlib 並有圖形環境
無 GUI 環境請使用 --no-gui 選項
```

## 另請參閱

- [系統架構](architecture.md)
- [API 參考](api/MockCADEngine.md)
- [繪圖說明](drawings/)
- [測試說明](testing.md)
