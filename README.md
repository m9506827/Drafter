# Drafter - AI 自動繪圖系統

> **版本 1.0.0**

一個智能的 3D 到 2D 工程圖自動轉換系統，支援 STEP/STL 檔案轉換為 DXF 工程圖。

## 功能特色

- **四張施工圖自動生成**：從 STEP 檔案自動產出 4 張標準化 DXF 施工圖（總覽圖 + 直線段 + 彎軌 + 組合圖）
- **HLR 投影**：使用 OpenCASCADE Hidden Line Removal 產生等角視圖、俯視圖等精確 2D 線圖
- **智能特徵提取**：自動從 3D 模型中提取管路中心線、軌道、腳架、支撐架等幾何特徵
- **STEP 中繼資料解析**：自動從 STEP 標頭擷取作者、日期、軟體、產品名稱等資訊
- **設定檔系統**：透過 `drafter_config.json` 自訂公司名稱、繪圖者、材料等標題欄資訊
- **程式版本管理**：標題欄「版次」自動取用程式版本 (`__version__`)，無需手動維護
- **3D 模型檢視**：支援 STL、STEP/STP 格式的 3D 模型預覽（使用 PyVista）
- **2D 工程圖檢視**：支援 DXF、DWG 格式的 2D 工程圖檢視（使用 Matplotlib）
- **Headless 測試**：支援無 GUI 的自動投影測試，可輸出 PNG 並與基準圖比對（SSIM）
- **自動儲存**：轉換後的檔案自動儲存到 `output/` 目錄

## 系統需求

- Python 3.8+
- 以下 Python 套件：
  - `cadquery` - 3D CAD 建模和 STEP 檔案讀取
  - `ezdxf` - DXF/DWG 檔案處理
  - `pyvista` - 3D 視覺化
  - `matplotlib` - 2D 繪圖
  - `tkinter` - GUI 檔案選擇（Python 內建）

## 安裝

```bash
# 安裝主要依賴
pip install cadquery ezdxf pyvista matplotlib

# 如果需要 DWG 支援（可選）
pip install 'ezdxf[odafc]'
```

### ODA File Converter（DWG 支援，可選）

如果要讀取 DWG 檔案，需要安裝 ODA File Converter：
- 下載：https://www.opendesign.com/guestfiles/oda_file_converter
- 安裝後，程式會自動檢測

## 使用方式

### 1. 自動繪圖系統（主要功能）

```bash
python auto_drafter_system.py
```

**執行流程：**
1. 彈出檔案選擇對話框，選擇 3D 模型檔案（STEP/STP）
2. 若同目錄或專案根目錄有 `drafter_config.json`，自動載入設定
3. 開啟圖檔資訊視窗，顯示 STEP 中繼資料、幾何特徵、BOM 等
4. 自動生成 **4 張施工圖** 到 `output/` 目錄：
   - `{name}-0.dxf` — Drawing 0: 組件總覽圖（等角視圖 + 俯視圖）
   - `{name}-1.dxf` — Drawing 1: 直線段施工圖
   - `{name}_2.dxf` — Drawing 2: 彎軌施工圖
   - `{name}_3.dxf` — Drawing 3: 完整組合施工圖
5. 自動顯示 2D 工程圖視窗（Matplotlib）

### 2. 檔案檢視器（獨立工具）

```bash
# 互動模式：彈出檔案選擇對話框
python simple_viewer.py

# 查看指定檔案（命令列）
python simple_viewer.py 1-2.stp          # 3D 檔案
python simple_viewer.py output/1-2.dxf    # 2D 檔案
python simple_viewer.py LM-02.dwg         # DWG 檔案

# 同時查看 3D 和 2D（對比模式）
python simple_viewer.py 1-2.stp output/1-2.dxf
```

**檢視器功能：**
- 自動識別檔案類型（3D 或 2D）
- 支援白色背景、黑色線條的清晰顯示
- 自動調整視圖範圍和比例
- 保留原始顏色（白色線條會自動轉為黑色以確保可見）

### 3. 投影測試（Headless）

```bash
# 生成投影 + PNG（無 GUI）
python test/test_projection.py

# 將當前輸出存為基準圖
python test/test_projection.py --save-reference

# 之後每次執行會自動與基準圖比對 SSIM
python test/test_projection.py
```

**測試流程：**
1. 載入 STEP 檔案，生成 6 方向 DXF 投影
2. 每張 DXF 渲染為 PNG（存於 `test/output/`）
3. 若 `test/reference/` 有基準圖，自動進行 SSIM 比對
4. 輸出 PASS/FAIL 結果摘要

## 設定檔 (`drafter_config.json`)

可在 STEP 檔案同目錄或專案根目錄建立 `drafter_config.json`，自訂標題欄資訊：

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

### 欄位說明

| 欄位 | 說明 | 預設值 |
|------|------|--------|
| `company` | 公司名稱 | iDrafter股份有限公司 |
| `drawer` | 繪圖者 | Drafter |
| `material` | 材料 | STK-400 |
| `finish` | 表面處理 | 裁切及焊接 |
| `scale` | 比例 | 1:10 |
| `quantity` | 數量 | 1 |
| `project` | 案名（覆蓋 STP 產品名稱） | *(自動從 STP 讀取)* |
| `date` | 日期（覆蓋 STP 建立日期） | *(自動從 STP 讀取)* |

> **注意**：`version`（版次）由程式版本 `__version__` 自動帶入，不在設定檔中設定。

### 優先順序

標題欄各欄位的取值優先順序為：

1. **函式參數覆蓋** (overrides) — 各圖面特定值
2. **設定檔** (`drafter_config.json`) — 使用者自訂
3. **STEP 中繼資料** — 從 STEP 標頭自動擷取（案名、日期、單位）
4. **程式預設值** (`_TB_DEFAULTS`) — 硬編碼預設

## 專案結構

```
Drafter/
├── auto_drafter_system.py       # 主要系統：3D 到 2D 自動轉換
│   ├── __version__              #   程式版本號（標題欄版次來源）
│   ├── MockCADEngine            #   CAD 核心（模型載入、特徵提取、投影）
│   │   ├── _TB_DEFAULTS         #     標題欄預設值
│   │   ├── _load_drafter_config #     載入 drafter_config.json
│   │   ├── _build_tb_info       #     建構標題欄（統一優先順序）
│   │   └── _extract_step_metadata #   STEP 中繼資料解析
│   ├── AIIntentParser           #   AI 意圖解析器
│   └── AutoDraftingSystem       #   系統協調器
├── simple_viewer.py             # 檔案檢視器：3D/2D 檢視 + PNG 輸出
│   └── EngineeringViewer        #   工程檔案檢視器類別
├── drafter_config.json          # 標題欄設定檔（可自訂）
├── generate_assembly_drawing.py # 組合件工程圖生成器
├── info2xml.py                  # STEP 資訊轉 XML 工具
├── test/                        # 自動化測試
│   ├── verify_values.py         #   主要數值驗證測試（67 項）
│   ├── test_projection.py       #   Headless 投影測試腳本
│   ├── 2-2.stp                  #   測試用 STEP 檔案
│   ├── output/                  #   測試輸出 PNG（自動生成）
│   └── reference/               #   基準 PNG（用於 SSIM 比對）
├── docs/                        # 完整技術文件
├── output/                      # 輸出目錄（自動生成）
├── .gitignore                   # Git 忽略規則
└── README.md                    # 本文件
```

## 支援的檔案格式

### 3D 格式（輸入）
- **STEP/STP**：使用 CadQuery 讀取和轉換
  - 支援參數化 CAD 模型
  - 可提取幾何特徵（圓形、邊緣等）
- **STL**：使用 PyVista 讀取
  - 網格格式，主要用於預覽
  - 特徵提取功能有限

### 2D 格式（輸入/輸出）
- **DXF**：使用 ezdxf 讀取/寫入
  - 完全支援讀取和寫入
  - 保留原始顏色和線條樣式
- **DWG**：使用 ezdxf + ODA File Converter 讀取
  - 需要安裝 ODA File Converter
  - 支援 AutoCAD 原生格式
  - 自動轉換為 DXF 後讀取

## 技術架構

### 核心模組

1. **MockCADEngine** (`auto_drafter_system.py`)
   - 負責 3D 模型載入（STEP/STL）
   - 特徵提取（管路中心線、軌道分段、角度計算、取料明細）
   - HLR (Hidden Line Removal) 投影：產生等角視圖、俯視圖等 2D 線圖
   - STEP 中繼資料解析：從 `FILE_NAME` 和 `PRODUCT` 實體擷取作者、日期、軟體等
   - 設定檔系統：載入 `drafter_config.json` 並與 STP 中繼資料合併
   - 標題欄建構：`_build_tb_info()` 統一管理各欄位的優先順序

2. **AIIntentParser** (`auto_drafter_system.py`)
   - AI 意圖解析器（目前為簡化版本，使用關鍵字匹配）
   - 可擴展整合真實的 LLM API（OpenAI、Claude 等）

3. **AutoDraftingSystem** (`auto_drafter_system.py`)
   - 系統協調器，整合所有功能
   - 處理用戶請求，生成 2D 工程圖
   - 自動檔案管理和開啟

4. **EngineeringViewer** (`simple_viewer.py`)
   - 工程檔案檢視器
   - 支援 3D 視圖（PyVista）和 2D 視圖（Matplotlib）
   - 支援 `save_path` 參數：指定路徑時輸出 PNG，不開啟 GUI 視窗

### 工作流程

```
3D 模型 (STEP/STP)
    ↓
XCAF 載入 + STEP 中繼資料解析
    ↓
drafter_config.json 載入（可選）
    ↓
幾何特徵提取 → 管路中心線 → 軌道分段 → 角度計算 → 取料明細
    ↓
生成 4 張施工圖（Drawing 0-3）
    ↓
儲存到 output/ 目錄 + 標題欄（config + STP 中繼資料 + 預設值）
    ↓
顯示 2D 工程圖
```

### 擴展建議

- **整合 LLM API**：擴展 `AIIntentParser` 整合 OpenAI/Claude API
- **改進特徵提取**：增強幾何特徵提取支援更多管路結構
- **多語言標題欄**：擴展設定檔支援多語系標題欄
- **批次處理**：支援多個 STEP 檔案批次生成施工圖

## 更新日誌

### v1.0.0 (最新)
- **四張施工圖**：新增 Drawing 0 組件總覽圖（等角視圖 + 俯視圖），系統完整產出 4 張施工圖
- **Drawing 0 投影修正**：等角視圖 `(1,1,1)` 與俯視圖 `(0,0,-1)` 方向正確對齊標準圖
- **Drawing 0 尺寸修正**：等角視圖內側距離計算扣除管徑、俯視圖外側跨距加上管徑、半徑標註使用實際弧半徑 R250
- **設定檔系統**：新增 `drafter_config.json` 外部設定檔，自訂公司、繪圖者、材料等標題欄欄位
- **STEP 中繼資料增強**：解析 `FILE_NAME` 實體取得建立日期、作者、軟體；改善 `PRODUCT` 實體解析
- **標題欄重構**：新增 `_build_tb_info()` 共用方法，統一 4 張圖的標題欄建構邏輯（overrides > config > STP > defaults）
- **程式版本管理**：標題欄「版次」自動取用 `__version__`，不再由設定檔覆蓋

### v0.3
- **六方向投影**：XY/XZ/YZ 直接投影 + 反向投影，共 6 張 DXF
- **完整邊緣提取**：使用 OCP `TopExp_Explorer` 遍歷 Compound 中所有 Solid
- **Headless 測試**：`test/test_projection.py` 支援無 GUI 自動測試 + SSIM 基準比對
- **PNG 輸出**：`view_2d_dxf()` 和 `view_projected_2d()` 支援 `save_path` 參數直接輸出 PNG

### v0.2
- 圖檔資訊視窗：獨立視窗完整顯示圖檔資訊
- BOM 材料清單：自動生成並顯示
- 來源軟體識別、單位資訊、元資料提取
- 幾何統計：實體數、面數、邊數、頂點數、體積、表面積

### v0.1
- 基本 3D 到 2D 轉換功能
- 支援 STEP/STL 檔案讀取
- 支援 DXF 檔案檢視和生成
- 自動特徵提取
- 圖形化檔案選擇介面

## 常見問題

### Q: 為什麼 DWG 檔案無法讀取？
A: DWG 檔案需要安裝 ODA File Converter。如果已安裝但無法找到，程式會提示手動選擇路徑。

### Q: 轉換後的 2D 圖只有簡單的邊界框？
A: 這是因為特徵提取是簡化版本。系統會優先使用 CadQuery 直接導出 DXF，這樣可以保留更多細節。

### Q: 如何啟用修改功能？
A: 在 `auto_drafter_system.py` 的 main 區塊中，取消註解修改指令相關的程式碼即可。

### Q: 支援哪些 Python 版本？
A: 建議使用 Python 3.8 或更高版本。已測試 Python 3.13。

### Q: 圖檔資訊視窗顯示「無資訊」怎麼辦？
A: 這表示該項目在檔案中沒有相關資訊。例如：
- STL 檔案不包含元資料，大部分項目會顯示「無資訊」
- STEP 檔案的元資料取決於來源 CAD 軟體是否有寫入
- 這是正常現象，不影響模型轉換功能

### Q: 如何查看完整的圖檔資訊？
A: 執行 `auto_drafter_system.py` 後，會自動彈出圖檔資訊視窗，顯示所有可提取的資訊，包含 BOM、來源軟體、零件列表、單位、幾何統計等。

## 測試

### 執行測試

```bash
# 主要數值驗證測試（67 項）
py test/verify_values.py

# 執行所有 unittest 測試
python -m unittest discover test -v

# 使用 PowerShell 腳本
.\run_tests.ps1

# 監控模式（檔案變更時自動執行）
.\run_tests.ps1 -Watch
```

### 測試涵蓋範圍

- 管路中心線提取（弧形半徑、角度、高低差、弧長）
- 取料明細（直管、彎管、腳架）
- stp_data 組裝參數（管徑、弧半徑、仰角、弦長等）
- 4 張施工圖 DXF 生成與內容驗證
- Drawing 2 尺寸標註、支撐架、BOM 球號
- Drawing 1/3 腳架長度、取料明細標籤

詳細說明請參閱 [測試文件](docs/testing.md)

## 文件

完整文件位於 `docs/` 目錄：
- [快速入門](docs/getting-started.md)
- [使用者手冊](docs/user-guide.md)
- [系統架構](docs/architecture.md)
- [API 參考](docs/api/)
- [測試說明](docs/testing.md)

## 授權

本專案為開源專案，可自由使用和修改。

## 貢獻

歡迎提交 Issue 和 Pull Request！

如有問題或建議，請在 GitHub 上開 Issue。
