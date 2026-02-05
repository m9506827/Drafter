# Drafter - AI 自動繪圖系統

一個智能的 3D 到 2D 工程圖自動轉換系統，支援 STEP/STL 檔案轉換為 DXF 工程圖。

## 功能特色

- **3D 模型檢視**：支援 STL、STEP/STP 格式的 3D 模型預覽（使用 PyVista）
- **2D 工程圖檢視**：支援 DXF、DWG 格式的 2D 工程圖檢視（使用 Matplotlib）
- **六方向投影**：自動生成 XY/XZ/YZ 三平面的直接投影與反向投影（共 6 張 DXF）
- **完整邊緣提取**：使用 OCP TopExp_Explorer 遍歷複合體所有 solid 的邊，支援多實體組合件
- **自動轉換**：將 3D 模型自動轉換為 2D 工程圖（使用 CadQuery + OCP）
- **智能特徵提取**：自動從 3D 模型中提取幾何特徵（圓形、矩形、多段線等）
- **Headless 測試**：支援無 GUI 的自動投影測試，可輸出 PNG 並與基準圖比對（SSIM）
- **圖檔資訊視窗**：獨立視窗完整顯示圖檔資訊，包含 BOM、來源軟體、零件、單位等
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
2. **開啟圖檔資訊視窗**，完整顯示：
   - 基本檔案資訊（路徑、大小、格式）
   - 來源軟體與元資料（作者、組織、建立日期）
   - 單位資訊
   - 幾何統計（實體數、面數、邊數、體積、表面積）
   - 邊界框資訊
   - 零件列表
   - BOM 材料清單
   - 幾何特徵列表
   - 若無特定資訊則顯示「無資訊」
3. 關閉資訊視窗或按 Enter 繼續
4. 自動將 3D 模型轉換為 2D 工程圖
5. 自動儲存到 `output/原檔名.dxf`（例如：`output/1-2.dxf`）
6. 自動顯示 2D 工程圖視窗（Matplotlib）

**注意：** 目前版本直接轉換，不修改原圖內容。修改功能已註解，可根據需要啟用。

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

## 專案結構

```
Drafter/
├── auto_drafter_system.py       # 主要系統：3D 到 2D 自動轉換
│   ├── MockCADEngine            #   CAD 核心（模型載入、特徵提取、投影）
│   ├── AIIntentParser           #   AI 意圖解析器
│   └── AutoDraftingSystem       #   系統協調器
├── simple_viewer.py             # 檔案檢視器：3D/2D 檢視 + PNG 輸出
│   └── EngineeringViewer        #   工程檔案檢視器類別
├── generate_assembly_drawing.py # 組合件工程圖生成器
├── info2xml.py                  # STEP 資訊轉 XML 工具
├── test/                        # 自動化測試
│   ├── test_projection.py       #   Headless 投影測試腳本
│   ├── output/                  #   測試輸出 PNG（自動生成）
│   └── reference/               #   基準 PNG（用於 SSIM 比對）
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
   - 特徵提取（從 3D 模型提取幾何特徵）
   - 3D 到 2D 投影：使用 OCP `TopExp_Explorer` 遍歷所有 solid 邊緣，投影到 XY/XZ/YZ 平面
   - 支援多實體複合件（Compound Shape），確保所有零件邊緣都被投影
   - **圖檔資訊提取**：從 STEP 檔案提取元資料（來源軟體、作者、單位等）

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
CadQuery 載入
    ↓
直接導出為 DXF（或提取特徵後生成 DXF）
    ↓
儲存到 output/目錄
    ↓
顯示 2D 工程圖
```

### 擴展建議

- **整合 LLM API**：擴展 `AIIntentParser` 整合 OpenAI/Claude API
- **改進特徵提取**：增強 `_extract_features()` 提取更複雜的幾何特徵
- **尺寸標註**：自動添加尺寸標註和技術要求
- **圖層管理**：支援 DXF 圖層和線型設定

## 更新日誌

### v1.3 (最新)
- **六方向投影**：XY/XZ/YZ 直接投影 + 反向投影，共 6 張 DXF
- **完整邊緣提取**：使用 OCP `TopExp_Explorer` 遍歷 Compound 中所有 Solid（修正多實體組合件邊緣遺漏問題）
- **Headless 測試**：`test/test_projection.py` 支援無 GUI 自動測試 + SSIM 基準比對
- **PNG 輸出**：`view_2d_dxf()` 和 `view_projected_2d()` 支援 `save_path` 參數直接輸出 PNG

### v1.2
- 圖檔資訊視窗：獨立視窗完整顯示圖檔資訊
- BOM 材料清單：自動生成並顯示
- 來源軟體識別、單位資訊、元資料提取
- 幾何統計：實體數、面數、邊數、頂點數、體積、表面積
- 零件列表、視窗可捲動

### v1.1
- ✅ 支援 DWG 檔案讀取（需要 ODA File Converter）
- ✅ 同時顯示 3D 和 2D 視圖功能
- ✅ 改進的顏色處理（保留原始顏色，白色自動轉黑色）
- ✅ 自動儲存到 output 目錄
- ✅ 改進的錯誤處理和調試訊息
- ✅ 直接使用 CadQuery 導出 DXF（更快更可靠）

### v1.0
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
# 執行所有測試
python -m unittest discover test -v

# 使用 PowerShell 腳本
.\run_tests.ps1

# 監控模式（檔案變更時自動執行）
.\run_tests.ps1 -Watch

# 快速測試
.\run_tests.ps1 -Quick
```

### 自動測試設定

**GitHub Actions**：推送到 GitHub 後自動執行測試（已設定）

**本地 pre-commit hook**：
```bash
git config core.hooksPath .githooks
```

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
