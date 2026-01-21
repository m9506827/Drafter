# Drafter - AI 自動繪圖系統

一個智能的 3D 到 2D 工程圖自動轉換系統，支援 STEP/STL 檔案轉換為 DXF 工程圖。

## 功能特色

- 🎨 **3D 模型檢視**：支援 STL、STEP/STP 格式的 3D 模型預覽
- 📐 **2D 工程圖檢視**：支援 DXF、DWG 格式的 2D 工程圖檢視
- 🔄 **自動轉換**：將 3D 模型自動轉換為 2D 工程圖
- 🎯 **智能特徵提取**：自動從 3D 模型中提取幾何特徵
- 📁 **批量處理**：支援批量檔案處理

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

### 1. 自動繪圖系統

```bash
python auto_drafter_system.py
```

執行後會：
1. 選擇 3D 模型檔案（STEP/STP）
2. 顯示 3D 模型預覽
3. 自動轉換為 2D 工程圖
4. 儲存到 `output/原檔名.dxf`
5. 顯示 2D 工程圖

### 2. 檔案檢視器

```bash
# 查看單一檔案
python simple_viewer.py

# 查看指定檔案
python simple_viewer.py 1-2.stp
python simple_viewer.py output/1-2.dxf

# 同時查看 3D 和 2D
python simple_viewer.py 1-2.stp output/1-2.dxf
```

### 3. 生成測試模型

```bash
python generate_test_flange.py
```

## 專案結構

```
Drafter/
├── auto_drafter_system.py    # 主要系統：3D 到 2D 自動轉換
├── simple_viewer.py          # 檔案檢視器：支援 3D/2D 檢視
├── generate_test_flange.py   # 測試模型生成器
├── test_stp_viewer.py        # STEP 檔案測試工具
├── output/                   # 輸出目錄（自動生成）
└── README.md                 # 本文件
```

## 支援的檔案格式

### 3D 格式
- **STEP/STP**：使用 CadQuery 讀取
- **STL**：使用 PyVista 讀取

### 2D 格式
- **DXF**：使用 ezdxf 讀取/寫入
- **DWG**：使用 ezdxf + ODA File Converter 讀取

## 開發說明

### 核心模組

- `MockCADEngine`：模擬 CAD 核心，負責 3D 模型載入和特徵提取
- `AIIntentParser`：AI 意圖解析器（目前為簡化版本）
- `AutoDraftingSystem`：系統協調器，整合所有功能
- `EngineeringViewer`：工程檔案檢視器

### 擴展功能

- 可以擴展 `AIIntentParser` 整合真實的 LLM API
- 可以改進 `_extract_features()` 提取更複雜的幾何特徵
- 可以添加更多 2D 視圖（側視圖、前視圖等）

## 授權

本專案為開源專案，可自由使用和修改。

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 更新日誌

### v1.0
- 基本 3D 到 2D 轉換功能
- 支援 STEP/STL 檔案讀取
- 支援 DXF/DWG 檔案檢視
- 自動特徵提取
