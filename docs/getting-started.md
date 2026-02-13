# 快速入門

本指南將協助您快速開始使用 AutoDrafter 自動化工程製圖系統。

## 系統需求

- Python 3.10+
- Windows 10/11（已測試）
- 約 2GB 磁碟空間（含依賴套件）

## 安裝步驟

### 1. 複製專案

```bash
git clone https://github.com/your-repo/Drafter-deepseek.git
cd Drafter-deepseek
```

### 2. 建立虛擬環境

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

### 3. 安裝依賴

```bash
pip install -r requirement.txt
```

主要依賴套件：
- `cadquery-ocp` - OpenCASCADE Python 綁定
- `ezdxf` - DXF 檔案讀寫
- `matplotlib` - 圖形繪製
- `numpy` - 數值計算

## 第一次執行

### 使用圖形介面

```bash
python simple_viewer.py test\2-2.stp
```

這將開啟 3D 檢視器，顯示 STEP 模型。

### 生成施工圖

```bash
python simple_viewer.py test\2-2.stp --dxf
```

這將在 `output/` 目錄生成：
- `2-2-0.dxf` - 組件總覽圖（等角 + 俯視）
- `2-2-1.dxf` - 直線段施工圖
- `2-2_2.dxf` - 彎軌施工圖
- `2-2_3.dxf` - 完整組合施工圖
- 對應的 PNG 預覽圖

### 無 GUI 模式

```bash
python simple_viewer.py test\2-2.stp --dxf --no-gui
```

適用於伺服器或批次處理。

## 輸出檔案說明

| 檔案 | 說明 |
|-----|------|
| `*-0.dxf` | Drawing 0: 組件總覽圖（等角 + 俯視） |
| `*-1.dxf` | Drawing 1: 直線段施工圖（腳架詳圖） |
| `*_2.dxf` | Drawing 2: 彎軌施工圖 |
| `*_3.dxf` | Drawing 3: 完整組合施工圖 |
| `*_preview.png` | 各圖的 PNG 預覽 |
| `*_info.txt` | 模型資訊文字檔 |

## 設定檔（可選）

若需自訂標題欄資訊（公司名稱、繪圖者等），可在專案根目錄建立 `drafter_config.json`：

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

> 版次 (`version`) 由程式版本自動帶入，無需設定。

詳見 [使用者手冊](user-guide.md) 的「設定檔」章節。

## 下一步

- [使用者手冊](user-guide.md) - 詳細功能說明
- [系統架構](architecture.md) - 了解系統設計
- [API 參考](api/MockCADEngine.md) - 程式介面說明
