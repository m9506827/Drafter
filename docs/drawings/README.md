# 繪圖說明

本目錄說明 AutoDrafter 系統生成的 4 種施工圖。

## 圖面概覽

| 圖號 | 名稱 | 用途 |
|-----|------|------|
| Drawing 0 | 組件總覽圖 | 等角視圖 + 俯視圖，含尺寸標註 |
| Drawing 1 | 直線段施工圖 | 直軌段 + 腳架詳圖 |
| Drawing 2 | 彎軌施工圖 | 彎管段詳圖 |
| Drawing 3 | 完整組合施工圖 | 全系統組合圖 |

## 目錄

- [Drawing 0: 組件總覽圖](drawing0-overview.md)
- [Drawing 1: 直線段施工圖](drawing1-straight.md)
- [Drawing 2: 彎軌施工圖](drawing2-curved.md)
- [Drawing 3: 完整組合施工圖](drawing3-assembly.md)

## 共通規格

### 圖紙
- 尺寸：A3 橫向（420 x 297 mm）
- 邊界：10mm
- 比例：1:10

### 標題欄

標題欄由 `_build_tb_info()` 統一建構，欄位來源依序為：
overrides > `drafter_config.json` > STEP 中繼資料 > `_TB_DEFAULTS`

| 欄位 | 來源 |
|------|------|
| 公司名稱 | config > 預設 |
| 專案名稱 | config > STP product_name > 檔名 |
| 圖面名稱 | 各圖面固定值 |
| 繪圖者 | config > 預設 |
| 日期 | config > STP creation_date > 系統日期 |
| 單位 | config > STP > 預設 (mm) |
| 比例 | config > 預設 (1:10) |
| 材料 | config > 預設 (STK-400) |
| 表面處理 | config > 預設 (裁切及焊接) |
| 圖號 | 各圖面固定值 |
| 版次 | 程式版本 `__version__`（不可被 config 覆蓋） |
| 數量 | config > 預設 (1) |

### 取料明細表
- 球號（U1, U2... / D1, D2...）
- 取料尺寸（管徑、長度/角度/半徑/弧長）
- 備註

### BOM 表
- 序號
- 品名
- 數量
- 規格
