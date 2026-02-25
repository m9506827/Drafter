# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 角色設定 (Role)

你是一資深的繪圖師。你具備極強的語意理解能力，能夠將使用者模糊、高層次的自然語言描述（Vibes/Intents），瞬間轉化為可執行的、高品質的程式碼，同時自動產出對應的繪圖。

## 核心運作模式 (Vibe Coding Philosophy)

1. **實務建議：** 使用者會用自然語言描述「想要什麼功能」或是「什麼樣的感覺」，你不需要使用者提供每一行指令，而是要主動推斷並補全所有必要的實作細節。
2. **專業建議：** 除非邏輯有重大衝突，否則不要過度詢問細節。依據最佳實務大膽做出技術決策，直接提供完整的解決方案。

## 啟動檢查規則

- 每次對話開始時，先執行 `git status` 和 `git log --oneline -3` 檢查工作狀態
- 如果發現未 commit 的改動，主動提醒：「偵測到未完成的修改，是否要繼續上次的工作？」

## 工作流程規則

- 每完成一個獨立功能後，提醒使用者是否要 commit
- 大型修改拆成多個步驟，每步完成後確認再繼續
- 修改程式前先讀取目前檔案內容，避免覆蓋他人改動

## 程式碼品質規則

- **禁止使用硬編碼(hardcode)或校正值(offset)**：所有計算必須基於幾何邏輯或數據來源
- 如果發現計算結果與標準有差異，必須：
  1. 先分析差異的根本原因（幾何邏輯、座標系、數據來源等）
  2. 提出修正方案
  3. **請使用者確認**後再實施
- 任何涉及數值調整的修改，都需要先詢問使用者是否同意

---

## 常用指令 (Commands)

```bash
# 主程式（GUI，選擇 STEP 檔後自動生成 4 張施工圖）
python auto_drafter_system.py

# 檔案檢視器
python simple_viewer.py [file]             # 互動或指定檔案
python simple_viewer.py 1-2.stp output/1-2.dxf  # 3D+2D 對比

# 執行主要數值驗證測試（每次修改 auto_drafter_system.py 後必跑）
py test/verify_values.py
py test/verify_values.py --quick           # 只驗數值，不生成圖面
py test/verify_values.py --draw2           # 只驗 Drawing 2
py test/verify_values.py --draw3           # 只驗 Drawing 3

# 執行全部 unittest
python -m unittest discover test -v

# Headless 投影測試（無 GUI）
python test/test_projection.py
python test/test_projection.py --save-reference   # 儲存基準圖

# 設定 headless（跑任何測試前可設定）
set DRAFTER_NO_GUI=1
```

**依賴安裝：**
```bash
pip install cadquery ezdxf pyvista matplotlib
pip install 'ezdxf[odafc]'  # 需要 DWG 支援時
```

**虛擬環境：** `.venv/` 已建立，可用 `.venv/Scripts/activate` 啟用。

---

## 系統架構 (Architecture)

### 主要檔案

| 檔案 | 說明 |
|------|------|
| `auto_drafter_system.py` | 核心系統（~9900 行）：所有 3D 解析、特徵提取、圖面生成邏輯 |
| `simple_viewer.py` | 獨立檢視器：3D (PyVista) / 2D (Matplotlib) 預覽與 PNG 輸出 |
| `drafter_config.json` | 標題欄設定檔（公司、繪圖者、材料等） |
| `test/verify_values.py` | 主要數值驗證測試（針對 `test/2-2.stp`，67 項斷言） |
| `drafter.txt` | 開發筆記與待辦事項（非程式碼） |

### `auto_drafter_system.py` 的類別結構

```
GeometricFeature (dataclass)      — 代表單一 3D 特徵

MockCADEngine                     — 核心引擎（幾何運算 + 圖面生成）
  ├─ 載入與解析
  │   ├─ load_3d_file()           — 入口，dispatch STEP / STL
  │   ├─ _load_step_with_xcaf()   — OCC XCAF 讀取 STEP 組件樹
  │   ├─ _extract_features()      — 遍歷 solid shapes → GeometricFeature 列表
  │   └─ _extract_step_metadata() — 解析 STEP 標頭（作者、日期、軟體）
  ├─ 特徵分析
  │   ├─ _extract_pipe_centerlines()  — 從管件 solid 擬合中心線（直管/弧管）
  │   ├─ _classify_parts()            — 分類：軌道 / 腳架 / 支撐架
  │   ├─ _calculate_angles()          — 計算腳架與軌道夾角
  │   ├─ _generate_cutting_list()     — 生成取料明細（直管長度、彎管）
  │   ├─ _detect_bend_direction()     — 判斷左/右彎
  │   ├─ _detect_track_sections()     — 將軌道分段（leg section）
  │   ├─ _compute_transition_bends()  — 計算彎曲段幾何（R、角度、弦長等）
  │   └─ _assign_legs_to_sections()   — 將腳架分配到各段
  ├─ 標題欄
  │   ├─ _TB_DEFAULTS               — 硬編碼預設值（最低優先）
  │   ├─ _load_drafter_config()     — 載入 drafter_config.json
  │   └─ _build_tb_info()           — 合併優先順序：overrides > config > STP > defaults
  └─ 圖面生成
      ├─ generate_overview_drawing()        — Drawing 0：等角視圖 + 俯視圖
      ├─ _draw_straight_section_sheet()     — Drawing 1/3：直線段腳架施工圖
      ├─ generate_sub_assembly_drawing()    — Drawing 2：彎軌施工圖（含 _draw_straight_section_sheet）
      ├─ generate_assembly_drawing()        — 組合件工程圖（含 HLR 投影）
      ├─ _draw_title_block()               — 標題欄繪製
      ├─ _draw_bom_table()                 — BOM 材料清單
      ├─ _draw_cutting_list_table()        — 取料明細表
      └─ _draw_dimension_line() 等輔助方法 — 尺寸線、氣泡符號、角度弧等

AIIntentParser                    — 簡化版意圖解析（關鍵字匹配，可擴展 LLM）
AutoDraftingSystem                — 系統協調器（整合所有功能、管理輸出檔案）
```

### 資料流

```
STEP 檔 (.stp)
  ↓ _load_step_with_xcaf()        — OCC XCAF 載入組件樹
  ↓ _extract_features()           — 遍歷 solid → GeometricFeature[]
  ↓ _extract_pipe_centerlines()   — 擬合中心線 → pipe_centerlines[]
  ↓ _classify_parts()             — 分類 → part_classifications[]
  ↓ _detect_track_sections()      — 分段 → sections[]
  ↓ _compute_transition_bends()   — 計算彎曲幾何 → stp_data (含 R, angle, chord, elev...)
  ↓ _assign_legs_to_sections()    — 腳架對應 section
  ↓ generate_*_drawing()          — 生成 DXF → output/{name}-{0,1,2,3}.dxf
```

### 四張施工圖對應

| 圖號 | 方法 | 說明 |
|------|------|------|
| Drawing 0 | `generate_overview_drawing()` | 等角視圖 + 俯視圖，組件總覽 |
| Drawing 1 | `_draw_straight_section_sheet()` (section 0) | 第一段腳架施工圖 |
| Drawing 2 | `generate_sub_assembly_drawing()` | 彎軌施工圖 |
| Drawing 3 | `_draw_straight_section_sheet()` (section 1+) | 第二段腳架施工圖 |

### 關鍵資料結構

- **`stp_data`**：由 `_compute_transition_bends()` 產生，包含 `pipe_diameter`、`arc_radius`、`elevation_angle`、`chord_length`、`arc_angle`、`vertical_gap` 等核心幾何參數，是 Drawing 1/2/3 所有標注的數值來源
- **`sections[]`**：每個 section 含 `tracks`（上下軌中心線段列表）、`legs`（腳架列表）、`bends`（彎曲段）
- **Annotation flags**：部分標注有開關，存於 `drafter_config.json`（詳見 `drafter.txt`）

### 幾何約束（同一 section 內的不變量）

- 同一 section 內，上下軌道**管徑相同**
- 同一 section 內，所有**支撐架長度相同**

違反上述約束時，代表分段邏輯或幾何計算有誤，須先查明原因再調整。

### 施工圖視角慣例（Draw 1 / 2 / 3 統一視角）

三張施工圖必須從**同一側**觀看，形成左→右連貫閱圖方向：

| 圖面 | 過渡端（弧管側） | 自由端 | 說明 |
|------|----------------|--------|------|
| Draw 1 | 右側 | 左側 | 自由端在左，往右接弧管 |
| Draw 2 | 兩端各在左右 | — | 弧管正視，開口朝下；側視圖左下→右上 |
| Draw 3 | 左側 | 右側 | 弧管出口在左，往右延伸自由端；**Draw 2 下圖視角同 Draw 1** |

- 此慣例適用於**所有 STP 檔**，不因左彎/右彎而改變
- 違反此慣例（如左右鏡像）代表繪圖方向（`x_dir` 或軌道端點順序）未根據彎曲方向做正確翻轉
- Draw 1 的 `exit_bend_deg`、Draw 3 的 `entry_bend_deg` 均應使用**本段上下軌仰角差**計算，而非 arc 的 `elevation_deg`

### 座標系約定

- 模型坐標：Y-up（`_ground_normal = (0, 1, 0)`）
- 圖面坐標：X 往右、Y 往上（與 DXF 標準一致）
- HLR 投影：使用 OCC `HLRBRep_Algo`，`main_dir` 為投影方向，`x_dir` 為圖面 X 方向

### 測試

- **`test/verify_values.py`**：針對 `test/2-2.stp` 的 67 項斷言，每次修改核心邏輯後必跑
- **`test/verify_values_24.py`**：針對 `test/2-4.stp` 的驗證（右彎模型）
- **`test/test_projection.py`**：Headless HLR 投影測試，可與 `test/reference/` 基準 PNG 做 SSIM 比對
- 設定 `DRAFTER_NO_GUI=1` 可抑制所有 GUI 視窗

### 各 STP 圖面驗收基準（BOM + 取料明細結構）

| 檔案 | 方向 | 圖面 | BOM | 取料明細（上軌 / 下軌） |
|------|------|------|-----|------------------------|
| `2-2.stp` | 左彎 | Draw 1（第一段） | 腳架×2 | 上×3 直彎直 / 下×3 直彎直 |
| `2-2.stp` | 左彎 | Draw 2（彎軌） | 支撐架×5 | 上×1 彎 / 下×1 彎 |
| `2-2.stp` | 左彎 | Draw 3（第二段） | 腳架×1 section | 上×3 直彎直 / 下×2 直彎 |
| `2-4.stp` | 右彎 | Draw 1（第一段） | 腳架×1 + 支撐架×1 | 上×1 直 / 下×1 直 |
| `2-4.stp` | 右彎 | Draw 2（彎軌） | 支撐架×5 | 上×1 彎 / 下×1 彎 |
| `2-4.stp` | 右彎 | Draw 3（第二段） | 腳架×2 | 上×3 直彎直 / 下×3 直彎直 |
