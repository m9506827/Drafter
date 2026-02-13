# 測試說明

## 概述

AutoDrafter 使用 Python `unittest` 框架進行自動化測試，確保系統功能正確。

## 測試目錄結構

```
test/
├── test_components.py    # 主要單元測試（15 項）
├── test_projection.py    # 投影測試
├── analyze_dxf.py        # DXF 分析工具
├── 2-2.stp              # 測試用 STEP 檔案
├── 2-2-1.jpg            # 參考圖 (Drawing 1)
├── 2-2-2.jpg            # 參考圖 (Drawing 2)
├── 2-2-3.jpg            # 參考圖 (Drawing 3)
├── output/              # 測試輸出
└── reference/           # 參考圖片
```

## 執行測試

### 數值驗證測試（推薦）

```bash
py test/verify_values.py
```

此腳本涵蓋 67 項檢查，包含管路中心線、取料明細、stp_data 參數、
4 張施工圖 DXF 生成與內容驗證。所有檢查通過會顯示 `ALL PASSED: 67/67`。

### 執行所有 unittest 測試

```bash
python -m unittest discover test -v
```

### 執行特定測試檔案

```bash
python -m unittest test.test_components -v
```

### 執行單一測試

```bash
python -m unittest test.test_components.TestSubAssemblyDrawings.test_04_drawing1_cutting_list -v
```

## 測試項目

### test_components.py（15 項測試）

#### 模型載入
| # | 測試名稱 | 說明 |
|---|---------|------|
| 01 | `test_01_load_model` | STEP 載入成功，提取特徵 |

#### 檔案生成
| # | 測試名稱 | 說明 |
|---|---------|------|
| 02 | `test_02_generates_3_dxf` | 生成 4 個 DXF 檔案 |
| 03 | `test_03_generates_3_png` | 生成 PNG 預覽（>1KB, <5MB） |

#### Drawing 1 驗證
| # | 測試名稱 | 說明 |
|---|---------|------|
| 04 | `test_04_drawing1_cutting_list` | 取料明細（U1-U3, D1-D3） |
| 05 | `test_05_drawing1_bom` | BOM（腳架 L=490, 529, 563） |
| 06 | `test_06_drawing1_rail_spacing` | 軌道間距（219.6mm） |
| 07 | `test_07_drawing1_title_block` | 標題欄（羅布森、STK-400） |
| 08 | `test_08_drawing1_leg_vertical` | 腳架垂直紅線 ≥2 條 |

#### Drawing 2 驗證
| # | 測試名稱 | 說明 |
|---|---------|------|
| 09 | `test_09_drawing2_cutting_list` | 取料明細（R242、178度、弧長960） |
| 10 | `test_10_drawing2_bom` | BOM（支撐架 x5） |
| 11 | `test_11_drawing2_dimensions` | 尺寸（R242、仰角32度、高低差490.3） |

#### Drawing 3 驗證
| # | 測試名稱 | 說明 |
|---|---------|------|
| 12 | `test_12_drawing3_cutting_list` | 完整取料明細 |
| 13 | `test_13_drawing3_bom` | BOM（腳架 L≈489.8） |
| 14 | `test_14_drawing3_angles` | 角度標註（178°） |

#### 輸出檢查
| # | 測試名稱 | 說明 |
|---|---------|------|
| 15 | `test_15_no_warnings` | 無 error/traceback 輸出 |

### test_projection.py

| 測試 | 說明 |
|-----|------|
| `test_projections()` | 生成 XY, XZ, YZ 投影 |
| `compare_with_reference()` | 與參考圖比對（SSIM > 0.85） |

## 測試資料

### 測試模型
- `test/2-2.stp` - 標準測試用 STEP 檔案
- `test/1-2.stp` - 額外測試模型

### 參考圖片
- `test/2-2-1.jpg` - Drawing 1 參考
- `test/2-2-2.jpg` - Drawing 2 參考
- `test/2-2-3.jpg` - Drawing 3 參考

## 測試輔助函數

### _get_all_texts
從 DXF 檔案提取所有文字。

```python
def _get_all_texts(dxf_path: str) -> list[str]
```

### _assert_approx
驗證數值在容差範圍內。

```python
def _assert_approx(test_case, actual, expected, tol=0.05, msg="")
```

### _find_value_near
在數值列表中查找接近目標的值。

```python
def _find_value_near(nums: list[float], target: float, tol=0.05) -> float | None
```

### _has_text_matching
檢查是否有文字匹配正則表達式。

```python
def _has_text_matching(texts: list[str], pattern: str) -> bool
```

## 容差設定

| 項目 | 容差 |
|-----|------|
| 長度 | ±5% |
| 角度 | ±5% (Drawing 2 仰角 ±10%) |
| 半徑 | ±5% |
| SSIM 相似度 | > 0.85 |

## 新增測試

### 步驟

1. 在 `test/test_components.py` 中新增測試方法
2. 方法名稱以 `test_` 開頭
3. 使用 `self.assert*` 系列方法驗證

### 範例

```python
def test_16_drawing1_new_feature(self):
    """驗證 Drawing 1 的新功能"""
    dxf1 = _find_dxf("-1.dxf")
    self.assertIsNotNone(dxf1)
    texts = _get_all_texts(dxf1)
    
    # 驗證新功能
    self.assertTrue(
        _has_text_matching(texts, r'新功能'),
        "Missing 新功能 in Drawing 1"
    )
```

## 持續整合

### 本地執行

```bash
# 執行所有測試
python -m unittest discover test -v

# 檢查覆蓋率（需安裝 coverage）
coverage run -m unittest discover test
coverage report
```

### GitHub Actions

可在 `.github/workflows/test.yml` 設定 CI：

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirement.txt
      - run: python -m unittest discover test -v
```

## 故障排除

### 測試失敗

1. 檢查測試模型是否存在
2. 檢查輸出目錄權限
3. 查看錯誤訊息中的實際值與預期值

### 圖形比對失敗

1. 確認參考圖片存在
2. 檢查 SSIM 分數
3. 視覺比對生成的圖片

## 另請參閱

- [使用者手冊](user-guide.md)
- [系統架構](architecture.md)
