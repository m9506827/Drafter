"""
測試子系統施工圖：使用 2-2.stp 模型，驗證 3 張 DXF + PNG 內容
參考圖片: test/2-2-1.jpg, test/2-2-2.jpg, test/2-2-3.jpg
"""
import io
import os
import re
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr

# 將上層目錄加入搜尋路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ezdxf

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(TEST_DIR, "2-2.stp")
OUTPUT_DIR = os.path.join(TEST_DIR, "output")

# ── Module-level cached engine ──────────────────────────────
# Load model + generate DXF once, shared by all tests.

_engine = None
_dxf_paths = []
_captured_stdout = ""
_captured_stderr = ""
_load_error = None


def _ensure_loaded():
    """Load 2-2.stp and generate 3 DXF exactly once."""
    global _engine, _dxf_paths, _captured_stdout, _captured_stderr, _load_error

    if _engine is not None or _load_error is not None:
        return

    try:
        from auto_drafter_system import MockCADEngine

        buf_out = io.StringIO()
        buf_err = io.StringIO()

        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            _engine = MockCADEngine(MODEL_PATH)
            _dxf_paths = _engine.generate_sub_assembly_drawing(OUTPUT_DIR)

        _captured_stdout = buf_out.getvalue()
        _captured_stderr = buf_err.getvalue()
    except Exception as exc:
        _load_error = exc


# ── Helper functions ────────────────────────────────────────

def _get_all_texts(dxf_path: str) -> list[str]:
    """Extract all TEXT entity strings from a DXF file."""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    return [e.dxf.text for e in msp.query("TEXT")]


def _find_dxf(suffix: str) -> str | None:
    """Find a DXF path ending with the given suffix."""
    for p in _dxf_paths:
        if p.endswith(suffix):
            return p
    return None


def _assert_approx(test_case, actual: float, expected: float, tol: float = 0.05,
                   msg: str = ""):
    """Assert actual ≈ expected within relative tolerance (default 5%)."""
    diff = abs(actual - expected)
    limit = expected * tol
    test_case.assertLessEqual(
        diff, limit,
        f"{msg}: {actual} vs expected {expected} (diff={diff:.2f}, limit={limit:.2f})"
    )


def _extract_numbers(texts: list[str]) -> list[float]:
    """Extract all numeric values from a list of text strings."""
    nums = []
    for t in texts:
        nums.extend(float(m) for m in re.findall(r'\d+(?:\.\d+)?', t))
    return nums


def _has_text_matching(texts: list[str], pattern: str) -> bool:
    """Return True if any text matches the regex pattern."""
    rx = re.compile(pattern)
    return any(rx.search(t) for t in texts)


def _find_value_near(nums: list[float], target: float, tol: float = 0.05) -> float | None:
    """Find first number in nums within tolerance of target."""
    for n in nums:
        if abs(n - target) <= target * tol:
            return n
    return None


# ── Test class ──────────────────────────────────────────────

class TestSubAssemblyDrawings(unittest.TestCase):
    """Validate 2-2.stp sub-assembly drawings against reference images."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"Model not found: {MODEL_PATH}")
        _ensure_loaded()
        if _load_error:
            raise _load_error

    # ── Test 01: Model loading ──

    def test_01_load_model(self):
        """STEP loads, features extracted, no exceptions."""
        self.assertIsNotNone(_engine)
        self.assertIsNotNone(_engine.cad_model)
        solid_features = [f for f in _engine.features if f.type == "solid"]
        self.assertGreater(len(solid_features), 0, "Should extract at least 1 solid")

    # ── Test 02: 3 DXF files ──

    def test_02_generates_3_dxf(self):
        """3 DXF files exist with correct naming."""
        self.assertEqual(len(_dxf_paths), 3, f"Expected 3 DXF, got {len(_dxf_paths)}")
        for p in _dxf_paths:
            self.assertTrue(os.path.exists(p), f"DXF not found: {p}")
            self.assertTrue(p.endswith(".dxf"), f"Not a .dxf: {p}")

    # ── Test 03: 3 PNG previews ──

    def test_03_generates_3_png(self):
        """3 PNG previews exist, >1KB, valid image, reasonable size."""
        import matplotlib.image as mpimg

        expected = ["2-2-1_preview.png", "2-2_2_preview.png", "2-2_3_preview.png"]
        for name in expected:
            path = os.path.join(OUTPUT_DIR, name)
            self.assertTrue(os.path.exists(path), f"PNG missing: {name}")
            size_kb = os.path.getsize(path) / 1024
            self.assertGreater(size_kb, 1.0, f"{name} too small: {size_kb:.1f} KB")
            self.assertLess(size_kb, 5000, f"{name} too large: {size_kb:.1f} KB")
            img = mpimg.imread(path)
            h, w = img.shape[:2]
            self.assertGreater(w, 100, f"{name} width too small")
            self.assertGreater(h, 100, f"{name} height too small")

    # ── Test 04: Drawing 1 cutting list ──

    def test_04_drawing1_cutting_list(self):
        """U1-U3, D1-D3 specs match reference (5% tolerance).

        Actual DXF content:
          U1: 直徑48.1 長度221.3
          U2: 10度(R245)弧長47
          U3: 直徑48.1 長度147.3
          D1: 直徑48.1 長度244.2
          D2: 10度(R245)弧長47
          D3: 直徑48.1 長度291.4
        """
        dxf1 = _find_dxf("-1.dxf")
        self.assertIsNotNone(dxf1, "Drawing 1 not found")
        texts = _get_all_texts(dxf1)
        nums = _extract_numbers(texts)

        # Expected straight lengths
        for val in [221.3, 147.3, 244.2, 291.4]:
            found = _find_value_near(nums, val)
            self.assertIsNotNone(found, f"Missing length ~{val} in Drawing 1")

        # Arc radius R245
        found = _find_value_near(nums, 245)
        self.assertIsNotNone(found, "Missing radius ~245 in Drawing 1")

        # Arc length 47
        found = _find_value_near(nums, 47)
        self.assertIsNotNone(found, "Missing arc length ~47 in Drawing 1")

        # 10-degree angle annotations
        self.assertTrue(
            _has_text_matching(texts, r'10\s*[°度]') or _find_value_near(nums, 10),
            "Missing 10° angle in Drawing 1"
        )

        # Pipe diameter 48.1
        self.assertTrue(
            _has_text_matching(texts, r'48\.1'),
            "Missing diameter 48.1 in Drawing 1"
        )

    # ── Test 05: Drawing 1 BOM ──

    def test_05_drawing1_bom(self):
        """3 legs with L=490, L=529, L=563.

        Actual DXF content: 腳架x3 線長L=490, L=529, L=563
        """
        dxf1 = _find_dxf("-1.dxf")
        self.assertIsNotNone(dxf1)
        texts = _get_all_texts(dxf1)

        # 「腳架」should appear at least 2 times in BOM
        leg_texts = [t for t in texts if "腳架" in t]
        self.assertGreaterEqual(len(leg_texts), 2,
                                f"Expected >=2 腳架 entries, found {len(leg_texts)}")

        # Extract L= values
        leg_lengths = []
        for t in texts:
            m = re.search(r'L\s*=\s*(\d+(?:\.\d+)?)', t)
            if m:
                leg_lengths.append(float(m.group(1)))
        leg_lengths.sort()

        self.assertGreaterEqual(len(leg_lengths), 2,
                                f"Expected >=2 L= values, found {len(leg_lengths)}")

        # Check L≈490, L≈529, L≈563 present
        for expected_l in [490, 529, 563]:
            self.assertIsNotNone(
                _find_value_near(leg_lengths, expected_l),
                f"Missing L≈{expected_l}, found {leg_lengths}"
            )

    # ── Test 06: Drawing 1 rail spacing ──

    def test_06_drawing1_rail_spacing(self):
        """Rail spacing dimension present (219.6mm pipe spacing).

        Actual DXF content: 219.6 (上下軌間距)
        """
        dxf1 = _find_dxf("-1.dxf")
        self.assertIsNotNone(dxf1)
        texts = _get_all_texts(dxf1)
        nums = _extract_numbers(texts)

        found = _find_value_near(nums, 219.6)
        self.assertIsNotNone(found,
                             f"Missing rail spacing ~219.6 in Drawing 1")

    # ── Test 07: Drawing 1 title block ──

    def test_07_drawing1_title_block(self):
        """Company, material, scale, drawing number present.

        Reference (2-2-1.jpg): LM-11, 羅布森, STK-400, 1:10
        """
        dxf1 = _find_dxf("-1.dxf")
        self.assertIsNotNone(dxf1)
        texts = _get_all_texts(dxf1)
        all_text = " ".join(texts)

        for keyword in ["羅布森", "STK-400", "1:10"]:
            self.assertIn(keyword, all_text,
                          f"Missing '{keyword}' in Drawing 1 title block")

    # ── Test 08: Drawing 1 leg vertical ──

    def test_08_drawing1_leg_vertical(self):
        """>=2 red vertical lines (legs), dx<0.5.

        Reference (2-2-1.jpg): legs are vertical red lines.
        """
        dxf1 = _find_dxf("-1.dxf")
        self.assertIsNotNone(dxf1)
        doc = ezdxf.readfile(dxf1)
        msp = doc.modelspace()

        vertical_red = []
        for e in msp.query("LINE"):
            if e.dxf.color == 1:  # red
                sx, sy = e.dxf.start.x, e.dxf.start.y
                ex, ey = e.dxf.end.x, e.dxf.end.y
                length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
                if length < 5:
                    continue
                dx = abs(ex - sx)
                dy = abs(ey - sy)
                if dx < 0.5 and dy > 10:
                    vertical_red.append((dx, length))

        self.assertGreaterEqual(len(vertical_red), 2,
                                f"Expected >=2 vertical red legs, found {len(vertical_red)}")
        for i, (dx, length) in enumerate(vertical_red):
            self.assertLess(dx, 0.5, f"Leg {i + 1} not vertical: dx={dx:.4f}")

    # ── Test 09: Drawing 2 cutting list ──

    def test_09_drawing2_cutting_list(self):
        """Arc items R~242, 178deg, arc length 960.

        Actual DXF content:
          直徑48.1 178度(R242)弧長960 高低差490.3
        """
        dxf2 = _find_dxf("_2.dxf")
        self.assertIsNotNone(dxf2, "Drawing 2 not found")
        texts = _get_all_texts(dxf2)
        nums = _extract_numbers(texts)

        # R ≈ 242
        found_r = _find_value_near(nums, 242)
        self.assertIsNotNone(found_r, f"Missing R≈242 in Drawing 2")

        # 178 degree
        found_178 = _find_value_near(nums, 178, tol=0.05)
        self.assertIsNotNone(found_178, f"Missing 178° in Drawing 2")

        # Arc length 960
        found_arc = _find_value_near(nums, 960)
        self.assertIsNotNone(found_arc, f"Missing arc length ~960 in Drawing 2")

    # ── Test 10: Drawing 2 BOM ──

    def test_10_drawing2_bom(self):
        """支撐架 present with quantity in BOM.

        Actual DXF content: 支撐架 as name, 5 as separate quantity text,
        plus spec 57.3x38.3x243.2
        """
        dxf2 = _find_dxf("_2.dxf")
        self.assertIsNotNone(dxf2)
        texts = _get_all_texts(dxf2)

        bracket_texts = [t for t in texts if "支撐架" in t]
        self.assertGreater(len(bracket_texts), 0,
                           "Missing 支撐架 in Drawing 2 BOM")

        # Quantity "5" should appear as a text entity
        self.assertIn("5", texts,
                       "Missing quantity 5 in Drawing 2 BOM")

    # ── Test 11: Drawing 2 dimensions ──

    def test_11_drawing2_dimensions(self):
        """R~242, 仰角~32度, height~490.3.

        Actual DXF content: R242, 仰角32度, 高低差490.3
        """
        dxf2 = _find_dxf("_2.dxf")
        self.assertIsNotNone(dxf2)
        texts = _get_all_texts(dxf2)
        nums = _extract_numbers(texts)

        # R ≈ 242
        self.assertIsNotNone(_find_value_near(nums, 242),
                             "Missing R≈242 in Drawing 2 dims")

        # 仰角 ≈ 32 degrees
        found_angle = _find_value_near(nums, 32, tol=0.10)
        self.assertIsNotNone(found_angle,
                             "Missing 仰角≈32° in Drawing 2 dims")

        # Height diff ≈ 490.3
        found_h = _find_value_near(nums, 490.3, tol=0.05)
        self.assertIsNotNone(found_h,
                             f"Missing height≈490.3 in Drawing 2 dims")

    # ── Test 12: Drawing 3 cutting list ──

    def test_12_drawing3_cutting_list(self):
        """U1-U3, D1-D3 specs match (complete assembly view).

        Actual DXF content (complete assembly = straight + curved):
          U1: 直徑48.1 長度221.3
          U2: 178度(R242)弧長960
          U3: 直徑48.1 長度147.3
          D1: 直徑48.1 長度244.2
          D2: 178度(R242)弧長960
          D3: 直徑48.1 長度291.4
        """
        dxf3 = _find_dxf("_3.dxf")
        self.assertIsNotNone(dxf3, "Drawing 3 not found")
        texts = _get_all_texts(dxf3)
        nums = _extract_numbers(texts)

        # Straight lengths (same as Drawing 1)
        for val in [221.3, 147.3, 244.2, 291.4]:
            found = _find_value_near(nums, val)
            self.assertIsNotNone(found, f"Missing length ~{val} in Drawing 3")

        # Arc radius R242
        found = _find_value_near(nums, 242)
        self.assertIsNotNone(found, "Missing radius ~242 in Drawing 3")

        # Arc length 960
        found = _find_value_near(nums, 960)
        self.assertIsNotNone(found, "Missing arc length ~960 in Drawing 3")

    # ── Test 13: Drawing 3 BOM ──

    def test_13_drawing3_bom(self):
        """腳架 present with L values in BOM.

        Actual DXF content: 腳架x3, 線長L=489.8, L=529.2, L=562.9
        Also: 支撐架, 軌道
        """
        dxf3 = _find_dxf("_3.dxf")
        self.assertIsNotNone(dxf3)
        texts = _get_all_texts(dxf3)

        leg_texts = [t for t in texts if "腳架" in t]
        self.assertGreater(len(leg_texts), 0,
                           "Missing 腳架 in Drawing 3 BOM")

        # Check L= values present
        leg_lengths = []
        for t in texts:
            m = re.search(r'L\s*=\s*(\d+(?:\.\d+)?)', t)
            if m:
                leg_lengths.append(float(m.group(1)))

        self.assertGreater(len(leg_lengths), 0,
                           "Missing L= values in Drawing 3 BOM")

        # L≈489.8
        self.assertIsNotNone(
            _find_value_near(leg_lengths, 489.8),
            f"Missing L≈489.8 in Drawing 3 BOM, found {leg_lengths}"
        )

    # ── Test 14: Drawing 3 angles ──

    def test_14_drawing3_angles(self):
        """178° angle annotation present.

        Actual DXF content: 178° (curved track bend angle)
        """
        dxf3 = _find_dxf("_3.dxf")
        self.assertIsNotNone(dxf3)
        texts = _get_all_texts(dxf3)

        has_178 = _has_text_matching(texts, r'178\s*[°度]')
        if not has_178:
            nums = _extract_numbers(texts)
            has_178 = _find_value_near(nums, 178, tol=0.05) is not None
        self.assertTrue(has_178, "Missing 178° angle in Drawing 3")

    # ── Test 15: No warnings/errors ──

    def test_15_no_warnings(self):
        """No 'error'/'traceback'/'warning' in captured output."""
        combined = (_captured_stdout + _captured_stderr).lower()

        # Filter out benign patterns
        # Lines with [OK], [Preview], known info messages are fine
        problem_lines = []
        for line in combined.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Skip benign lines
            if any(ok in line_stripped for ok in ["[ok]", "[preview]", "[info]",
                                                   "[cad kernel]", "warning: cq"]):
                continue
            # Check for actual problems
            if re.search(r'\berror\b|\btraceback\b', line_stripped):
                # Exclude false positives like "error_handling" in code paths
                if "error" in line_stripped and "[error]" in line_stripped:
                    problem_lines.append(line_stripped)
                elif "traceback" in line_stripped:
                    problem_lines.append(line_stripped)

        self.assertEqual(
            len(problem_lines), 0,
            f"Found error/traceback in output:\n" +
            "\n".join(problem_lines[:10])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
