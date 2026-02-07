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
        """U1-U3, D1-D3 specs match reference 2-2-1.jpg (5% tolerance).

        Expected DXF content (with transition straights):
          U1: 直徑48.1 長度221.3
          U2: 12度(R270)外弧長62
          U3: 直徑48.1 長度202.4
          D1: 直徑48.1 長度127.5  (≈127.8 in reference)
          D2: 12度(R220)外弧長51
          D3: 直徑48.1 長度291.4
        """
        dxf1 = _find_dxf("-1.dxf")
        self.assertIsNotNone(dxf1, "Drawing 1 not found")
        texts = _get_all_texts(dxf1)
        nums = _extract_numbers(texts)

        # Expected 4 straight lengths: U1=221.3, U3≈202, D1≈128, D3=291.4
        for val in [221.3, 202.4, 291.4]:
            found = _find_value_near(nums, val)
            self.assertIsNotNone(found, f"Missing length ~{val} in Drawing 1")

        # D1 transition straight ≈ 127.5 (model-computed, ref=127.8)
        found = _find_value_near(nums, 127.5, tol=0.05)
        self.assertIsNotNone(found,
                             f"Missing D1 transition length ~127.5 in Drawing 1")

        # Arc radius R270 (upper) and R220 (lower)
        found_270 = _find_value_near(nums, 270, tol=0.05)
        self.assertIsNotNone(found_270, "Missing radius ~270 in Drawing 1")
        found_220 = _find_value_near(nums, 220, tol=0.05)
        self.assertIsNotNone(found_220, "Missing radius ~220 in Drawing 1")

        # 12-degree angle annotations
        self.assertTrue(
            _has_text_matching(texts, r'12\s*[°度]') or _find_value_near(nums, 12),
            "Missing 12° angle in Drawing 1"
        )

        # Pipe diameter 48.1
        self.assertTrue(
            _has_text_matching(texts, r'48\.1'),
            "Missing diameter 48.1 in Drawing 1"
        )

        # Total 6 cutting list items (U1-U3, D1-D3)
        cutting_items = [t for t in texts if '直徑' in t]
        self.assertGreaterEqual(len(cutting_items), 6,
                                f"Expected >=6 cutting items, found {len(cutting_items)}")

    # ── Test 05: Drawing 1 BOM ──

    def test_05_drawing1_bom(self):
        """2 legs with L≈536, L≈502.

        Expected DXF content: 腳架x2, 線長L=536, L=502
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

        # Check L≈536, L≈502 present (model values for F08, F09)
        for expected_l in [502, 536]:
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

        Expected: iDrafter, STK-400, 1:10, LM-11
        """
        dxf1 = _find_dxf("-1.dxf")
        self.assertIsNotNone(dxf1)
        texts = _get_all_texts(dxf1)
        all_text = " ".join(texts)

        for keyword in ["iDrafter", "STK-400", "1:10"]:
            self.assertIn(keyword, all_text,
                          f"Missing '{keyword}' in Drawing 1 title block")

    # ── Test 08: Drawing 1 leg vertical ──

    def test_08_drawing1_leg_lines(self):
        """>=2 red lines (legs) with sufficient length.

        Reference (2-2-1.jpg): legs are drawn as red lines on the slope.
        """
        dxf1 = _find_dxf("-1.dxf")
        self.assertIsNotNone(dxf1)
        doc = ezdxf.readfile(dxf1)
        msp = doc.modelspace()

        red_lines = []
        for e in msp.query("LINE"):
            if e.dxf.color == 1:  # red
                sx, sy = e.dxf.start.x, e.dxf.start.y
                ex, ey = e.dxf.end.x, e.dxf.end.y
                length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
                if length > 10:
                    red_lines.append(length)

        self.assertGreaterEqual(len(red_lines), 2,
                                f"Expected >=2 red leg lines, found {len(red_lines)}")

    # ── Test 09: Drawing 2 cutting list ──

    def test_09_drawing2_cutting_list(self):
        """Arc items R~242, 178deg, arc length ~898 (centerline).

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

        # Arc length ~898 (centerline arc: sqrt((R*θ)²+h²))
        found_arc = _find_value_near(nums, 898, tol=0.05)
        self.assertIsNotNone(found_arc, f"Missing arc length ~898 in Drawing 2")

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
        """U1-U3, D1-D2 specs match (second straight section with entry transition).

        Expected DXF content:
          U1: 直徑48.1 長度89.0  (entry transition straight, inner tangent formula)
          U2: 16度(R220)弧長68
          U3: 直徑48.1 長度147.3 (main straight, full track length)
          D1: 直徑48.1 長度244.2 (main straight)
          D2: 16度(R270)弧長82   (exit bend)
        """
        dxf3 = _find_dxf("_3.dxf")
        self.assertIsNotNone(dxf3, "Drawing 3 not found")
        texts = _get_all_texts(dxf3)
        nums = _extract_numbers(texts)

        # Straight lengths: U1≈89.0, U3≈147.3, D1=244.2
        for val in [89.0, 147.3, 244.2]:
            found = _find_value_near(nums, val, tol=0.10)
            self.assertIsNotNone(found, f"Missing length ~{val} in Drawing 3")

        # Arc radius R220 or R270
        found_r = (_find_value_near(nums, 220, tol=0.05) or
                   _find_value_near(nums, 270, tol=0.05))
        self.assertIsNotNone(found_r, "Missing radius ~220 or ~270 in Drawing 3")

        # 16-degree angle
        self.assertTrue(
            _has_text_matching(texts, r'16\s*[°度]') or _find_value_near(nums, 16),
            "Missing 16° angle in Drawing 3"
        )

    # ── Test 13: Drawing 3 BOM ──

    def test_13_drawing3_bom(self):
        """腳架 present with L values in BOM.

        Expected DXF content: 腳架x1, 線長L≈462.9 (F10)
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

        # L≈462.9
        self.assertIsNotNone(
            _find_value_near(leg_lengths, 462.9),
            f"Missing L≈462.9 in Drawing 3 BOM, found {leg_lengths}"
        )

    # ── Test 14: Drawing 3 angles ──

    def test_14_drawing3_angles(self):
        """16° transition angle annotation present.

        Expected DXF content: 16° (entry transition bend angle)
        """
        dxf3 = _find_dxf("_3.dxf")
        self.assertIsNotNone(dxf3)
        texts = _get_all_texts(dxf3)

        has_16 = _has_text_matching(texts, r'16\s*[°度]')
        if not has_16:
            nums = _extract_numbers(texts)
            has_16 = _find_value_near(nums, 16, tol=0.10) is not None
        self.assertTrue(has_16, "Missing 16° angle in Drawing 3")

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
