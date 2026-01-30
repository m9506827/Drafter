import matplotlib
matplotlib.use('Agg')  # Must be before pyplot import

import sys
import os
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_drafter_system import MockCADEngine
from simple_viewer import EngineeringViewer

STEP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "1-2.stp")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference")


def test_projections():
    """Generate all projections and save as PNG"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(STEP_FILE):
        print(f"[Error] STEP file not found: {STEP_FILE}")
        sys.exit(1)

    print(f"[Test] Loading STEP file: {STEP_FILE}")
    cad = MockCADEngine(STEP_FILE)

    print(f"[Test] Exporting projections to DXF: {OUTPUT_DIR}")
    dxf_files = cad.export_projections_to_dxf(OUTPUT_DIR)

    if not dxf_files:
        print("[Error] No DXF files generated")
        sys.exit(1)

    print(f"[Test] Generated {len(dxf_files)} DXF files")

    png_files = []
    for dxf in dxf_files:
        png = dxf.replace('.dxf', '.png')
        EngineeringViewer.view_2d_dxf(dxf, fast_mode=True, save_path=png)
        png_files.append(png)

    print(f"[Test] Saved {len(png_files)} PNG files")
    return png_files


def compare_with_reference(png_files):
    """Compare output PNGs against reference images"""
    if not os.path.exists(REFERENCE_DIR):
        print("[Skip] No reference directory, run with --save-reference first")
        return

    ref_files = [f for f in os.listdir(REFERENCE_DIR) if f.endswith('.png')]
    if not ref_files:
        print("[Skip] No reference images found, run with --save-reference first")
        return

    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.io import imread
        HAS_SKIMAGE = True
    except ImportError:
        HAS_SKIMAGE = False

    results = []
    for png in png_files:
        basename = os.path.basename(png)
        ref = os.path.join(REFERENCE_DIR, basename)
        if not os.path.exists(ref):
            results.append((basename, "SKIP", "no reference"))
            continue

        if HAS_SKIMAGE:
            img1 = imread(png, as_gray=True)
            img2 = imread(ref, as_gray=True)
            # Resize if dimensions differ
            if img1.shape != img2.shape:
                from skimage.transform import resize
                img2 = resize(img2, img1.shape, anti_aliasing=True)
            score = ssim(img1, img2)
            status = "PASS" if score > 0.85 else "FAIL"
            results.append((basename, status, f"SSIM={score:.3f}"))
        else:
            # Fallback: file size comparison
            s1 = os.path.getsize(png)
            s2 = os.path.getsize(ref)
            ratio = min(s1, s2) / max(s1, s2) if max(s1, s2) > 0 else 0
            status = "PASS" if ratio > 0.8 else "WARN"
            results.append((basename, status, f"size ratio={ratio:.2f}"))

    # Print summary
    print("\n" + "=" * 50)
    print("Projection Test Results")
    print("=" * 50)
    for name, status, detail in results:
        print(f"  [{status}] {name} ({detail})")

    failures = [r for r in results if r[1] == "FAIL"]
    if failures:
        print(f"\n  {len(failures)} FAILED")
    else:
        print(f"\n  All {len(results)} checks passed")

    return results


def save_as_reference(png_files):
    """Save current outputs as reference images"""
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    for png in png_files:
        dst = os.path.join(REFERENCE_DIR, os.path.basename(png))
        shutil.copy2(png, dst)
        print(f"[Saved] {dst}")


if __name__ == "__main__":
    png_files = test_projections()

    if "--save-reference" in sys.argv:
        save_as_reference(png_files)
    else:
        compare_with_reference(png_files)
