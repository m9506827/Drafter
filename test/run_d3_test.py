"""Quick test: generate Drawing 3 with annotations disabled (only seg length)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['DRAFTER_NO_GUI'] = '1'

from auto_drafter_system import MockCADEngine

stp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2-2.stp')
eng = MockCADEngine(model_file=stp_path)
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_d3_test')
os.makedirs(out_dir, exist_ok=True)
results = eng.generate_sub_assembly_drawing(out_dir)
if results:
    print(f"Generated {len(results)} files:")
    for r in results:
        print(f"  {r}")
else:
    print("No files generated!")
print("Done!")
