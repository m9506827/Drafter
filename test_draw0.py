"""Quick test: generate Drawing 0/1/2/3 from test\2-2.stp"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Suppress Tk GUI
os.environ['DISPLAY'] = ''

from auto_drafter_system import MockCADEngine, log_print

stp_path = os.path.join(os.path.dirname(__file__), "test", "2-2.stp")
print(f"Loading: {stp_path}")
engine = MockCADEngine(model_file=stp_path)
print("Generating sub-assembly drawings...")
results = engine.generate_sub_assembly_drawing("output")
print(f"Done! Generated {len(results)} drawings:")
for r in results:
    print(f"  {r}")
