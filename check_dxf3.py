"""分析 Drawing 3 的 DXF 內容"""
import ezdxf
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

doc = ezdxf.readfile("output/2-2_3.dxf")
msp = doc.modelspace()

print("=== Drawing 3 內容分析 ===\n")

texts = []
for e in msp:
    if e.dxftype() == 'MTEXT':
        texts.append(e.text)
    elif e.dxftype() == 'TEXT':
        texts.append(e.dxf.text)

print("軌道取料明細:")
for t in texts:
    t_stripped = t.strip()
    if '直徑' in t_stripped:
        print(f"  {t_stripped}")

ball_numbers = []
for t in texts:
    matches = re.findall(r'[UD]\d+', t)
    ball_numbers.extend(matches)

ball_numbers = sorted(set(ball_numbers))
print(f"\n球號列表: {ball_numbers}")
