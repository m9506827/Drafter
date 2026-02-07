# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('output/2-2_info.txt', encoding='utf-8') as f:
    content = f.read()

checks = [
    ('半徑/R=', 'R='),
    ('長度', '長度='),
    ('角度', '角度='),
    ('螺旋', '螺旋'),
    ('高低差', '高低差'),
    ('仰角', '仰角'),
    ('弧長', '弧長'),
    ('外弧長', '外弧長'),
    ('右螺旋', '右螺旋'),
    ('左螺旋', '左螺旋'),
]
print('=== 關鍵字搜尋 ===')
for label, kw in checks:
    count = content.count(kw)
    print(f'  {label}: {"有" if count > 0 else "【無】"} ({count} 次)')

# Show pipe centerline arc entries
print('\n=== 弧線管路資料 ===')
lines = content.split('\n')
for i, line in enumerate(lines):
    if '弧線' in line:
        print(f'  Line {i+1}: {line.strip()}')

# Show cutting list
print('\n=== 取料明細 ===')
in_cl = False
for line in lines:
    if '軌道取料明細' in line:
        in_cl = True
    if in_cl:
        print(line)
    if in_cl and '幾何特徵' in line:
        break
