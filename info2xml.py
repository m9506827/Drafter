import re
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

class GroupedBOMAutomator:
    """
    資深繪圖師開發：分級 BOM 暨數量自動彙整系統
    功能：
    1. 分開顯示 BOM 1 (1-6) 與 BOM 2 (7-14)。
    2. 增加「數量 (Qty)」欄位：相同名稱與尺寸的零件自動合併。
    3. 保留完整 XML 實體生成功能。
    """
    def __init__(self, input_file):
        self.input_file = input_file
        self.entities = {}
        self.solids_list = []
        self.main_info = {"part_no": "Unknown", "name": "Unknown"}

    def run_pipeline(self):
        print(f"[*] 正在分析: {self.input_file}")
        self._parse_raw_data()
        self._calculate_individual_solids()
        self._export_grouped_reports()
        print(f"[!] 成功！已產出分級 BOM 表與 info_tree.xml。")

    def _parse_raw_data(self):
        content = ""
        for enc in ['utf-8', 'cp950', 'big5']:
            try:
                with open(self.input_file, 'r', encoding=enc) as f:
                    content = f.read()
                    break
            except: continue
        
        content = re.sub(r'【.*?】|---.*', '', content)
        pattern = re.compile(r'#(\d+)\s*=\s*([\w_]+)\s*\((.*?)\)\s*;', re.DOTALL)
        for match in pattern.finditer(content):
            eid, etype, eargs = match.groups()
            self.entities[eid] = {'type': etype, 'args': eargs.strip()}

        # 提取產品號
        for data in self.entities.values():
            if data['type'] == 'PRODUCT':
                attrs = re.findall(r"'(.*?)'", data['args'])
                self.main_info["part_no"] = attrs[0] if attrs else "N/A"
                break

    def _decode_text(self, text):
        def hex_to_char(match):
            try: return bytes.fromhex(match.group(1)).decode('utf-16-be')
            except: return match.group(1)
        return re.sub(r'\\X2\\([0-9A-F]+)\\X0\\', hex_to_char, text).replace("'", "").strip()

    def _calculate_individual_solids(self):
        # 建立點座標快速對照
        coords_map = {}
        for eid, data in self.entities.items():
            if data['type'] == 'CARTESIAN_POINT':
                m = re.search(r'\(([-0-9.E, ]+)\)', data['args'])
                if m: coords_map[eid] = [float(x.strip()) for x in m.group(1).split(',')]

        # 找出所有實體本體
        solids = [(eid, data) for eid, data in self.entities.items() if data['type'] == 'MANIFOLD_SOLID_BREP']
        
        for eid, data in solids:
            name_m = re.search(r"'(.*?)'", data['args'])
            s_name = self._decode_text(name_m.group(1)) if name_m else f"Solid_{eid}"
            
            # 獲取關聯點並計算尺寸
            pt_ids = self._get_points_recursive(eid, set())
            pts = [coords_map[pid] for pid in pt_ids if pid in coords_map]
            
            if pts:
                dims = [max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)]
                size_str = f"{dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}"
            else:
                size_str = "0.00 x 0.00 x 0.00"

            self.solids_list.append({"id": eid, "name": s_name, "size": size_str})

    def _get_points_recursive(self, eid, visited):
        if eid in visited: return set()
        visited.add(eid)
        points = set()
        ent = self.entities.get(eid)
        if not ent: return points
        if ent['type'] == 'CARTESIAN_POINT':
            points.add(eid)
            return points
        refs = re.findall(r'#(\d+)', ent['args'])
        for ref in refs:
            points.update(self._get_points_recursive(ref, visited))
        return points

    def _export_grouped_reports(self):
        # 1. 切分並彙整數據
        bom1_raw = self.solids_list[:6]
        bom2_raw = self.solids_list[6:]

        def aggregate(raw_list):
            agg = {}
            for item in raw_list:
                key = (item['name'], item['size'])
                if key not in agg: agg[key] = {'count': 0, 'ids': []}
                agg[key]['count'] += 1
                agg[key]['ids'].append(f"#{item['id']}")
            return agg

        bom1 = aggregate(bom1_raw)
        bom2 = aggregate(bom2_raw)

        # 2. 生成 TXT 報告
        with open('bom_report.txt', 'w', encoding='utf-8') as f:
            f.write("="*105 + "\n")
            f.write(f"{'分級物料彙整總表 (Grouped BOM Report)':^105}\n")
            f.write("="*105 + "\n")
            f.write(f"主件號: {self.main_info['part_no']} | 來源: {os.path.basename(self.input_file)}\n\n")

            # BOM 1 輸出
            f.write(f"[ BOM 1 - 基礎實體組件 (Part 1-6) ]\n")
            f.write("-" * 105 + "\n")
            f.write(f"{'零件名稱':<25} | {'尺寸 (L x W x H) mm':<35} | {'數量':<6} | {'原始實體 ID'}\n")
            f.write("-" * 105 + "\n")
            for (name, size), info in bom1.items():
                f.write(f"{name:<25} | {size:<35} | {info['count']:<6} | {', '.join(info['ids'])}\n")
            f.write("\n")

            # BOM 2 輸出
            f.write(f"[ BOM 2 - 延伸/鏡射組件 (Part 7-14) ]\n")
            f.write("-" * 105 + "\n")
            f.write(f"{'零件名稱':<25} | {'尺寸 (L x W x H) mm':<35} | {'數量':<6} | {'原始實體 ID'}\n")
            f.write("-" * 105 + "\n")
            for (name, size), info in bom2.items():
                f.write(f"{name:<25} | {size:<35} | {info['count']:<6} | {', '.join(info['ids'])}\n")
            
            f.write("\n" + "="*105 + "\n")

        # 3. 生成 XML
        root = ET.Element("StepData", Project=self.main_info['part_no'])
        ents_node = ET.SubElement(root, "Entities")
        for eid, data in self.entities.items():
            node = ET.SubElement(ents_node, "Entity", id=eid, type=data['type'])
            node.text = data['args']
        
        xml_str = ET.tostring(root, encoding='utf-8')
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
        with open('info_tree.xml', 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

if __name__ == "__main__":
    automator = GroupedBOMAutomator('info.txt')
    automator.run_pipeline()