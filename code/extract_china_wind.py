"""
extract_china_wind.py
用途：从 global_wind_2020.csv 中提取中国(CID_0=CHN)的风机数据
      将坐标从 Eckert IV (ESRI:54012) 转换为 WGS84 (EPSG:4326)
      输出 CSV 文件，用于上传到 GEE Asset

输入：data/raw/global_wind_2020.csv (Eckert IV 投影)
输出：data/samples/china_wind_turbines_wgs84.csv (WGS84 经纬度)
"""

import csv
from pathlib import Path
from pyproj import Transformer

# === 配置 ===
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / 'data' / 'raw' / 'global_wind_2020.csv'
OUTPUT_FILE = PROJECT_DIR / 'data' / 'samples' / 'china_wind_turbines_wgs84.csv'
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 坐标转换器：Eckert IV → WGS84
transformer = Transformer.from_crs('ESRI:54012', 'EPSG:4326', always_xy=True)

# === 筛选 + 转换 ===
chn_rows = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['GID_0'] == 'CHN':
            x = float(row['X'])
            y = float(row['Y'])
            lon, lat = transformer.transform(x, y)
            # 添加 WGS84 坐标，移除原始 Eckert IV 坐标
            row['longitude'] = round(lon, 6)
            row['latitude'] = round(lat, 6)
            row.pop('X', None)
            row.pop('Y', None)
            chn_rows.append(row)

# === 输出 ===
fieldnames = ['longitude', 'latitude', 'wind_id', 'GID_0', 'turbines', 'landscape.area', 'water', 'urban', 'power']
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(chn_rows)

# === 统计 ===
total_turbines = sum(int(r['turbines']) for r in chn_rows if r['turbines'].isdigit())
lons = [float(r['longitude']) for r in chn_rows]
lats = [float(r['latitude']) for r in chn_rows]

print(f'完成！')
print(f'输出文件: {OUTPUT_FILE}')
print(f'中国风机记录数: {len(chn_rows)}')
print(f'中国风机总数: {total_turbines}')
print(f'经度范围: {min(lons):.2f}°E ~ {max(lons):.2f}°E')
print(f'纬度范围: {min(lats):.2f}°N ~ {max(lats):.2f}°N')
print(f'')
print(f'GEE Asset 上传命令:')
print(f'  进入 GEE Code Editor → Assets → NEW → Table upload')
print(f'  选择 {OUTPUT_FILE}')
print(f'  Asset ID: users/你的用户名/china_wind_turbine_samples')
print(f'')
print(f'或者使用 CLI:')
print(f'  earthengine upload table --asset_id users/你的用户名/china_wind_turbine_samples {OUTPUT_FILE}')
