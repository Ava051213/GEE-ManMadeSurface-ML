# -*- coding: utf-8 -*-
"""
风机检测数据预处理脚本
从全球风机数据中提取中国数据，生成正负样本
"""

import pandas as pd
import numpy as np

# ========== 第1步：读取数据 ==========
print("正在读取全球风机数据...")
df = pd.read_csv('00_raw_data/global_wind_2020.csv')

print(f"读取完成！全球共有 {len(df)} 个风机记录")
print("数据列名: {columns}".format(columns=df.columns.tolist()))
print("\n前5行数据预览:")
print(df.head())

# ========== 第2步：查看国家分布 ==========
print("\n风机数量最多的10个国家:")
country_counts = df['GID_0'].value_counts().head(10)
print(country_counts)

# ========== 第3步：筛选中国数据 ==========
print("\n正在筛选中国数据...")
china_df = df[df['GID_0'] == 'CHN'].copy()

print(f"✅ 中国风机数量: {len(china_df)}")
print("📍 坐标范围:")
print(f"   X: {china_df['X'].min():.2f} ~ {china_df['X'].max():.2f}")
print(f"   Y: {china_df['Y'].min():.2f} ~ {china_df['Y'].max():.2f}")

# ========== 第4步：坐标转换（Eckert IV → WGS84） ==========
print("\n🔄 正在进行坐标转换...")

try:
    from pyproj import Transformer
    
    # Eckert IV投影的Proj4字符串
    eckert_iv_proj4 = '+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    
    # 创建坐标转换器
    transformer = Transformer.from_crs(eckert_iv_proj4, "EPSG:4326", always_xy=True)
    
    # 转换坐标
    lon, lat = transformer.transform(np.array(china_df['X']), np.array(china_df['Y']))
    
    china_df['longitude'] = lon
    china_df['latitude'] = lat
    
    print("✅ 坐标转换完成！")
    print(f"   经度范围: {lon.min():.2f}° ~ {lon.max():.2f}°")
    print(f"   纬度范围: {lat.min():.2f}° ~ {lat.max():.2f}°")
    
except Exception as e:
    print(f"⚠️ 坐标转换出错: {e}")
    print("📝 尝试备用方法...")
    # 如果已经是经纬度，直接使用
    china_df['longitude'] = china_df['X']
    china_df['latitude'] = china_df['Y']

# ========== 第5步：生成正样本（风机位置） ==========
print("\n➕ 正在生成正样本...")

# 为每个风机生成边界框
buffer = 0.0013  # 约0.0013度 = 约145米（半边）
bbox_size = 0.0026  # 完整边界框约0.0026度 = 约289米

positive_samples = []

for idx, row in china_df.iterrows():
    center_x = row['longitude']
    center_y = row['latitude']
    
    sample = {
        'wind_id': row['wind_id'],
        'center_x': center_x,
        'center_y': center_y,
        'xmin': center_x - buffer,
        'ymin': center_y - buffer,
        'xmax': center_x + buffer,
        'ymax': center_y + buffer,
        'turbines': row['turbines'],
        'class': 1,  # 正样本标记为1
        'label': 'wind_turbine'
    }
    positive_samples.append(sample)

print(f"✅ 正样本数量: {len(positive_samples)}")

# ========== 第6步：生成负样本（非风机位置） ==========
print("\n➖ 正在生成负样本...")

# 设置随机种子，保证可重复
np.random.seed(42)

# 中国边界范围
china_bounds = {
    'xmin': 73.5,   # 最西端
    'xmax': 135.0,  # 最东端
    'ymin': 18.0,   # 最南端
    'ymax': 53.5    # 最北端
}

# 获取正样本的中心点
positive_points = np.array([[s['center_x'], s['center_y']] for s in positive_samples])

# 目标负样本数量（正样本的2倍）
n_negative = len(positive_samples) * 2
print(f"🎯 目标负样本数量: {n_negative}")

negative_samples = []
attempts = 0
max_attempts = n_negative * 10  # 最多尝试10倍
min_distance = 0.01  # 与正样本的最小距离（约1.1公里）

while len(negative_samples) < n_negative and attempts < max_attempts:
    attempts += 1
    
    # 在中国边界内随机生成点
    x = np.random.uniform(china_bounds['xmin'], china_bounds['xmax'])
    y = np.random.uniform(china_bounds['ymin'], china_bounds['ymax'])
    
    # 计算与所有正样本的距离
    distances = np.sqrt((positive_points[:, 0] - x)**2 + (positive_points[:, 1] - y)**2)
    
    # 如果距离足够远，接受为负样本
    if np.min(distances) > min_distance:
        sample = {
            'wind_id': f'neg_{len(negative_samples)}',
            'center_x': x,
            'center_y': y,
            'xmin': x - buffer,
            'ymin': y - buffer,
            'xmax': x + buffer,
            'ymax': y + buffer,
            'turbines': 0,
            'class': 0,  # 负样本标记为0
            'label': 'non_turbine'
        }
        negative_samples.append(sample)
    
    # 每1000次尝试显示一次进度
    if attempts % 1000 == 0:
        print(f"   进度: {len(negative_samples)}/{n_negative} (尝试{attempts}次)", end='\r')

print(f"\n✅ 实际生成负样本数量: {len(negative_samples)}")
print(f"   总共尝试了 {attempts} 次")

# ========== 第7步：合并正负样本 ==========
print("\n🔄 正在合并正负样本...")

all_samples = positive_samples + negative_samples
all_df = pd.DataFrame(all_samples)

print(f"✅ 总样本数量: {len(all_df)}")
print(f"   正样本: {len(all_df[all_df['class'] == 1])}")
print(f"   负样本: {len(all_df[all_df['class'] == 0])}")

def classify_regions(df):
    """根据经纬度划分四个区域"""
    regions = []
    for _, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        if 110 <= lon <= 120 and 35 <= lat <= 45:
            regions.append('north_china')
        elif 115 <= lon <= 125 and 25 <= lat <= 35:
            regions.append('east_china')
        elif 95 <= lon <= 110 and 25 <= lat <= 35:
            regions.append('southwest_china')
        elif 75 <= lon <= 100 and 35 <= lat <= 45:
            regions.append('northwest_china')
        else:
            regions.append('other')
    df['region'] = regions
    return df

# ========== 第8步：区域划分 ==========
print("\n🗺️  正在进行区域划分...")

# 在生成样本后添加区域分类
all_df = classify_regions(all_df)

# 统计各区域样本数量
regions = ['north_china', 'east_china', 'southwest_china', 'northwest_china', 'other']
for region in regions:
    count = len(all_df[all_df['region'] == region])
    print(f"   {region}: {count}")

# ========== 第9步：保存结果 ==========
print("\n💾 正在保存结果...")

import os
os.makedirs('processed_data', exist_ok=True)

# 保存各区域数据
regions = ['north_china', 'east_china', 'southwest_china', 'northwest_china']
for region in regions:
    region_df = all_df[all_df['region'] == region]
    if not region_df.empty:
        region_df.to_csv(f'processed_data/{region}_samples.csv', index=False)
        print(f"✅ {region} CSV已保存")
        
        # 保存为GeoJSON
        try:
            import geopandas as gpd
            from shapely.geometry import box
            
            # 创建几何图形（边界框）
            geometries = []
            for _, row in region_df.iterrows():
                geom = box(float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax']))
                geometries.append(geom)
            
            # 创建GeoDataFrame
            region_gdf = gpd.GeoDataFrame(region_df, geometry=geometries, crs='EPSG:4326')
            region_gdf.to_file(f'processed_data/{region}_samples.geojson', driver='GeoJSON')
            print(f"✅ {region} GeoJSON已保存")
            
        except ImportError:
            print("⚠️  未安装geopandas，跳过GeoJSON导出")
            print("   如需GeoJSON，请运行: pip install geopandas shapely")

# 也保存完整的数据集
all_df.to_csv('processed_data/all_samples.csv', index=False)
print("✅ 完整CSV已保存: processed_data/all_samples.csv")

# 保存完整的GeoJSON（用于GEE）
try:
    import geopandas as gpd
    from shapely.geometry import box
    
    # 创建几何图形（边界框）
    geometries = []
    for _, row in all_df.iterrows():
        geom = box(float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax']))
        geometries.append(geom)
    
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(all_df, geometry=geometries, crs='EPSG:4326')
    
    # 保存为GeoJSON
    geojson_path = 'processed_data/all_samples.geojson'
    gdf.to_file(geojson_path, driver='GeoJSON')
    print("✅ 完整GeoJSON已保存: processed_data/all_samples.geojson")
    
    # 分别保存正负样本
    positive_gdf = gdf[gdf['class'] == 1]
    negative_gdf = gdf[gdf['class'] == 0]
    
    positive_gdf.to_file('processed_data/positive_samples.geojson', driver='GeoJSON')
    negative_gdf.to_file('processed_data/negative_samples.geojson', driver='GeoJSON')
    print("✅ 正负样本GeoJSON已分别保存")
    
except ImportError:
    print("⚠️  未安装geopandas，跳过完整GeoJSON导出")

# ========== 第10步：生成统计报告 ==========
print("\n📊 数据预处理完成！统计报告:")
print("=" * 50)
print(f"总样本数: {len(all_df)}")
print(f"  - 正样本（风机）: {len(all_df[all_df['class'] == 1])}")
print(f"  - 负样本: {len(all_df[all_df['class'] == 0])}")
print("\n区域分布:")
for region in ['north_china', 'east_china', 'southwest_china', 'northwest_china', 'other']:
    count = len(all_df[all_df['region'] == region])
    print(f"  - {region}: {count}")
print(f"\n边界框大小: {bbox_size}° x {bbox_size}° (约289m x 289m)")
print("坐标系统: WGS84 (EPSG:4326)")
print("=" * 50)

print("\n🎉 数据预处理完成！")
print("\n下一步:")
print("1. 将 processed_data/positive_samples.geojson 上传到GEE")
print("2. 将 processed_data/negative_samples.geojson 上传到GEE")
print("3. 修改GEE代码中的Asset ID")
