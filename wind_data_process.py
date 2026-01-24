import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from pyproj import Transformer
import json
import os

# ===== 坐标转换函数 =====
def convert_coordinates(df):
    """
    将投影坐标转换为经纬度坐标
    从Eckert IV投影转换到 WGS84 (EPSG:4326)
    """
    print("正在进行坐标转换...")
    
    # Eckert IV投影的Proj4字符串
    eckert_iv_proj4 = '+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    
    try:
        # 创建坐标转换器: 从Eckert IV到WGS84
        transformer = Transformer.from_crs(eckert_iv_proj4, "EPSG:4326", always_xy=True)
        
        # 转换坐标
        lon, lat = transformer.transform(df['X'].values, df['Y'].values)
        
        df['longitude'] = lon
        df['latitude'] = lat
        
        print(f"坐标转换完成！")
        print(f"经度范围: {lon.min():.2f} 至 {lon.max():.2f}")
        print(f"纬度范围: {lat.min():.2f} 至 {lat.max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"坐标转换出错: {e}")
        print("尝试使用备用方法...")
        
        # 如果坐标看起来已经是经纬度，直接使用
        if df['X'].abs().max() < 180 and df['Y'].abs().max() < 90:
            print("检测到坐标可能已经是经纬度格式")
            df['longitude'] = df['X']
            df['latitude'] = df['Y']
        else:
            # 使用ESRI的Eckert IV定义
            try:
                transformer = Transformer.from_crs("ESRI:54012", "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(df['X'].values, df['Y'].values)
                df['longitude'] = lon
                df['latitude'] = lat
                print("使用ESRI:54012转换成功")
            except:
                raise ValueError("无法进行坐标转换，请检查坐标系统")
        
        return df

# ===== 第一步：读取和筛选中国风机数据 =====
def load_china_wind_data(csv_path):
    """读取风机数据并筛选中国区域"""
    print("正在读取风机数据...")
    df = pd.read_csv(csv_path)
    
    print(f"全球风机总数: {len(df)}")
    print(f"数据列名: {df.columns.tolist()}")
    
    # 筛选中国数据
    china_wind = df[df['GID_0'] == 'CHN'].copy()
    print(f"中国风机总数: {len(china_wind)}")
    
    if len(china_wind) == 0:
        print("警告: 未找到中国风机数据！")
        print(f"可用的国家代码: {df['GID_0'].unique()}")
        return None
    
    # 转换坐标
    china_wind = convert_coordinates(china_wind)
    
    return china_wind

# ===== 第二步：为每个风机生成边界框 =====
def generate_bounding_boxes(wind_df, buffer=0.0013):
    """
    为每个风机生成0.0026x0.0026度的边界框
    buffer = 0.0013度 (半径)
    """
    print("正在生成风机边界框...")
    
    bboxes = []
    for idx, row in wind_df.iterrows():
        center_x = row['longitude']  # 经度
        center_y = row['latitude']    # 纬度
        
        # 生成边界框 [xmin, ymin, xmax, ymax]
        bbox = {
            'wind_id': row['wind_id'],
            'center_x': center_x,
            'center_y': center_y,
            'xmin': center_x - buffer,
            'ymin': center_y - buffer,
            'xmax': center_x + buffer,
            'ymax': center_y + buffer,
            'geometry': box(center_x - buffer, center_y - buffer, 
                          center_x + buffer, center_y + buffer),
            'turbines': row['turbines'],
            'class': 1,  # 正样本标签
            'label': 'wind_turbine'
        }
        bboxes.append(bbox)
    
    # 转换为GeoDataFrame
    gdf = gpd.GeoDataFrame(bboxes, crs='EPSG:4326')
    print(f"生成边界框数量: {len(gdf)}")
    
    return gdf

# ===== 第三步：根据纬度划分南北方 =====
def classify_north_south(gdf, latitude_threshold=33.0):
    """
    根据纬度划分南北方
    纬度阈值默认33度（秦岭-淮河线）
    北方: 4月1日-5月31日（无雪期）
    南方: 1月1日-2月28日
    """
    gdf['region'] = gdf['center_y'].apply(
        lambda y: 'north' if y >= latitude_threshold else 'south'
    )
    
    # 设置时间范围
    gdf['date_start'] = gdf['region'].apply(
        lambda r: '2020-04-01' if r == 'north' else '2020-01-01'
    )
    gdf['date_end'] = gdf['region'].apply(
        lambda r: '2020-05-31' if r == 'north' else '2020-02-28'
    )
    
    north_count = len(gdf[gdf['region'] == 'north'])
    south_count = len(gdf[gdf['region'] == 'south'])
    
    print(f"北方风机数量: {north_count} (4-5月)")
    print(f"南方风机数量: {south_count} (1-2月)")
    
    return gdf

# ===== 第四步：生成负样本 =====
def generate_negative_samples(positive_gdf, n_samples=20000, 
                             china_bounds=None, min_distance=0.01):
    """
    在中国区域随机生成负样本
    min_distance: 与正样本的最小距离（度）
    """
    print(f"\n正在生成{n_samples}个负样本...")
    
    # 中国边界
    if china_bounds is None:
        china_bounds = {
            'xmin': 73.5,   # 中国最西端
            'xmax': 135.0,  # 中国最东端
            'ymin': 18.0,   # 中国最南端
            'ymax': 53.5    # 中国最北端
        }
    
    # 获取正样本的中心点
    positive_points = positive_gdf[['center_x', 'center_y']].values
    
    negative_samples = []
    attempts = 0
    max_attempts = n_samples * 10
    
    print("生成进度: ", end='')
    progress_step = n_samples // 10
    
    while len(negative_samples) < n_samples and attempts < max_attempts:
        attempts += 1
        
        # 显示进度
        if len(negative_samples) % progress_step == 0 and len(negative_samples) > 0:
            print(f"{len(negative_samples)}...", end='', flush=True)
        
        # 随机生成点
        x = np.random.uniform(china_bounds['xmin'], china_bounds['xmax'])
        y = np.random.uniform(china_bounds['ymin'], china_bounds['ymax'])
        
        # 检查与所有正样本的距离
        distances = np.sqrt((positive_points[:, 0] - x)**2 + 
                          (positive_points[:, 1] - y)**2)
        
        # 如果距离所有正样本都足够远，则接受
        if np.min(distances) > min_distance:
            buffer = 0.0013
            region = 'north' if y >= 33.0 else 'south'
            
            bbox = {
                'wind_id': f'neg_{len(negative_samples)}',
                'center_x': x,
                'center_y': y,
                'xmin': x - buffer,
                'ymin': y - buffer,
                'xmax': x + buffer,
                'ymax': y + buffer,
                'geometry': box(x - buffer, y - buffer, x + buffer, y + buffer),
                'turbines': 0,
                'class': 0,  # 负样本标签
                'label': 'non_turbine',
                'region': region,
                'date_start': '2020-04-01' if region == 'north' else '2020-01-01',
                'date_end': '2020-05-31' if region == 'north' else '2020-02-28'
            }
            negative_samples.append(bbox)
    
    print(f" 完成!")
    
    if len(negative_samples) < n_samples:
        print(f"警告: 只生成了{len(negative_samples)}个负样本（尝试{attempts}次）")
    
    neg_gdf = gpd.GeoDataFrame(negative_samples, crs='EPSG:4326')
    print(f"实际生成负样本数量: {len(neg_gdf)}")
    
    return neg_gdf

# ===== 第五步：合并正负样本并导出 =====
def combine_and_export(positive_gdf, negative_gdf, output_dir='processed_data'):
    """合并正负样本并导出为多种格式"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并数据
    all_samples = pd.concat([positive_gdf, negative_gdf], ignore_index=True)
    
    print(f"\n" + "="*60)
    print(f"数据汇总:")
    print(f"="*60)
    print(f"总样本数: {len(all_samples)}")
    print(f"  - 正样本（风机）: {len(positive_gdf)}")
    print(f"  - 负样本: {len(negative_gdf)}")
    print(f"\n区域分布:")
    print(f"  - 北方样本: {len(all_samples[all_samples['region'] == 'north'])} (2020-04-01 至 2020-05-31)")
    print(f"  - 南方样本: {len(all_samples[all_samples['region'] == 'south'])} (2020-01-01 至 2020-02-28)")
    
    # 导出为CSV（不含geometry列）
    csv_columns = ['wind_id', 'center_x', 'center_y', 'xmin', 'ymin', 'xmax', 'ymax',
                   'class', 'label', 'region', 'date_start', 'date_end', 'turbines']
    csv_path = os.path.join(output_dir, 'all_samples.csv')
    all_samples[csv_columns].to_csv(csv_path, index=False)
    print(f"\n已保存: {csv_path}")
    
    # 导出为GeoJSON（用于GEE）
    geojson_path = os.path.join(output_dir, 'all_samples.geojson')
    all_samples.to_file(geojson_path, driver='GeoJSON')
    print(f"已保存: {geojson_path}")
    
    # 导出为Shapefile（GEE Asset上传需要）
    print("\n正在导出Shapefile格式（用于GEE上传）...")
    
    # 分别导出正负样本为Shapefile
    pos_shp_path = os.path.join(output_dir, 'positive_samples.shp')
    positive_gdf.to_file(pos_shp_path, driver='ESRI Shapefile')
    print(f"已保存: {pos_shp_path}")
    
    neg_shp_path = os.path.join(output_dir, 'negative_samples.shp')
    negative_gdf.to_file(neg_shp_path, driver='ESRI Shapefile')
    print(f"已保存: {neg_shp_path}")
    
    # 导出所有样本为Shapefile
    all_shp_path = os.path.join(output_dir, 'all_samples.shp')
    all_samples.to_file(all_shp_path, driver='ESRI Shapefile')
    print(f"已保存: {all_shp_path}")
    
    # 导出南北分区数据
    north_samples = all_samples[all_samples['region'] == 'north']
    south_samples = all_samples[all_samples['region'] == 'south']
    
    north_shp_path = os.path.join(output_dir, 'north_samples.shp')
    north_samples.to_file(north_shp_path, driver='ESRI Shapefile')
    print(f"已保存: {north_shp_path}")
    
    south_shp_path = os.path.join(output_dir, 'south_samples.shp')
    south_samples.to_file(south_shp_path, driver='ESRI Shapefile')
    print(f"已保存: {south_shp_path}")
    
    # 将Shapefile打包成ZIP（方便上传）
    print("\n正在打包Shapefile为ZIP...")
    import zipfile
    
    def zip_shapefile(shp_path):
        """将shapefile及其相关文件打包成zip"""
        base_name = shp_path.replace('.shp', '')
        zip_path = base_name + '.zip'
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # 添加所有相关文件
            extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
            for ext in extensions:
                file_path = base_name + ext
                if os.path.exists(file_path):
                    zipf.write(file_path, os.path.basename(file_path))
        
        return zip_path
    
    pos_zip = zip_shapefile(pos_shp_path)
    neg_zip = zip_shapefile(neg_shp_path)
    all_zip = zip_shapefile(all_shp_path)
    north_zip = zip_shapefile(north_shp_path)
    south_zip = zip_shapefile(south_shp_path)
    
    print(f"已打包: {pos_zip}")
    print(f"已打包: {neg_zip}")
    print(f"已打包: {all_zip}")
    print(f"已打包: {north_zip}")
    print(f"已打包: {south_zip}")
    
    # 生成统计报告
    report_path = os.path.join(output_dir, 'data_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("风机检测数据集统计报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
        f.write(f"总样本数: {len(all_samples)}\n")
        f.write(f"  正样本（风机）: {len(positive_gdf)}\n")
        f.write(f"  负样本: {len(negative_gdf)}\n\n")
        f.write(f"区域分布:\n")
        f.write(f"  北方: {len(north_samples)} (时间: 2020-04-01 至 2020-05-31)\n")
        f.write(f"  南方: {len(south_samples)} (时间: 2020-01-01 至 2020-02-28)\n\n")
        f.write(f"边界框大小: 0.0026° x 0.0026° (约289m x 289m)\n")
        f.write(f"坐标系统: EPSG:4326 (WGS84)\n")
    
    print(f"已保存: {report_path}")
    
    return all_samples

# ===== 主函数 =====
def main():
    print("="*60)
    print("风机检测数据预处理脚本")
    print("="*60)
    
    # 文件路径
    csv_path = r"C:\Users\34248\Desktop\大二\GEE-ManMadeSurface-ML\00_raw_data\global_wind_2020.csv"
    
    # 1. 读取中国风机数据
    china_wind = load_china_wind_data(csv_path)
    
    if china_wind is None or len(china_wind) == 0:
        print("错误: 无法读取中国风机数据，程序终止")
        return
    
    # 2. 生成风机边界框
    positive_gdf = generate_bounding_boxes(china_wind)
    
    # 3. 划分南北方
    positive_gdf = classify_north_south(positive_gdf)
    
    # 4. 生成负样本
    negative_gdf = generate_negative_samples(positive_gdf, n_samples=20000)
    
    # 5. 合并并导出
    all_samples = combine_and_export(positive_gdf, negative_gdf)
    
    print("\n" + "="*60)
    print("数据处理完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 将 GeoJSON 文件上传到 Google Earth Engine 作为 Asset")
    print("2. 在 GEE 中使用这些样本提取 Sentinel-1/2 影像特征")
    print("3. 训练随机森林分类器")
    print("\n输出文件位于 'processed_data' 目录")

if __name__ == "__main__":
    main()