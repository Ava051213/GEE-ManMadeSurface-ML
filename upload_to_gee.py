"""
GEE数据上传脚本 - 将处理后的数据上传到GEE Asset
需要先安装earthengine-api: pip install earthengine-api
"""

import ee
import os
import time

def initialize_ee():
    """初始化GEE"""
    ee.Initialize()

def upload_region_data(region_name, username):
    """上传单个区域的数据到GEE"""
    geojson_path = f'processed_data/{region_name}_samples.geojson'
    asset_id = f'users/{username}/{region_name}_wind_samples'
    
    if not os.path.exists(geojson_path):
        print(f"❌ {geojson_path} not found")
        return False
    
    # 创建上传任务
    task = ee.batch.Export.table.toAsset(
        collection=ee.FeatureCollection(geojson_path),
        description=f'upload_{region_name}',
        assetId=asset_id
    )
    
    task.start()
    print(f"📤 Uploading {region_name} to {asset_id}")
    
    # 监控上传状态
    while task.status()['state'] in ['RUNNING', 'READY']:
        print(f"   Status: {task.status()['state']}")
        time.sleep(10)
    
    final_status = task.status()['state']
    if final_status == 'COMPLETED':
        print(f"✅ {region_name} uploaded successfully")
        return True
    else:
        print(f"❌ Upload failed: {task.status()}")
        return False

def main():
    initialize_ee()
    username = "wuxuanyue051213"  # 替换为你的GEE用户名
    regions = ['north_china', 'east_china', 'southwest_china', 'northwest_china']
    
    for region in regions:
        upload_region_data(region, username)

if __name__ == "__main__":
    main()