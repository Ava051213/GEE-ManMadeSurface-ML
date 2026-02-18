import pandas as pd
import os

def generate_region_prediction_script(region_name, bbox_coords, time_range):
    """生成单个区域的GEE预测脚本"""
    
    script_template = f'''
// 风机检测 - {region_name.replace("_", " ").title()} 预测脚本
// 生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

// 定义区域边界
var region_bbox = ee.Geometry.Rectangle({bbox_coords});

// 加载样本数据
var samples = ee.FeatureCollection('users/wuxuanyue051213/{region_name}_wind_samples');

// 加载Sentinel-2数据
var sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterDate('{time_range[0]}', '{time_range[1]}')
  .filterBounds(region_bbox)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(function(image) {{
    var cloudProb = image.select('MSK_CLDPRB');
    var mask = cloudProb.lt(20);
    return image.updateMask(mask).select(['B2','B3','B4','B8','B11','B12']);
  }});

// 创建合成影像
var composite = sentinel2.median().clip(region_bbox);

// 计算指数
var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
var ndbi = composite.normalizedDifference(['B11', 'B8']).rename('NDBI');

// 合并特征
var allFeatures = ee.Image.cat([composite, ndvi, ndwi, ndbi]);

// 提取训练数据
var training = allFeatures.sampleRegions({{
  collection: samples,
  properties: ['class'],
  scale: 20,
  tileScale: 4
}});

// 划分训练/验证集
var withRandom = training.randomColumn('random');
var trainingSet = withRandom.filter(ee.Filter.lt('random', 0.7));
var validationSet = withRandom.filter(ee.Filter.gte('random', 0.7));

// 训练模型
var classifier = ee.Classifier.smileRandomForest(50).train({{
  features: trainingSet,
  classProperty: 'class',
  inputProperties: allFeatures.bandNames()
}});

// 评估模型
var validation = validationSet.classify(classifier);
var confusionMatrix = validation.errorMatrix('class', 'classification');
print('=== {region_name.upper()} EVALUATION ===');
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa:', confusionMatrix.kappa());

// 预测
var classified = allFeatures.classify(classifier);
Map.centerObject(region_bbox, 6);
Map.addLayer(composite, {{bands: ['B4','B3','B2'], min: 0, max: 3000}}, 'Composite');
Map.addLayer(classified, {{min: 0, max: 1, palette: ['white','green']}}, 'Wind Turbine Prediction');
Map.addLayer(samples.filter(ee.Filter.eq('class', 1)), {{color: 'yellow'}}, 'Positive Samples');
'''
    
    return script_template

def main():
    regions_config = {
        'north_china': {
            'bbox': [110, 35, 120, 45],
            'time_range': ['2024-04-01', '2024-09-30']
        },
        'east_china': {
            'bbox': [115, 25, 125, 35],
            'time_range': ['2024-01-01', '2024-09-30']
        },
        'southwest_china': {
            'bbox': [95, 25, 110, 35],
            'time_range': ['2024-01-01', '2024-09-30']
        },
        'northwest_china': {
            'bbox': [75, 35, 100, 45],
            'time_range': ['2024-04-01', '2024-09-30']
        }
    }
    
    os.makedirs('gee_scripts', exist_ok=True)
    
    for region, config in regions_config.items():
        script = generate_region_prediction_script(
            region, 
            config['bbox'], 
            config['time_range']
        )
        
        script_path = f'gee_scripts/{region}_prediction.js'
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)
        
        print(f"Generated {script_path}")

if __name__ == "__main__":
    main()