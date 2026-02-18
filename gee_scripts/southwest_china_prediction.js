
// 风机检测 - Southwest China 预测脚本
// 生成时间: 2026-02-18 22:52:23

// 定义区域边界
var region_bbox = ee.Geometry.Rectangle([95, 25, 110, 35]);

// 加载样本数据
var samples = ee.FeatureCollection('users/wuxuanyue051213/southwest_china_wind_samples');

// 加载Sentinel-2数据
var sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterDate('2024-01-01', '2024-09-30')
  .filterBounds(region_bbox)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(function(image) {
    var cloudProb = image.select('MSK_CLDPRB');
    var mask = cloudProb.lt(20);
    return image.updateMask(mask).select(['B2','B3','B4','B8','B11','B12']);
  });

// 创建合成影像
var composite = sentinel2.median().clip(region_bbox);

// 计算指数
var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
var ndbi = composite.normalizedDifference(['B11', 'B8']).rename('NDBI');

// 合并特征
var allFeatures = ee.Image.cat([composite, ndvi, ndwi, ndbi]);

// 提取训练数据
var training = allFeatures.sampleRegions({
  collection: samples,
  properties: ['class'],
  scale: 20,
  tileScale: 4
});

// 划分训练/验证集
var withRandom = training.randomColumn('random');
var trainingSet = withRandom.filter(ee.Filter.lt('random', 0.7));
var validationSet = withRandom.filter(ee.Filter.gte('random', 0.7));

// 训练模型
var classifier = ee.Classifier.smileRandomForest(50).train({
  features: trainingSet,
  classProperty: 'class',
  inputProperties: allFeatures.bandNames()
});

// 评估模型
var validation = validationSet.classify(classifier);
var confusionMatrix = validation.errorMatrix('class', 'classification');
print('=== SOUTHWEST_CHINA EVALUATION ===');
print('Overall Accuracy:', confusionMatrix.accuracy());
print('Kappa:', confusionMatrix.kappa());

// 预测
var classified = allFeatures.classify(classifier);
Map.centerObject(region_bbox, 6);
Map.addLayer(composite, {bands: ['B4','B3','B2'], min: 0, max: 3000}, 'Composite');
Map.addLayer(classified, {min: 0, max: 1, palette: ['white','green']}, 'Wind Turbine Prediction');
Map.addLayer(samples.filter(ee.Filter.eq('class', 1)), {color: 'yellow'}, 'Positive Samples');
