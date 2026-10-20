// 中国风机检测系统 - 基于 reference/raw_code.js 模板改写
//
// 数据来源:
// 1. data/raw/global_wind_2020.csv
// 2. code/extract_china_wind.py 转换中国风机坐标
// 3. data/samples/china_wind_turbines_wgs84.csv 上传为正样本 GEE Asset
// 4. code/generate_negative_samples.py 生成 data/samples/negative_samples.csv, 上传为负样本 GEE Asset
//
// 主流程:
// 加载预计算训练特征 -> Sentinel-2/Sentinel-1 预测区特征 -> 随机森林训练 -> 概率预测 -> 后处理 -> 导出


// ==================== 配置区 ====================

var CONFIG = {
    // --- 样本 Asset ---
    // 正样本来自 china_wind_turbines_wgs84.csv
    positiveAsset: 'users/YOUR_GEE_USERNAME/china_wind_turbines_wgs84',

    // 负样本来自 negative_samples.csv
    // 如果你上传时使用了其他 Asset ID, 只需要改这里。
    negativeAsset: 'users/YOUR_GEE_USERNAME/negative_samples',

    // 预计算训练特征表, 来自历史导出的 sampleRegions 结果。
    // 该表不需要有效几何, 只要包含下方 TRAINING_BANDS 和 class 字段即可。
    trainingAsset: 'users/YOUR_GEE_USERNAME/wind_turbine_training_data_v2',

    // --- 运行区域 ---
    // TEST_SHANGHAI: 快速检查代码能否跑通
    // PROVINCE_ZHANGBEI: 河北张北风电集中区
    // PROVINCE_SHAANXI: 陕西示例区域
    // CHINA: 全国范围, 计算量较大
    mode: 'PROVINCE_ZHANGBEI',

    // --- 遥感数据 ---
    year: 2024,
    sampleScale: 10,
    exportScale: 20,
    cloudPercent: 15,
    tileScale: 16,

    // --- 样本与模型 ---
    negativeMultiplier: 2,
    trainRatio: 0.7,
    trees: 200,
    bagFraction: 0.8,
    seed: 42,

    // --- 预测后处理 ---
    threshold: 0.55,
    minConnectedPixels: 3,

    // --- 导出 ---
    exportResults: true,
    exportFolder: 'GEE_export',
    filenamePrefix: 'wind_turbine_detection_2024'
};


// ==================== 区域定义 ====================

var REGIONS = {
    TEST_SHANGHAI: {
        name: 'test_shanghai',
        label: '上海测试区',
        geometry: ee.Geometry.Rectangle([120.85, 30.65, 122.05, 31.88]),
        trainingGeometry: ee.Geometry.Rectangle([110, 25, 123, 32]),
        trainingLabel: '华东训练区',
        zoom: 10
    },
    PROVINCE_ZHANGBEI: {
        name: 'zhangbei',
        label: '河北张北',
        geometry: ee.Geometry.Rectangle([114.5, 40.8, 115.8, 41.6]),
        trainingGeometry: ee.Geometry.Rectangle([105, 35, 123, 45]),
        trainingLabel: '华北-东北过渡训练区',
        zoom: 9
    },
    PROVINCE_SHAANXI: {
        name: 'shaanxi',
        label: '陕西',
        geometry: ee.Geometry.Rectangle([105.5, 31.5, 111, 39.5]),
        zoom: 7
    },
    CHINA: {
        name: 'china',
        label: '中国全境',
        geometry: ee.Geometry.Rectangle([73, 18, 135, 55]),
        zoom: 4
    }
};

var analysisConfig = REGIONS[CONFIG.mode] || REGIONS.PROVINCE_ZHANGBEI;
var predictionRegion = analysisConfig.geometry;

print('运行模式:', CONFIG.mode);
print('预测区域:', analysisConfig.label);
print('训练数据:', CONFIG.trainingAsset);


// 必须与 wind_turbine_training_data_v2 中的字段完全一致。
var TRAINING_BANDS = [
    'B11', 'B12', 'B2', 'B3', 'B4',
    'B4_asm', 'B4_contrast', 'B4_diss', 'B4_var',
    'B8', 'B8_asm', 'B8_contrast', 'B8_diss', 'B8_var',
    'BSI', 'NDBI', 'NDVI',
    'NDVI_autumn', 'NDVI_spring', 'NDVI_summer', 'NDVI_winter',
    'NDWI', 'VH', 'VV', 'VV_VH_ratio'
];


// ==================== 样本加载 ====================

function setSampleClass(feature, classValue) {
    return feature.set('class', classValue);
}

var positiveFeatures = ee.FeatureCollection(CONFIG.positiveAsset)
    .map(function (feature) {
        return setSampleClass(feature, 1);
    });

var negativePool = ee.FeatureCollection(CONFIG.negativeAsset)
    .map(function (feature) {
        return setSampleClass(feature, 0);
    });

var positiveInPrediction = positiveFeatures.filterBounds(predictionRegion);
var negativeInPrediction = negativePool.filterBounds(predictionRegion);
print('预测区域正样本数量:', positiveInPrediction.size());
print('预测区域负样本数量:', negativeInPrediction.size());


// ==================== 遥感特征工程 ====================

var startDate = CONFIG.year + '-01-01';
var endDate = CONFIG.year + '-12-31';

var seasons = [
    { name: 'winter', startDate: CONFIG.year + '-01-01', endDate: CONFIG.year + '-03-31' },
    { name: 'spring', startDate: CONFIG.year + '-04-01', endDate: CONFIG.year + '-06-30' },
    { name: 'summer', startDate: CONFIG.year + '-07-01', endDate: CONFIG.year + '-09-30' },
    { name: 'autumn', startDate: CONFIG.year + '-10-01', endDate: CONFIG.year + '-12-31' }
];

function maskS2clouds(image) {
    var qa = image.select('QA60');
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
        .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

    return image.updateMask(mask)
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
        .copyProperties(image, ['system:time_start']);
}

var sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(startDate, endDate)
    .filterBounds(predictionRegion)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.cloudPercent))
    .map(maskS2clouds);

var sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterDate(startDate, endDate)
    .filterBounds(predictionRegion)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);

print('Sentinel-2 影像数量:', sentinel2.size());
print('Sentinel-1 SAR 影像数量:', sentinel1.size());

var selectedBands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
var composite = sentinel2.select(selectedBands).median().clip(predictionRegion);

var s1Composite = sentinel1.median().clip(predictionRegion);
var s1VV = s1Composite.select('VV');
var s1VH = s1Composite.select('VH');
var s1Ratio = s1VV.divide(s1VH).rename('VV_VH_ratio');

var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
var ndbi = composite.normalizedDifference(['B11', 'B8']).rename('NDBI');
var bsi = composite.expression(
    '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))',
    {
        B11: composite.select('B11'),
        B4: composite.select('B4'),
        B8: composite.select('B8'),
        B2: composite.select('B2')
    }
).rename('BSI');

function makeSeasonalNdvi(season) {
    var seasonCollection = sentinel2.filterDate(season.startDate, season.endDate);
    var seasonalImage = ee.Image(ee.Algorithms.If(
        seasonCollection.size().gt(0),
        seasonCollection.median().normalizedDifference(['B8', 'B4']),
        ee.Image.constant(0)
    ));

    return seasonalImage.rename('NDVI_' + season.name).clip(predictionRegion);
}

var seasonalNdvis = [];
for (var s = 0; s < seasons.length; s++) {
    seasonalNdvis.push(makeSeasonalNdvi(seasons[s]));
}

var scaleFactor = 100;
var b4Texture = composite.select('B4').multiply(scaleFactor).toInt()
    .glcmTexture({ size: 3 });
var b8Texture = composite.select('B8').multiply(scaleFactor).toInt()
    .glcmTexture({ size: 3 });

var allFeatures = ee.Image.cat([
    composite,
    ndvi, ndwi, ndbi, bsi,
    s1VV, s1VH, s1Ratio,
    b4Texture.select(['B4_contrast', 'B4_diss', 'B4_asm', 'B4_var']),
    b8Texture.select(['B8_contrast', 'B8_diss', 'B8_asm', 'B8_var']),
    ee.Image.cat(seasonalNdvis)
]).select(TRAINING_BANDS);

print('特征波段:', allFeatures.bandNames());
print('特征波段数量:', allFeatures.bandNames().size());


// ==================== 训练与评估 ====================

var trainingProperties = TRAINING_BANDS.concat(['class']);
var training = ee.FeatureCollection(CONFIG.trainingAsset)
    .select(trainingProperties)
    .filter(ee.Filter.notNull(trainingProperties));

print('预计算训练样本总数:', training.size());

var withRandom = training.randomColumn('random', CONFIG.seed);
var trainingSet = withRandom.filter(ee.Filter.lt('random', CONFIG.trainRatio));
var validationSet = withRandom.filter(ee.Filter.gte('random', CONFIG.trainRatio));

var positiveTraining = trainingSet.filter(ee.Filter.eq('class', 1));
var negativeTraining = trainingSet.filter(ee.Filter.eq('class', 0));
var balancedTrainingSet = positiveTraining.merge(positiveTraining).merge(negativeTraining);

print('训练集正样本:', positiveTraining.size());
print('训练集负样本:', negativeTraining.size());
print('训练集平衡后数量:', balancedTrainingSet.size());
print('验证集数量:', validationSet.size());

var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: CONFIG.trees,
    minLeafPopulation: 1,
    bagFraction: CONFIG.bagFraction,
    seed: CONFIG.seed
}).train({
    features: balancedTrainingSet,
    classProperty: 'class',
    inputProperties: TRAINING_BANDS
});

var validation = validationSet.classify(classifier);
var confusionMatrix = validation.errorMatrix('class', 'classification');
print('混淆矩阵:', confusionMatrix);
print('总体精度:', confusionMatrix.accuracy());
print('Kappa系数:', confusionMatrix.kappa());

var cmArray = confusionMatrix.array();
var TP = cmArray.get([1, 1]);
var FN = cmArray.get([1, 0]);
var FP = cmArray.get([0, 1]);
var recall = ee.Number(TP).divide(ee.Number(TP).add(ee.Number(FN)));
var precision = ee.Number(TP).divide(ee.Number(TP).add(ee.Number(FP)));
var f1 = precision.multiply(recall).multiply(2).divide(precision.add(recall));
print('召回率 (Recall):', recall);
print('精确率 (Precision):', precision);
print('F1分数:', f1);


// ==================== 预测与后处理 ====================

var probabilityClassifier = classifier.setOutputMode('PROBABILITY');
var predictionFeatures = allFeatures.clip(predictionRegion);
var probability = predictionFeatures.classify(probabilityClassifier).rename('probability');

var thresholded = probability.gte(CONFIG.threshold).rename('wind_turbine');
var pixelCount = thresholded.connectedPixelCount(50, true);
var sieved = thresholded
    .updateMask(pixelCount.gte(CONFIG.minConnectedPixels))
    .selfMask()
    .rename('wind_turbine');

var areaImage = sieved.multiply(ee.Image.pixelArea()).rename('area_sqm');
var areaStats = areaImage.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: predictionRegion,
    scale: CONFIG.exportScale,
    maxPixels: 1e10,
    tileScale: CONFIG.tileScale
});

var windAreaSqMeters = areaStats.get('area_sqm');
var windAreaHectares = ee.Number(windAreaSqMeters).divide(10000);
print('预测风机区域面积 (平方米):', windAreaSqMeters);
print('预测风机区域面积 (公顷):', windAreaHectares);


// ==================== 可视化 ====================

Map.centerObject(predictionRegion, analysisConfig.zoom);

Map.addLayer(
    composite.clip(predictionRegion),
    { bands: ['B4', 'B3', 'B2'], min: 0, max: 3000 },
    analysisConfig.label + ' Sentinel-2 真彩色',
    true
);

Map.addLayer(s1VV.clip(predictionRegion), { min: -25, max: 0 }, 'Sentinel-1 VV', false);
Map.addLayer(s1VH.clip(predictionRegion), { min: -30, max: -5 }, 'Sentinel-1 VH', false);

Map.addLayer(
    probability,
    { min: 0, max: 1, palette: ['ffffff', 'ffffcc', 'ff9900', 'ff0000'] },
    '风机概率图',
    true
);

Map.addLayer(
    probability.updateMask(probability.gte(CONFIG.threshold)),
    { min: CONFIG.threshold, max: 1, palette: ['ffcc00', 'ff0000'] },
    '超过阈值的概率区域',
    false
);

Map.addLayer(
    sieved,
    { min: 0, max: 1, palette: ['0066ff'] },
    '预测风机区域 (后处理)',
    true
);

Map.addLayer(
    positiveInPrediction.style({ color: 'yellow', pointSize: 4, width: 1 }),
    {},
    '风机正样本',
    true
);

Map.addLayer(
    negativeInPrediction.style({ color: 'red', pointSize: 2, width: 1 }),
    {},
    '负样本',
    false
);


// ==================== 导出 ====================

if (CONFIG.exportResults) {
    Export.image.toDrive({
        image: sieved.toByte(),
        description: CONFIG.filenamePrefix + '_' + analysisConfig.name,
        folder: CONFIG.exportFolder,
        scale: CONFIG.exportScale,
        region: predictionRegion,
        maxPixels: 1e10
    });

    Export.image.toDrive({
        image: probability.toFloat(),
        description: CONFIG.filenamePrefix + '_' + analysisConfig.name + '_probability',
        folder: CONFIG.exportFolder,
        scale: CONFIG.exportScale,
        region: predictionRegion,
        maxPixels: 1e10
    });
}


// ==================== 图例 ====================

var legend = ui.Panel({
    style: {
        position: 'bottom-right',
        padding: '8px 15px'
    }
});

legend.add(ui.Label({
    value: '风机检测图例',
    style: {
        fontWeight: 'bold',
        fontSize: '16px',
        margin: '0 0 4px 0',
        padding: '0'
    }
}));

function makeLegendRow(color, name) {
    var colorBox = ui.Label({
        style: {
            backgroundColor: color,
            padding: '8px',
            margin: '0 0 4px 0'
        }
    });

    var description = ui.Label({
        value: name,
        style: { margin: '0 0 4px 6px' }
    });

    return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
    });
}

legend.add(makeLegendRow('#0066ff', '预测风机区域'));
legend.add(makeLegendRow('yellow', '风机正样本'));
legend.add(makeLegendRow('red', '负样本'));
legend.add(makeLegendRow('#ff9900', '高概率区域'));

Map.add(legend);

print('===== 风机检测脚本执行完成 =====');
