# 风机检测系统 — 开发约定

## 工作流程

修改代码必须遵循以下顺序：

1. **分析** — 先阅读相关代码和环境，定位问题根因，解释为什么会出现
2. **建议** — 提出具体修改方案，说明改动内容和预期效果
3. **提问** — 询问是否执行，等待回复
4. **执行** — 得到确认后再修改代码

## 项目结构

| 脚本 | 用途 |
|------|------|
| `code/wind_turbine_detection.js` | 主脚本：加载预计算训练特征 → 训练 → 影像预测 |
| `code/generate_negative_samples.py` | 本地生成 data/samples/negative_samples.csv |
| `code/extract_china_wind.py` | 从原始数据提取中国位置记录并转换坐标 |
| `reference/raw_code.js` | 原始跑道检测脚本（参考用） |

## 执行流程

从项目根目录运行：

1. python code/extract_china_wind.py → data/samples/china_wind_turbines_wgs84.csv
2. python code/generate_negative_samples.py → data/samples/negative_samples.csv
3. 手动上传样本CSV到GEE Asset。
4. 在GEE Code Editor中运行 code/wind_turbine_detection.js。

主脚本依赖预计算训练特征Asset，当前流程不会直接从位置CSV重建该表。

## 上传说明

本地样本CSV位于 data/samples/。上传步骤：

1. GEE Code Editor → Assets → NEW → Table Upload → CSV File。
2. 分别上传正样本与负样本CSV。
3. 在 code/wind_turbine_detection.js 中配置 positiveAsset、negativeAsset 和 trainingAsset。
