# 风机检测系统

## 系统架构
- **数据预处理**: `prepare_data.py`
- **GEE集成**: `upload_to_gee.py`, `gee_scripts/`
- **本地训练**: `local_training.py`
- **完整工作流**: `run_complete_workflow.py`

## 快速开始
1. 安装依赖: `pip install pandas geopandas scikit-learn earthengine-api joblib`
2. 运行完整工作流: `python run_complete_workflow.py`
3. 在GEE中运行生成的预测脚本

## 目录结构
- `00_raw_data/`: 原始数据
- `processed_data/`: 处理后的数据
- `gee_scripts/`: GEE预测脚本
- `models/`: 本地训练的模型
- `docs/plans/`: 实现计划文档