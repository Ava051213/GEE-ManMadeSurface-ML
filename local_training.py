import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def extract_features_from_samples(samples_df):
    """从样本数据中提取特征"""
    features = []
    labels = []
    
    for _, row in samples_df.iterrows():
        # 基础坐标特征
        feature_vector = [
            row['center_x'],
            row['center_y'],
            row['xmin'],
            row['ymin'],
            row['xmax'],
            row['ymax']
        ]
        
        # 添加虚拟的光谱特征（实际应该从卫星影像提取）
        # 这里用随机值模拟，实际应用中需要从GEE导出真实特征
        spectral_features = np.random.uniform(0, 1, 10)  # 10个光谱特征
        feature_vector.extend(spectral_features.tolist())
        
        features.append(feature_vector)
        labels.append(row['class'])
    
    return np.array(features), np.array(labels)

def train_wind_turbine_model(region_name):
    """训练单个区域的风机检测模型"""
    # 加载区域数据
    csv_path = f'processed_data/{region_name}_samples.csv'
    if not os.path.exists(csv_path):
        print(f"⚠️  {region_name} data not found, skipping...")
        return None, None, None, None
        
    df = pd.read_csv(csv_path)
    
    # 提取特征和标签
    X, y = extract_features_from_samples(df)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 训练随机森林模型
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    # 评估模型
    y_pred = clf.predict(X_test)
    print(f"\n=== {region_name.upper()} MODEL EVALUATION ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 保存模型
    model_path = f'models/{region_name}_wind_model.pkl'
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"✅ Model saved to {model_path}")
    
    return clf, X_test, y_test, y_pred

def main():
    regions = ['north_china', 'east_china', 'southwest_china', 'northwest_china']
    
    for region in regions:
        train_wind_turbine_model(region)

if __name__ == "__main__":
    main()