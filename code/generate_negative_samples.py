"""
生成负样本 CSV（本地运行）
============================
从正样本 CSV 读取风机坐标，在 7 个区域中分别生成随机候选点，
排除所有正样本 1000m 范围内的点，输出为 CSV。

用法:
    python code/generate_negative_samples.py

输出:
    data/samples/negative_samples.csv  (longitude, latitude, class)
    所有 class 为 0（负样本）

工作流程:
    本地生成 → 你手动上传 CSV 到 GEE Asset →
    wind_turbine_detection.js 从 Asset 加载
"""

import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import math
from pathlib import Path

# ==================== 配置 ====================
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
POSITIVE_CSV = SAMPLE_DIR / "china_wind_turbines_wgs84.csv"
OUTPUT_CSV = SAMPLE_DIR / "negative_samples.csv"
NEGATIVE_MULTIPLIER = 2      # 负样本数 = 正样本数 × 2
BUFFER_METERS = 1000          # 正样本排除区半径
SEED = 42
CANDIDATE_MULTIPLIER = 20     # 生成 N× 候选点，过滤后取需要数量

# 7 个区域 (与 GEE 脚本一致)
REGIONS = [
    {"name": "西北v1", "bbox": [73, 35, 105, 55]},
    {"name": "西北v2", "bbox": [73, 20, 100, 35]},
    {"name": "东北",   "bbox": [105, 35, 135, 55]},
    {"name": "华北",   "bbox": [100, 30, 123, 40]},
    {"name": "西南",   "bbox": [100, 20, 110, 30]},
    {"name": "华东",   "bbox": [110, 25, 123, 32]},
    {"name": "华南",   "bbox": [108, 18, 123, 25]},
]
# bbox 格式: [west, south, east, north]


# ==================== 坐标转换 ====================

def degrees_to_meters(lon, lat, ref_lat=35.0):
    """
    将经纬度转换为近似米坐标（以中国中部纬度 ~35°N 为参考）。
    用在 KD-tree 中做 1000m 半径查询。
    """
    lat_rad = math.radians(ref_lat)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat_rad)
    return lon * m_per_deg_lon, lat * m_per_deg_lat


# ==================== 过滤逻辑 ====================

def is_within_bbox(lon, lat, bbox):
    """检查点是否在 bbox [west, south, east, north] 内"""
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def count_positives_in_region(pos_df, bbox):
    """统计某区域内的正样本数"""
    mask = (
        pos_df['longitude'].between(bbox[0], bbox[2])
        & pos_df['latitude'].between(bbox[1], bbox[3])
    )
    return mask.sum()


def generate_negatives_for_region(
    pos_df, tree, coord_m, bbox, needed_count, rng
):
    """
    为一个区域生成负样本。
    - 随机生成候选点
    - 用 KD-tree 排除在正样本 buffer 内的点
    - 返回 DataFrame
    """
    if needed_count < 1:
        return pd.DataFrame(columns=['longitude', 'latitude', 'class'])

    w, s, e, n = bbox
    candidates_count = needed_count * CANDIDATE_MULTIPLIER

    # 生成候选随机点
    lons = rng.uniform(w, e, candidates_count)
    lats = rng.uniform(s, n, candidates_count)

    # 转换候选点为米坐标
    cand_lon_m, cand_lat_m = degrees_to_meters(lons, lats)
    cand_coords = np.column_stack([cand_lon_m, cand_lat_m])

    # KD-tree 查询: 找 buffer 内的正样本
    # 对每个候选点，检查离最近正样本是否 < BUFFER_METERS
    distances, _ = tree.query(cand_coords, k=1, distance_upper_bound=BUFFER_METERS)

    # 过滤: 距离 >= BUFFER_METERS 的候选点保留
    safe_mask = distances >= BUFFER_METERS
    safe_lons = lons[safe_mask]
    safe_lats = lats[safe_mask]

    # 取所需数量
    available = len(safe_lons)
    if available < needed_count:
        print(f"    ⚠️ 警告: 只找到 {available} 个合格负样本, 需要 {needed_count}")
        needed_count = available

    take_lons = safe_lons[:needed_count]
    take_lats = safe_lats[:needed_count]

    return pd.DataFrame({
        'longitude': take_lons,
        'latitude': take_lats,
        'class': 0
    })


# ==================== 主流程 ====================

def main():
    rng = np.random.default_rng(SEED)

    # 1. 加载正样本
    print(f"加载正样本: {POSITIVE_CSV}")
    pos_df = pd.read_csv(POSITIVE_CSV)
    print(f"正样本总数: {len(pos_df)}")

    # 2. 构建 KD-tree (全量正样本, 坐标转米)
    lon_m_all, lat_m_all = degrees_to_meters(
        pos_df['longitude'].values, pos_df['latitude'].values
    )
    all_coords_m = np.column_stack([lon_m_all, lat_m_all])
    tree = KDTree(all_coords_m)
    print(f"KD-tree 构建完成 ({len(pos_df)} 个节点)")

    # 3. 逐区域生成负样本
    all_negatives = []
    total_needed = 0
    total_got = 0

    print("\n逐区域生成负样本:")
    print("-" * 50)
    for region in REGIONS:
        name = region["name"]
        bbox = region["bbox"]

        pos_count = count_positives_in_region(pos_df, bbox)
        neg_needed = pos_count * NEGATIVE_MULTIPLIER
        total_needed += neg_needed

        print(f"  {name}: 正样本 {pos_count} → 需负样本 {neg_needed}")

        if pos_count == 0:
            print(f"    跳过 (无正样本)")
            continue

        neg_df = generate_negatives_for_region(
            pos_df, tree, all_coords_m, bbox, neg_needed, rng
        )
        got = len(neg_df)
        total_got += got

        all_negatives.append(neg_df)
        print(f"    实际获取 {got} 个负样本")

    # 4. 合并 & 输出
    if all_negatives:
        result = pd.concat(all_negatives, ignore_index=True)
        result.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'=' * 50}")
        print(f"完成! 输出: {OUTPUT_CSV}")
        print(f"  正样本总数: {len(pos_df)}")
        print(f"  负样本总数: {len(result)}")
        print(f"  总计: {len(pos_df) + len(result)} 样本")
        print(f"\n下一步: 将 {OUTPUT_CSV} 上传到 GEE Asset:")
        print(f"  GEE Code Editor → Assets → NEW → Table Upload → CSV File")
        print(f"  Asset ID: users/你的用户名/wind_turbine_negative_samples")
        print(f"  然后在 wind_turbine_detection.js 中设置 negativeAsset")
    else:
        print("未生成任何负样本!")


if __name__ == "__main__":
    main()
