"""
构建郑州各区真实数据证据（基于 Grid500 网格数据）
================================================
用法:
  python scripts/build_zz_evidence.py

输出:
  configs/郑州/district_evidence.json  — 各区默认证据 {区名: {节点: 状态}}

流程:
  1. 读取 Grid500_AllCity_Nodes.csv（31021 网格，30 字段）
  2. 按地理分区规则分三组（山区/城区/平原）
  3. 计算每组字段中位数
  4. 按离散化映射表转为 BN 节点状态
  5. 输出 JSON，供 dashboard 加载
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径 ──
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "zhengzhou_720" / "Grid500_AllCity_Nodes.csv"
OUTPUT_PATH = ROOT / "configs" / "郑州" / "district_evidence.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 分组规则
# ============================================================================

# 全城分位数（从数据实际计算）
DEM_80PCT = 483.7   # Dem_Mean 80% 分位
DEM_50PCT = 158.3   # Dem_Mean 50% 分位
IMPERV_60PCT = 2107 # Impervious 60% 分位

# 各区所属分组
DISTRICT_GROUPS = {
    "山区": ["巩义市", "登封市", "新密市"],
    "城区": ["中原区", "二七区", "金水区", "管城回族区", "惠济区", "上街区"],
    "平原": ["中牟县", "新郑市", "荥阳市"],
}


# ============================================================================
# 离散化映射（与数据验证实验一致的阈值）
# ============================================================================

def discretize_evidence(group_stats: dict, city_stats: dict) -> dict:
    """
    根据分组统计中位数 + 全城统计，映射为 BN 节点状态

    返回值: {节点名: 状态, ...}
    """
    m = group_stats["median"]  # 组中位数

    evidence = {}

    # ── 气象 ──
    # 降水强度 ← rain0720 全城 33%/66% 分位数
    q33, q66 = city_stats["rain0720_q33"], city_stats["rain0720_q66"]
    rain_val = m.get("rain0720", city_stats["rain0720"])
    if rain_val <= q33:
        evidence["降水强度"] = "低"
    elif rain_val <= q66:
        evidence["降水强度"] = "中"
    else:
        evidence["降水强度"] = "高"

    # 降水时长 — 全城统一"长"（7·20 暴雨 6 小时≥5mm/h）
    evidence["降水时长"] = "长"

    # 风力 — 保持先验（无 ERA5 网格数据）
    # 风向 — 保持先验（郑州内陆无向岸风）
    # 湿度 ← rhu2021 50% 分位
    if "rhu2021" in m and not np.isnan(m["rhu2021"]):
        evidence["湿度"] = "干燥" if m["rhu2021"] <= 61.20 else "湿润"
    # 气压 ← prs2021 50% 分位
    if "prs2021" in m and not np.isnan(m["prs2021"]):
        evidence["气压"] = "正常" if m["prs2021"] <= 997.2 else "偏低"
    # 气温 — 保持先验（无 ERA5 网格）
    # 露点温度 — 保持先验
    # 对流有效位能CAPE — 保持先验
    # 垂直风切变 — 保持先验

    # ── 水文 ──
    # 前期土壤含水量 — 保持先验（无 ERA5 网格数据）
    # 径流系数 ← Impervious/10000 50% 分位
    imperv = m.get("Impervious", 0)
    runoff = imperv / 10000.0
    evidence["径流系数"] = "小" if runoff <= 0.0761 else "大"
    # 河道水位 — 保持先验
    # 地下水埋深 — 保持先验
    # 湖泊调蓄能力 ← WaterBody_ 50% 分位
    waterbody = m.get("WaterBody_", 0)
    evidence["湖泊调蓄能力"] = "强" if waterbody <= 0.0 else "弱"
    # 潮汐影响 — 保持先验（郑州非沿海）
    # 蒸发量 ← evp2021 50% 分位
    if "evp2021" in m and not np.isnan(m["evp2021"]):
        evidence["蒸发量"] = "小" if m["evp2021"] <= 1242 else "大"

    # ── 地形 ──
    # 海拔 ← Dem_Mean 50% 分位
    dem = m.get("Dem_Mean", 0)
    evidence["海拔"] = "低" if dem <= DEM_50PCT else "高"
    # 坡度 ← Slope_Mean 50% 分位
    slope = m.get("Slope_Mean", 0)
    evidence["坡度"] = "缓" if slope <= 1.65 else "陡"
    # 坡向 ← Aspect_Mea 180° 划分
    aspect = m.get("Aspect_Mea", 0)
    evidence["坡向"] = "阴坡" if aspect <= 180 else "阳坡"
    # 地形起伏度 ← Dem_Std 50% 分位
    dem_std = m.get("Dem_Std", 0)
    evidence["地形起伏度"] = "小" if dem_std <= 3.17 else "大"
    # 汇流面积 — 保持先验
    # 河网密度 — 保持先验
    # 地表粗糙度 — 保持先验
    # 沟谷密度 — 保持先验

    # ── 地质 ──
    # 植被覆盖 — 保持先验
    # 土壤渗透性 — 保持先验（无 ERA5 网格）
    # 岩性 — 保持先验
    # 距断层距离 — 保持先验
    # 地震烈度 — 保持先验
    # 风化程度 — 保持先验
    # 节理发育程度 — 保持先验
    # 滑坡历史密度 — 保持先验

    # ── 城市 ──
    # 历史排水时间 ← WorldPop_M 50% 分位
    pop = m.get("WorldPop_M", 0)
    evidence["历史排水时间"] = "快" if pop <= 4.44 else "慢"
    # 管网排水能力 ← Impervious 反向 50% 分位
    evidence["管网排水能力"] = "强" if imperv <= 761 else "弱"
    # 下垫面硬化率 ← Impervious/2500 33%/66% 分位
    imperv_ratio = imperv / 2500.0
    if imperv_ratio <= 0.102:
        evidence["下垫面硬化率"] = "低"
    elif imperv_ratio <= 0.570:
        evidence["下垫面硬化率"] = "中"
    else:
        evidence["下垫面硬化率"] = "高"
    # 道路积水历史频率 — 保持先验
    # 建筑密度 ← Light_N 33%/66% 分位
    light = m.get("Light_N", 0)
    if light <= 1.11:
        evidence["建筑密度"] = "低"
    elif light <= 3.35:
        evidence["建筑密度"] = "中"
    else:
        evidence["建筑密度"] = "高"
    # 绿地率 ← 1-Impervious/2500 33%/66% 分位
    green = 1.0 - imperv_ratio
    if green <= 0.430:
        evidence["绿地率"] = "低"
    elif green <= 0.898:
        evidence["绿地率"] = "中"
    else:
        evidence["绿地率"] = "高"
    # 应急排水能力 — 保持先验

    return evidence


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 60)
    print("郑州各区真实数据证据构建")
    print("=" * 60)

    # 1. 读取 CSV
    print(f"\n[1] 读取 CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"    总网格数: {len(df)}")

    # 2. 按地理分区规则分组
    print("\n[2] 按地理分区规则分组...")

    # 山区: Dem_Mean > 80% 分位
    mask_mountain = df["Dem_Mean"] > DEM_80PCT
    # 城区: Dem_Mean < 50% 分位 且 Impervious > 60% 分位
    mask_urban = (df["Dem_Mean"] < DEM_50PCT) & (df["Impervious"] > IMPERV_60PCT)
    # 平原: 其余
    mask_plain = ~(mask_mountain | mask_urban)

    groups = {
        "山区": df[mask_mountain],
        "城区": df[mask_urban],
        "平原": df[mask_plain],
    }

    for gname, gdf in groups.items():
        print(f"    {gname}: {len(gdf)} 网格"
              f" → {DISTRICT_GROUPS[gname]}")

    # 3. 计算组中位数 + 全城统计
    print("\n[3] 计算组中位数...")

    key_fields = [
        "Dem_Mean", "Dem_Std", "Slope_Mean", "Aspect_Mea",
        "Impervious", "WaterBody_", "WorldPop_M", "Light_N",
        "evp2021", "prs2021", "rhu2021", "rain0720",
    ]

    group_stats = {}
    for gname, gdf in groups.items():
        median = gdf[key_fields].median().to_dict()
        mean = gdf[key_fields].mean().to_dict()
        group_stats[gname] = {
            "count": len(gdf),
            "median": median,
            "mean": mean,
        }
        print(f"\n  {gname} (n={len(gdf)}):")
        for k in key_fields:
            print(f"    {k}: median={median[k]:.2f}, mean={mean[k]:.2f}")

    # 全城统计（用于降水强度等全城统一量）
    city_stats = {
        "rain0720": df["rain0720"].median(),
        "rain0720_q33": df["rain0720"].quantile(1/3),
        "rain0720_q66": df["rain0720"].quantile(2/3),
    }
    print(f"\n  全城 rain0720: median={city_stats['rain0720']:.2f}, "
          f"q33={city_stats['rain0720_q33']:.2f}, "
          f"q66={city_stats['rain0720_q66']:.2f}")

    # 4. 离散化映射 → 各区证据
    print("\n[4] 离散化映射 → BN 节点状态...")

    district_evidence = {}
    for gname, districts in DISTRICT_GROUPS.items():
        ev = discretize_evidence(group_stats[gname], city_stats)
        for d in districts:
            district_evidence[d] = dict(ev)
            print(f"    {d} ({gname}): {len(ev)} 节点映射")

    # 5. 输出 JSON
    print(f"\n[5] 输出 JSON: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(district_evidence, f, ensure_ascii=False, indent=2)
    print(f"    {len(district_evidence)} 个区，共 "
          f"{sum(len(v) for v in district_evidence.values())} 节点映射")

    # 6. 打印汇总
    print("\n" + "=" * 60)
    print("各区证据摘要")
    print("=" * 60)
    for d, ev in district_evidence.items():
        print(f"\n  {d}:")
        for k, v in sorted(ev.items()):
            print(f"    {k}: {v}")

    print("\n✅ 完成")


if __name__ == "__main__":
    main()