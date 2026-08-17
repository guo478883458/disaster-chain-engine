"""
预处理脚本：将郑州 7·20 验证数据离散化为节点状态，输出样本集
"""
import os, sys, shutil, tempfile, warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output", "validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def load_era5_copy(nc_relpath):
    """解决 netCDF4 在 Windows 上无法读取含 Unicode 路径的问题：
       复制到 D 盘根目录（ASCII 短路径）再读取，读取后删除临时文件。"""
    src = os.path.join(ROOT, nc_relpath)
    dst = r"D:\_era5_validation_temp.nc"
    shutil.copy2(src, dst)
    # 使用 with 语句确保文件正确关闭
    with xr.open_dataset(dst) as ds:
        ds.load()
        data = ds.copy()  # 复制到内存
    # 此时 Dataset 已关闭，可以删除文件
    if os.path.exists(dst):
        os.remove(dst)
    return data


def quantile_cut_2class(series, state_low, state_high):
    """二分法离散化：低于中位数=state_low，高于中位数=state_high"""
    thresh = series.median()
    result = np.where(series <= thresh, state_low, state_high)
    return result, thresh


def quantile_cut_3class(series, state_low, state_mid, state_high):
    """三分法离散化：33%/66% 分位数"""
    q33 = series.quantile(1/3)
    q66 = series.quantile(2/3)
    result = np.where(series <= q33, state_low,
                      np.where(series <= q66, state_mid, state_high))
    return result, (q33, q66)


def reverse_quantile_cut_2class(series, state_low, state_high):
    """反向二分法：高值→state_low，低值→state_high"""
    thresh = series.median()
    result = np.where(series <= thresh, state_high, state_low)
    return result, thresh


def reverse_quantile_cut_3class(series, state_low, state_mid, state_high):
    """反向三分法：高值→state_low，低值→state_high"""
    q33 = series.quantile(1/3)
    q66 = series.quantile(2/3)
    # 高值（如 Impervious 高→硬化率高）→ state_high
    result = np.where(series <= q33, state_low,
                      np.where(series <= q66, state_mid, state_high))
    return result, (q33, q66)


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("预处理：郑州 7·20 验证数据 → 节点状态样本集")
    print("=" * 60)

    # ── 1. 读取 CSV 数据 ──
    csv_path = os.path.join(DATA_DIR, "zhengzhou_720", "Grid500_AllCity_Nodes.csv")
    df = pd.read_csv(csv_path)
    print(f"\n[1] CSV 数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"    Label 分布: Y={int((df['Label']=='Y').sum())}, "
          f"N={int((df['Label']=='N').sum())}, "
          f"NaN={int(df['Label'].isna().sum())}")

    # ── 2. 读取 ERA5 数据 ──
    print("\n[2] 读取 ERA5-Land 数据...")
    nc_relpath = os.path.join("data", "era5land", "era5land_zhengzhou_20210718-0723.nc")
    era5 = load_era5_copy(nc_relpath)
    print(f"    ERA5 时间步: {len(era5.valid_time)}, "
          f"网格: {len(era5.latitude)}×{len(era5.longitude)}")

    # 提取 7/20 当天的数据 (valid_time 索引 48~71, 即 7/20 00:00~23:00)
    # 时间范围: 2021-07-18 00:00 ~ 2021-07-23 23:00, 共 144 小时
    jul20_idx = slice(48, 72)  # 7/20 00:00~23:00
    jul19_idx = 47  # 7/19 23:00 (雨前)

    # 计算 ERA5 域内平均值（城市尺度）
    def domain_mean(var, time_idx=None):
        if time_idx is not None:
            data = var.isel(valid_time=time_idx)
        else:
            data = var
        # 先求空间平均再求时间平均
        spatial_mean = data.mean(dim=["latitude", "longitude"])
        if "valid_time" in spatial_mean.dims:
            spatial_mean = spatial_mean.mean(dim="valid_time")
        return float(spatial_mean.values)

    # ── 2a. 降水时长：tp 差分 → 逐小时降水 → 统计 ≥5mm/h 小时数 ──
    tp_jul20 = era5.tp.isel(valid_time=jul20_idx)  # (24, 11, 21)
    tp_jul20_vals = tp_jul20.values  # 累积量，单位：米
    # 差分：tp_diff[i] = tp[i] - tp[i-1]（第一个值用 tp[0] - 0）
    tp_diff = np.diff(tp_jul20_vals, axis=0, prepend=tp_jul20_vals[0:1] * 0)
    # 如果 tp[0] 不是从 0 开始，用 tp[0] 作为第一个小时的值
    tp_diff[0] = tp_jul20_vals[0]
    # 转换为 mm
    tp_diff_mm = tp_diff * 1000.0
    # 域平均
    tp_hourly_mean = tp_diff_mm.mean(axis=(1, 2))  # (24,)
    heavy_rain_hours = int((tp_hourly_mean >= 5.0).sum())
    duration_state = "长" if heavy_rain_hours >= 2 else "短"
    print(f"    降水时长: ≥5mm/h 小时数={heavy_rain_hours}, 状态={duration_state}")

    # ── 2b. 风力/风向：u10/v10 合成 ──
    u10 = era5.u10.isel(valid_time=jul20_idx).mean(dim=["latitude", "longitude"]).values
    v10 = era5.v10.isel(valid_time=jul20_idx).mean(dim=["latitude", "longitude"]).values
    wind_speed = np.sqrt(u10**2 + v10**2)  # (24,)
    wind_dir = np.degrees(np.arctan2(v10, u10)) % 360  # (24,)
    wind_speed_mean = float(wind_speed.mean())
    wind_dir_mean = float(np.arctan2(v10.mean(), u10.mean()) * 180 / np.pi) % 360

    # ── 2c. 气温/露点温度 ──
    t2m_mean = domain_mean(era5.t2m, jul20_idx) - 273.15  # K→°C
    d2m_mean = domain_mean(era5.d2m, jul20_idx) - 273.15

    # ── 2d. 前期土壤含水量：swvl1 7/19 值 ──
    # 注意：swvl1 单位是 m³/m³，范围 0~1
    swvl1_before = domain_mean(era5.swvl1, jul19_idx)

    # ── 2e. 土壤渗透性：swvl2 7/20 均值（反向） ──
    swvl2_mean = domain_mean(era5.swvl2, jul20_idx)
    # swvl2 高 = 土壤含水多 = 渗透性差

    print(f"    风力: 风速均值={wind_speed_mean:.2f} m/s")
    print(f"    气温: {t2m_mean:.1f}°C, 露点: {d2m_mean:.1f}°C")
    print(f"    前期土壤含水量(swvl1): {swvl1_before:.4f}")
    print(f"    土壤渗透性(swvl2): {swvl2_mean:.4f}")

    # ── 3. 计算离散化阈值 (基于全样本) ──
    print("\n[3] 计算离散化阈值...")

    # 降水强度: [低, 中, 高]  ← rain0720 33%/66%
    rain_s, rain_th = quantile_cut_3class(df["rain0720"], "低", "中", "高")
    print(f"    降水强度: 阈值={rain_th[0]:.1f}/{rain_th[1]:.1f} mm")

    # 湿度: [干燥, 湿润]  ← rhu2021 50%
    humid_s, humid_th = quantile_cut_2class(df["rhu2021"], "干燥", "湿润")
    print(f"    湿度: 阈值={humid_th:.2f}%")

    # 气压: [正常, 偏低]  ← prs2021 50%
    press_s, press_th = quantile_cut_2class(df["prs2021"], "正常", "偏低")
    print(f"    气压: 阈值={press_th:.1f} hPa")

    # 蒸发量: [小, 大]  ← evp2021 50%
    evap_s, evap_th = quantile_cut_2class(df["evp2021"], "小", "大")
    print(f"    蒸发量: 阈值={evap_th:.0f} mm")

    # 海拔: [低, 高]  ← Dem_Mean 50%
    elev_s, elev_th = quantile_cut_2class(df["Dem_Mean"], "低", "高")
    print(f"    海拔: 阈值={elev_th:.1f} m")

    # 坡度: [缓, 陡]  ← Slope_Mean 50%
    slope_s, slope_th = quantile_cut_2class(df["Slope_Mean"], "缓", "陡")
    print(f"    坡度: 阈值={slope_th:.2f}°")

    # 坡向: [阴坡, 阳坡]  ← Aspect_Mea 180° 划分
    aspect_s = np.where(df["Aspect_Mea"] <= 180, "阴坡", "阳坡")
    print(f"    坡向: 180° 划分")

    # 地形起伏度: [小, 大]  ← Dem_Std 50%
    rugged_s, rugged_th = quantile_cut_2class(df["Dem_Std"], "小", "大")
    print(f"    地形起伏度: 阈值={rugged_th:.2f}")

    # 径流系数: [小, 大]  ← Impervious/10000 50% (代理)
    runoff_raw = df["Impervious"] / 10000.0
    runoff_s, runoff_th = quantile_cut_2class(runoff_raw, "小", "大")
    print(f"    径流系数: 阈值={runoff_th:.4f} (Impervious/10000 代理)")

    # 湖泊调蓄能力: [强, 弱]  ← WaterBody_ 50%
    lake_s, lake_th = quantile_cut_2class(df["WaterBody_"], "强", "弱")
    # WaterBody_ 中位数=0, 所以大部分=弱
    print(f"    湖泊调蓄能力: 阈值={lake_th:.1f} (WaterBody_ 代理)")

    # 历史排水时间: [快, 慢]  ← WorldPop_M 50% (代理)
    drain_time_s, drain_time_th = quantile_cut_2class(df["WorldPop_M"], "快", "慢")
    print(f"    历史排水时间: 阈值={drain_time_th:.4f} (WorldPop_M 代理)")

    # 管网排水能力: [强, 弱]  ← Impervious 反向 50% (代理)
    pipe_s, pipe_th = reverse_quantile_cut_2class(df["Impervious"], "强", "弱")
    print(f"    管网排水能力: 阈值={pipe_th:.0f} (Impervious 反向代理)")

    # 下垫面硬化率: [低, 中, 高]  ← Impervious/2500 33%/66%
    imperv_ratio = df["Impervious"] / 2500.0
    imperv_s, imperv_th = reverse_quantile_cut_3class(imperv_ratio, "低", "中", "高")
    print(f"    下垫面硬化率: 阈值={imperv_th[0]:.4f}/{imperv_th[1]:.4f}")

    # 建筑密度: [低, 中, 高]  ← Light_N 33%/66% (代理)
    # Light_N 有 -9999 无效值，先替换为 0
    light_clean = df["Light_N"].replace(-9999, 0)
    # 但有一些也是负值，取绝对值或 clip
    light_clean = light_clean.clip(lower=0)
    bldg_s, bldg_th = quantile_cut_3class(light_clean, "低", "中", "高")
    print(f"    建筑密度: 阈值={bldg_th[0]:.4f}/{bldg_th[1]:.4f} (Light_N 代理)")

    # 绿地率: [低, 中, 高]  ← (1-Impervious/2500) 33%/66% (代理)
    green_ratio = 1.0 - imperv_ratio
    green_s, green_th = quantile_cut_3class(green_ratio, "低", "中", "高")
    print(f"    绿地率: 阈值={green_th[0]:.4f}/{green_th[1]:.4f}")

    # 气温: [低温, 适温, 高温]  ← ERA5 域均值作为城市统一值
    # 所有网格使用同一 ERA5 域均值
    # 这里用分位数是城市尺度，不是网格尺度
    # 简单处理：所有网格统一值，不做空间离散化
    # 实际上 ERA5 只有一个值，所以
    # 需要定义阈值来三分
    # 使用 ERA5 域内不同网格的 t2m 值做三分
    t2m_grid = era5.t2m.isel(valid_time=jul20_idx).mean(dim="valid_time").values - 273.15
    t2m_flat = t2m_grid.flatten()
    t2m_q33 = np.percentile(t2m_flat, 33)
    t2m_q66 = np.percentile(t2m_flat, 66)
    print(f"    气温: ERA5 域内阈值={t2m_q33:.1f}/{t2m_q66:.1f}°C")

    # 露点温度: [低, 高]  ← ERA5 d2m 域均值 50%
    d2m_grid = era5.d2m.isel(valid_time=jul20_idx).mean(dim="valid_time").values - 273.15
    d2m_flat = d2m_grid.flatten()
    d2m_th = np.median(d2m_flat)
    print(f"    露点温度: 阈值={d2m_th:.1f}°C")

    # 前期土壤含水量: [低, 中, 高]  ← ERA5 swvl1 7/19 域内 33%/66%
    swvl1_grid = era5.swvl1.isel(valid_time=jul19_idx).values
    swvl1_flat = swvl1_grid.flatten()
    swvl1_q33 = np.percentile(swvl1_flat, 33)
    swvl1_q66 = np.percentile(swvl1_flat, 66)
    print(f"    前期土壤含水量: ERA5 阈值={swvl1_q33:.4f}/{swvl1_q66:.4f}")

    # 土壤渗透性: [好, 差]  ← ERA5 swvl2 域均值 50% (反向)
    swvl2_grid = era5.swvl2.isel(valid_time=jul20_idx).mean(dim="valid_time").values
    swvl2_flat = swvl2_grid.flatten()
    swvl2_th = np.median(swvl2_flat)
    print(f"    土壤渗透性: ERA5 阈值={swvl2_th:.4f} (反向)")

    # ── 4. 构建样本 DataFrame ──
    print("\n[4] 构建样本集...")

    # 只保留有 Label 的样本
    labeled = df[df["Label"].notna()].copy()
    print(f"    有标签样本: {len(labeled)} 行")

    # 标签映射：Y→1（内涝），N→0（无内涝）
    labeled["flood_label"] = (labeled["Label"] == "Y").astype(int)

    # ── 构建各节点状态列 ──
    # 气象 - 直接映射
    labeled["降水强度"] = rain_s[labeled.index]
    # 降水时长：ERA5 域均值，全样本统一
    labeled["降水时长"] = duration_state
    # 风力：ERA5 域均值
    wind_state = "强" if wind_speed_mean >= np.median(wind_speed) else "弱"
    labeled["风力"] = wind_state
    # 风向：保持先验（郑州内陆，无向岸/离岸概念）
    labeled["风向"] = "保持先验"
    labeled["湿度"] = humid_s[labeled.index]
    labeled["气压"] = press_s[labeled.index]
    # 气温：ERA5 各网格值
    # 每个网格的最近 ERA5 格点气温
    # 简化：使用 ERA5 域内所有网格的平均值，所有 CSV 网格统一
    # 但更精确的做法是空间匹配... 由于 CSV 无经纬度，使用域均值
    t2m_avg = float(np.mean(t2m_flat))
    if t2m_avg <= t2m_q33:
        labeled["气温"] = "低温"
    elif t2m_avg <= t2m_q66:
        labeled["气温"] = "适温"
    else:
        labeled["气温"] = "高温"
    # 露点温度
    d2m_avg = float(np.mean(d2m_flat))
    labeled["露点温度"] = "高" if d2m_avg > d2m_th else "低"
    labeled["对流有效位能CAPE"] = "保持先验"
    labeled["垂直风切变"] = "保持先验"

    # 水文
    swvl1_avg = float(np.mean(swvl1_flat))
    if swvl1_avg <= swvl1_q33:
        labeled["前期土壤含水量"] = "低"
    elif swvl1_avg <= swvl1_q66:
        labeled["前期土壤含水量"] = "中"
    else:
        labeled["前期土壤含水量"] = "高"
    labeled["径流系数"] = runoff_s[labeled.index]
    labeled["河道水位"] = "保持先验"
    labeled["地下水埋深"] = "保持先验"
    labeled["湖泊调蓄能力"] = lake_s[labeled.index]
    labeled["潮汐影响"] = "保持先验"
    labeled["蒸发量"] = evap_s[labeled.index]

    # 地形
    labeled["海拔"] = elev_s[labeled.index]
    labeled["坡度"] = slope_s[labeled.index]
    labeled["坡向"] = "阴坡"  # 域均值：Aspect_Mea 均值约 164°，≤180°→阴坡
    # 更精确：按每个网格的 Aspect_Mea
    labeled["坡向"] = aspect_s[labeled.index]
    labeled["地形起伏度"] = rugged_s[labeled.index]
    labeled["汇流面积"] = "保持先验"
    labeled["河网密度"] = "保持先验"
    labeled["地表粗糙度"] = "保持先验"
    labeled["沟谷密度"] = "保持先验"

    # 地质
    # 植被覆盖：保持先验（无可靠代理）
    # 注：pre2021 年降水不应用作植被代理（循环论证）
    labeled["植被覆盖"] = "保持先验"
    swvl2_avg = float(np.mean(swvl2_flat))
    labeled["土壤渗透性"] = "好" if swvl2_avg <= swvl2_th else "差"  # 反向
    labeled["岩性"] = "保持先验"
    labeled["距断层距离"] = "保持先验"
    labeled["地震烈度"] = "保持先验"
    labeled["风化程度"] = "保持先验"
    labeled["节理发育程度"] = "保持先验"
    labeled["滑坡历史密度"] = "保持先验"

    # 城市
    labeled["历史排水时间"] = drain_time_s[labeled.index]
    labeled["管网排水能力"] = pipe_s[labeled.index]
    labeled["下垫面硬化率"] = imperv_s[labeled.index]
    labeled["道路积水历史频率"] = "保持先验"
    labeled["建筑密度"] = bldg_s[labeled.index]
    labeled["绿地率"] = green_s[labeled.index]
    labeled["应急排水能力"] = "保持先验"

    # ── 5. 计算证据覆盖度 ──
    node_names = [
        "降水强度", "降水时长", "风力", "风向", "湿度", "气压", "气温", "露点温度",
        "对流有效位能CAPE", "垂直风切变",
        "前期土壤含水量", "径流系数", "河道水位", "地下水埋深", "湖泊调蓄能力",
        "潮汐影响", "蒸发量",
        "海拔", "坡度", "坡向", "地形起伏度", "汇流面积", "河网密度",
        "地表粗糙度", "沟谷密度",
        "植被覆盖", "土壤渗透性", "岩性", "距断层距离", "地震烈度",
        "风化程度", "节理发育程度", "滑坡历史密度",
        "历史排水时间", "管网排水能力", "下垫面硬化率", "道路积水历史频率",
        "建筑密度", "绿地率", "应急排水能力",
    ]

    def calc_coverage(row):
        provided = sum(1 for n in node_names if row.get(n) not in (None, "保持先验"))
        return provided / len(node_names)

    labeled["证据覆盖度"] = labeled.apply(calc_coverage, axis=1)

    # ── 6. 代理置信度 ──
    # 高置信度：直接映射的字段
    # 低置信度：通过代理映射的字段
    proxy_map = {
        "历史排水时间": "低",  # WorldPop_M 代理
        "管网排水能力": "低",  # Impervious 反向代理
        "建筑密度": "低",      # Light_N 代理
        "绿地率": "低",        # Impervious 反向代理
        "径流系数": "低",      # Impervious 代理
        "下垫面硬化率": "低",  # Impervious 代理
        "湖泊调蓄能力": "低",  # WaterBody_ 代理
    }
    # 为每个样本计算有多少低置信代理
    labeled["代理置信度"] = "高"  # 默认
    # 如果有低置信代理被启用，标记为"低"
    for n, conf in proxy_map.items():
        if conf == "低":
            # 任意一个低置信代理被使用，整行标记为"低"
            pass
    # 更精确：逐节点标注
    # 改为在样本中标注
    labeled["低置信代理数"] = 0
    for n, conf in proxy_map.items():
        if labeled[n].notna().any() and conf == "低":
            labeled["低置信代理数"] += 1
    labeled["代理置信度"] = labeled["低置信代理数"].apply(lambda x: "低" if x > 0 else "高")

    # ── 7. 输出 ──
    output_cols = node_names + ["flood_label", "证据覆盖度", "代理置信度"]
    out_path = os.path.join(OUTPUT_DIR, "samples_zhengzhou.csv")
    labeled[output_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[5] 输出: {out_path}")
    print(f"    样本数: {len(labeled)}")
    print(f"    正样本(Y): {int((labeled['flood_label']==1).sum())}, "
          f"负样本(N): {int((labeled['flood_label']==0).sum())}")
    print(f"    平均证据覆盖度: {labeled['证据覆盖度'].mean()*100:.1f}%")
    print(f"    低置信代理样本: {int((labeled['代理置信度']=='低').sum())} / {len(labeled)}")

    # ── 8. 打印各节点状态分布 ──
    print("\n[6] 各节点状态分布:")
    for n in node_names:
        vc = labeled[n].value_counts()
        vc_str = ", ".join(f"{k}={v}" for k, v in vc.items())
        print(f"    {n:14s}: {vc_str}")

    # 关闭 ERA5
    era5.close()
    print("\n完成!")


if __name__ == "__main__":
    main()