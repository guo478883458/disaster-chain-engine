"""
构建 ERA5 逐小时气象证据时间序列（实时监测回放模拟器数据源）
================================================================
读取 ERA5-Land 郑州 7·20 逐小时 NetCDF（144 步），按每时间步离散化为 BN 气象节点证据，
输出 JSON 供 dashboard"实时监测"页回放使用。

离散化规则沿用 tools/prepare_validation_data.py 已实现的映射逻辑。

输出: H:\dev\disaster-data\zhengzhou_720\era5_hourly_evidence.json
"""
import json, os, shutil, sys, warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ── 路径 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC_PATH = os.path.join(
    ROOT, "data", "era5land", "era5land_zhengzhou_20210718-0723.nc"
)
OUTPUT_DIR = r"H:\dev\disaster-data\zhengzhou_720"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "era5_hourly_evidence.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 工具函数：复制到 ASCII 短路径再读取（解决 netCDF4 Unicode 路径问题）
# ══════════════════════════════════════════════════════════════════════

def load_era5(nc_path):
    """复制到 D 盘根目录再读取，避免 Unicode 路径问题"""
    if not os.path.exists(nc_path):
        # 尝试直接从 H 盘读取
        h_path = r"H:\dev\disaster-data\era5land\era5land_zhengzhou_20210718-0723.nc"
        nc_path = h_path if os.path.exists(h_path) else nc_path
    dst = r"D:\_era5_timeseries_temp.nc"
    shutil.copy2(nc_path, dst)
    with xr.open_dataset(dst) as ds:
        ds.load()
        data = ds.copy()
    if os.path.exists(dst):
        os.remove(dst)
    return data


# ══════════════════════════════════════════════════════════════════════
# 离散化函数（与 prepare_validation_data.py 一致）
# ══════════════════════════════════════════════════════════════════════

def quantile_cut_2class(values, state_low, state_high):
    """二分法离散化：低于中位数=state_low，高于中位数=state_high"""
    thresh = np.median(values)
    result = np.where(values <= thresh, state_low, state_high)
    return result, thresh


def quantile_cut_3class(values, state_low, state_mid, state_high):
    """三分法离散化：33%/66% 分位数"""
    q33 = np.percentile(values, 33)
    q66 = np.percentile(values, 66)
    result = np.where(values <= q33, state_low,
                      np.where(values <= q66, state_mid, state_high))
    return result, (q33, q66)


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("ERA5 逐小时证据时间序列构建")
    print("=" * 60)

    # ── 1. 读取 ERA5 ──
    print("\n[1] 读取 ERA5-Land 数据...")
    era5 = load_era5(NC_PATH)
    n_steps = len(era5.valid_time)
    print(f"    时间步数: {n_steps}")
    print(f"    时间范围: {era5.valid_time.values[0]} ~ {era5.valid_time.values[-1]}")
    print(f"    网格: {len(era5.latitude)}×{len(era5.longitude)}")
    print(f"    变量: {list(era5.data_vars.keys())}")

    # 域平均函数
    def domain_mean(var, time_idx):
        return float(var.isel(valid_time=time_idx).mean(dim=["latitude", "longitude"]).values)

    # 提取所有时间步的域均值，用于计算分位数
    print("\n[2] 计算全时段域均值序列（用于分位数阈值）...")

    tp_hourly = []       # 逐小时降水 mm
    wind_speeds = []     # 风速 m/s
    t2m_vals = []        # 气温 °C
    d2m_vals = []        # 露点温度 °C
    swvl1_vals = []      # 土壤含水量
    swvl2_vals = []      # 土壤含水量（深层）
    evabs_vals = []      # 蒸发量

    for t in range(n_steps):
        # tp 差分（累积量→逐小时）
        tp_t = era5.tp.isel(valid_time=t).values  # 米
        if t == 0:
            tp_diff = tp_t  # 第 0 小时假设从 0 开始
        else:
            tp_prev = era5.tp.isel(valid_time=t - 1).values
            tp_diff = tp_t - tp_prev
            # 防止负值（ERA5 累积量偶尔重置）
            tp_diff = np.clip(tp_diff, 0, None)
        tp_mm = float(tp_diff.mean() * 1000.0)  # 米→mm，域平均
        tp_hourly.append(tp_mm)

        # 风速
        u = era5.u10.isel(valid_time=t).values
        v = era5.v10.isel(valid_time=t).values
        ws = float(np.sqrt(u**2 + v**2).mean())
        wind_speeds.append(ws)

        # 气温
        t2m = float(era5.t2m.isel(valid_time=t).mean(dim=["latitude", "longitude"]).values - 273.15)
        t2m_vals.append(t2m)

        # 露点温度
        d2m = float(era5.d2m.isel(valid_time=t).mean(dim=["latitude", "longitude"]).values - 273.15)
        d2m_vals.append(d2m)

        # 土壤含水量
        swvl1 = domain_mean(era5.swvl1, t)
        swvl2 = domain_mean(era5.swvl2, t)
        swvl1_vals.append(swvl1)
        swvl2_vals.append(swvl2)

        # 蒸发量
        evabs = domain_mean(era5.evabs, t)
        evabs_vals.append(evabs)

    tp_hourly = np.array(tp_hourly)
    wind_speeds = np.array(wind_speeds)
    t2m_vals = np.array(t2m_vals)
    d2m_vals = np.array(d2m_vals)
    swvl1_vals = np.array(swvl1_vals)
    swvl2_vals = np.array(swvl2_vals)

    # ── 3. 计算分位数阈值（全时段） ──
    print("\n[3] 计算离散化阈值...")

    # 降水强度：固定阈值（≥5mm/h=高, 1~5=中, <1=低）
    # 沿用 prepare_validation_data.py 的 quantile_cut_3class 逻辑
    # 但使用固定阈值更直观
    rain_cut, rain_th = quantile_cut_3class(tp_hourly, "低", "中", "高")
    print(f"    降水强度: 阈值={rain_th[0]:.2f}/{rain_th[1]:.2f} mm/h (全时段 33%/66% 分位)")

    # 降水时长：滑动窗口 ≥5mm/h 累计小时数
    # 用 3 小时滑动窗口
    def _calc_duration(hour_idx, window=3):
        start = max(0, hour_idx - window + 1)
        count = int(np.sum(tp_hourly[start:hour_idx + 1] >= 5.0))
        return "长" if count >= 2 else "短"

    # 风力（50% 分位）
    wind_cut, wind_th = quantile_cut_2class(wind_speeds, "弱", "强")
    print(f"    风力: 阈值={wind_th:.2f} m/s")

    # 气温（33%/66% 分位）
    t2m_cut, t2m_th = quantile_cut_3class(t2m_vals, "低温", "适温", "高温")
    print(f"    气温: 阈值={t2m_th[0]:.1f}/{t2m_th[1]:.1f} °C")

    # 露点温度（50% 分位）
    d2m_cut, d2m_th = quantile_cut_2class(d2m_vals, "低", "高")
    print(f"    露点温度: 阈值={d2m_th:.1f} °C")

    # 前期土壤含水量（33%/66% 分位）
    swvl1_cut, swvl1_th = quantile_cut_3class(swvl1_vals, "低", "中", "高")
    print(f"    前期土壤含水量: 阈值={swvl1_th[0]:.4f}/{swvl1_th[1]:.4f}")

    # 土壤渗透性（50% 分位，反向：swvl2 高→渗透性差）
    swvl2_cut, swvl2_th = quantile_cut_2class(swvl2_vals, "好", "差")
    # 注意：swvl2 高 = 含水多 = 渗透性差，所以高值→"差"
    # quantile_cut_2class: 低于中位数→state_low, 高于→state_high
    # swvl2 低→渗透性好, 高→渗透性差, 所以直接使用
    print(f"    土壤渗透性: 阈值={swvl2_th:.4f} (swvl2 反向)")

    # 蒸发量（50% 分位）
    evabs_cut, evabs_th = quantile_cut_2class(evabs_vals, "小", "大")
    print(f"    蒸发量: 阈值={evabs_th:.4f}")

    # ── 4. 构建逐小时证据 ──
    print("\n[4] 构建 144 小时证据序列...")

    # 湿度/气压/风向 保持先验（ERA5 无直接对应变量或暂不纳入）
    # 对流有效位能CAPE/垂直风切变 保持先验

    records = []
    for t in range(n_steps):
        dt = era5.valid_time.values[t]
        dt_str = str(dt)[:16]  # "2021-07-18T00:00"

        # 降水强度
        rain_state = rain_cut[t]

        # 降水时长
        duration_state = _calc_duration(t)

        # 风力/风向
        wind_state = wind_cut[t]

        # 气温
        t2m_state = t2m_cut[t]

        # 露点温度
        d2m_state = d2m_cut[t]

        # 前期土壤含水量
        swvl1_state = swvl1_cut[t]

        # 土壤渗透性
        swvl2_state = swvl2_cut[t]

        # 蒸发量
        evabs_state = evabs_cut[t]

        evidence = {
            "降水强度": rain_state,
            "降水时长": duration_state,
            "风力": wind_state,
            # 风向：郑州内陆，无向岸/离岸概念，保持先验
            "风向": "保持先验",
            # 湿度/气压：ERA5 无直接对应，暂保持先验
            "湿度": "保持先验",
            "气压": "保持先验",
            "气温": t2m_state,
            "露点温度": d2m_state,
            # CAPE/风切变：保持先验
            "对流有效位能CAPE": "保持先验",
            "垂直风切变": "保持先验",
            # 水文
            "前期土壤含水量": swvl1_state,
            "土壤渗透性": swvl2_state,
            "蒸发量": evabs_state,
            # 其余水文节点保持先验（河道水位/地下水埋深等由非气象因素决定）
            "径流系数": "保持先验",
            "河道水位": "保持先验",
            "地下水埋深": "保持先验",
            "湖泊调蓄能力": "保持先验",
            "潮汐影响": "保持先验",
        }

        records.append({
            "hour": t,
            "datetime": dt_str,
            "evidence": evidence,
        })

    # ── 5. 输出 ──
    print(f"\n[5] 输出: {OUTPUT_PATH}")
    output = {
        "meta": {
            "source": "era5land_zhengzhou_20210718-0723.nc",
            "n_steps": n_steps,
            "variables": list(era5.data_vars.keys()),
            "thresholds": {
                "降水强度": {"method": "33%/66% quantile", "thresh": [float(rain_th[0]), float(rain_th[1])]},
                "风力": {"method": "median", "thresh": float(wind_th)},
                "气温": {"method": "33%/66% quantile", "thresh": [float(t2m_th[0]), float(t2m_th[1])]},
                "露点温度": {"method": "median", "thresh": float(d2m_th)},
                "前期土壤含水量": {"method": "33%/66% quantile", "thresh": [float(swvl1_th[0]), float(swvl1_th[1])]},
                "土壤渗透性": {"method": "median (reverse)", "thresh": float(swvl2_th)},
                "蒸发量": {"method": "median", "thresh": float(evabs_th)},
            },
        },
        "records": records,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"    写入 {len(records)} 条记录")
    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"    文件大小: {file_size:,} 字节")

    # ── 6. 抽查验证 ──
    print("\n" + "=" * 60)
    print("抽查验证")
    print("=" * 60)

    check_hours = [0, 47, 53, 66, 100]  # 7/18 00:00, 7/19 23:00, 7/20 05:00, 7/20 18:00, 7/21 04:00
    check_labels = ["起始", "雨前", "峰值附近", "峰值后", "恢复期"]
    # 7/20 暴雨峰值在 14:00~18:00 之间，对应 hour 62~66

    for h, label in zip(check_hours, check_labels):
        if h < len(records):
            rec = records[h]
            ev = rec["evidence"]
            print(f"\n  hour={h:3d} ({rec['datetime']}) [{label}]")
            print(f"    降水强度={ev['降水强度']}, 降水时长={ev['降水时长']}, "
                  f"风力={ev['风力']}, 气温={ev['气温']}")
            print(f"    露点温度={ev['露点温度']}, 前期土壤含水量={ev['前期土壤含水量']}, "
                  f"土壤渗透性={ev['土壤渗透性']}, 蒸发量={ev['蒸发量']}")

    # 打印 7/20 当天降水强度统计
    jul20_start = 48  # 7/20 00:00
    jul20_end = 71    # 7/20 23:00
    jul20_rain = [r["evidence"]["降水强度"] for r in records[jul20_start:jul20_end + 1]]
    high_count = jul20_rain.count("高")
    mid_count = jul20_rain.count("中")
    print(f"\n  7/20 当天降水强度分布: 高={high_count}h, 中={mid_count}h, "
          f"低={24 - high_count - mid_count}h")

    era5.close()
    print("\n完成!")


if __name__ == "__main__":
    main()