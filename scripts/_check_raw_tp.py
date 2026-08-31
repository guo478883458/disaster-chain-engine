"""检查 ERA5 原始 tp_mm 值（T=85~95）"""
import numpy as np
import xarray as xr

NC_PATH = r'H:\dev\disaster-data\era5land\era5land_zhengzhou_20210718-0723.nc'

ds = xr.open_dataset(NC_PATH)
n_steps = ds.dims['valid_time']

tp_hourly = []
for t in range(n_steps):
    tp_t = ds.tp.isel(valid_time=t).values
    if t == 0:
        tp_diff = tp_t
    else:
        tp_prev = ds.tp.isel(valid_time=t - 1).values
        tp_diff = tp_t - tp_prev
        tp_diff = np.clip(tp_diff, 0, None)
    tp_mm = float(tp_diff.mean() * 1000.0)
    tp_hourly.append(tp_mm)

tp_hourly = np.array(tp_hourly)

# 分位数
q33 = float(np.percentile(tp_hourly, 33))
q66 = float(np.percentile(tp_hourly, 66))
print(f"降水阈值: Q33={q33:.2f}mm/h, Q66={q66:.2f}mm/h")

print("\nT=55~66 原始 tp_mm 值:")
for T in range(55, 67):
    val = tp_hourly[T]
    dt = str(ds.valid_time.values[T])[:16]
    if val <= q33:
        state = "低"
    elif val <= q66:
        state = "中"
    else:
        state = "高"
    print(f"  T={T:3d} ({dt}): tp_mm={val:.3f}mm/h => {state}")

print("\nT=115~126 原始 tp_mm 值:")
for T in range(115, 127):
    val = tp_hourly[T]
    dt = str(ds.valid_time.values[T])[:16]
    if val <= q33:
        state = "低"
    elif val <= q66:
        state = "中"
    else:
        state = "高"
    print(f"  T={T:3d} ({dt}): tp_mm={val:.3f}mm/h => {state}")

ds.close()