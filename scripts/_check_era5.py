"""临时：检查 ERA5 数据中 T=60~120 的降水强度"""
import json

with open(r'H:\dev\disaster-data\zhengzhou_720\era5_hourly_evidence.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for T in [60, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 100, 110, 120]:
    rec = data['records'][T]
    state = rec['evidence']['降水强度']
    val = 0.1 if state == '低' else (0.8 if state == '中' else 2.5)
    print(f'T={T:3d} ({rec["datetime"]}): 降水={state}, tp_mm={val:.1f}')