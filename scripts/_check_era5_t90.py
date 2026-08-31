"""检查 T=84~99 的 ERA5 降水证据状态"""
import json

with open(r'H:\dev\disaster-data\zhengzhou_720\era5_hourly_evidence.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("T=84~99 降水强度证据状态:")
for T in range(84, 100):
    rec = data['records'][T]
    print(f"  T={T:3d} ({rec['datetime']}): 降水={rec['evidence']['降水强度']}")

print("\nT=60~66 降水强度证据状态:")
for T in range(60, 67):
    rec = data['records'][T]
    print(f"  T={T:3d} ({rec['datetime']}): 降水={rec['evidence']['降水强度']}")

print("\nT=55~66 降水强度证据状态:")
for T in range(55, 67):
    rec = data['records'][T]
    print(f"  T={T:3d} ({rec['datetime']}): 降水={rec['evidence']['降水强度']}")

print("\nT=115~125 降水强度证据状态:")
for T in range(115, 126):
    rec = data['records'][T]
    print(f"  T={T:3d} ({rec['datetime']}): 降水={rec['evidence']['降水强度']}")