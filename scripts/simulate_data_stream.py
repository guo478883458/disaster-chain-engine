"""
模拟数据流发包程序
===================
模拟"传感器/数据源持续上报"，按设定间隔向 ingest_server 发送气象/水位数据。

两种模式（--mode）：
  era5：按顺序回放 ERA5 144 小时序列（复用 era5_hourly_evidence.json 的 evidence）
  random：随机波动数据，模拟真实传感器噪声

用法：
  python scripts/simulate_data_stream.py
  python scripts/simulate_data_stream.py --mode random --interval 1 --count 50
  python scripts/simulate_data_stream.py --mode era5 --interval 3 --count 144
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta

import requests

# ── 确保能找到 v2 模块 ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# 气象节点状态集（与 BN 定义一致）
# ============================================================================

METEOR_NODES = {
    "降水强度": ["低", "中", "高"],
    "降水时长": ["短", "中", "长"],
    "风力": ["弱", "中", "强"],
    "气温": ["低温", "适温", "高温"],
    "湿度": ["干燥", "适中", "湿润"],
    "气压": ["偏高", "正常", "偏低"],
    "前期土壤含水量": ["低", "中", "高"],
    "河道水位": ["正常", "警戒", "危险"],
    "径流系数": ["小", "中", "大"],
    "蒸发量": ["小", "中", "大"],
}

# 郑州 12 区
ZHENGZHOU_DISTRICTS = [
    "中原区", "二七区", "管城回族区", "金水区",
    "上街区", "惠济区", "中牟县", "巩义市",
    "荥阳市", "新密市", "新郑市", "登封市",
]


# ============================================================================
# 数据生成
# ============================================================================

def load_era5_evidence() -> list:
    """加载 ERA5 逐小时证据序列"""
    # 尝试多个路径
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "data", "zhengzhou_720", "era5_hourly_evidence.json"),
        r"H:\dev\disaster-data\zhengzhou_720\era5_hourly_evidence.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("records", [])
    raise FileNotFoundError("era5_hourly_evidence.json 未找到，请先运行 build_era5_timeseries.py")


def generate_era5_payload(records: list, idx: int, station: str) -> dict:
    """从 ERA5 记录生成一条数据"""
    rec = records[idx % len(records)]
    ev = rec.get("evidence", {})
    timestamp = (datetime.now() - timedelta(seconds=len(records) - idx)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": timestamp,
        "station": station,
        **ev,
    }


def generate_random_payload(station: str) -> dict:
    """生成随机波动数据（模拟真实传感器噪声）"""
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "station": station,
    }
    for node, states in METEOR_NODES.items():
        # 加权随机，使状态有一定连续性（倾向于当前状态）
        payload[node] = random.choices(states, weights=[0.3, 0.4, 0.3])[0]
    return payload


# ============================================================================
# 发送
# ============================================================================

def send_payload(url: str, payload: dict) -> bool:
    """发送一条数据到 ingest_server"""
    try:
        resp = requests.post(url, json=payload, timeout=3)
        if resp.status_code == 200:
            return True
        else:
            print(f"  ⚠️ 发送失败: HTTP {resp.status_code} {resp.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ❌ 连接失败: {url}，请确保 ingest_server 已启动")
        return False
    except Exception as e:
        print(f"  ❌ 发送异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="模拟数据流发包程序")
    parser.add_argument("--mode", type=str, default="era5",
                        choices=["era5", "random"],
                        help="数据模式：era5（回放）/ random（随机波动）")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="发送间隔（秒/条，默认 2）")
    parser.add_argument("--count", type=int, default=0,
                        help="发送条数（0=无限循环）")
    parser.add_argument("--port", type=int, default=8502,
                        help="ingest_server 端口（默认 8502）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="ingest_server 地址（默认 127.0.0.1）")
    parser.add_argument("--station", type=str, default=None,
                        help="指定站点（默认轮流使用郑州 12 区）")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/api/ingest"
    print(f"📤 模拟数据流发包程序")
    print(f"   模式: {args.mode}")
    print(f"   间隔: {args.interval} 秒/条")
    print(f"   目标: {url}")
    print(f"   条数: {'无限' if args.count == 0 else args.count}")
    print()

    # 加载 ERA5 数据（era5 模式需要）
    era5_records = []
    if args.mode == "era5":
        try:
            era5_records = load_era5_evidence()
            print(f"   ✅ 已加载 ERA5 序列: {len(era5_records)} 条记录")
        except FileNotFoundError as e:
            print(f"   ❌ {e}")
            sys.exit(1)

    # 确定站点列表
    stations = [args.station] if args.station else ZHENGZHOU_DISTRICTS

    # 发送循环
    sent = 0
    era5_idx = 0
    while True:
        if args.count > 0 and sent >= args.count:
            print(f"\n✅ 已发送 {args.count} 条，任务完成")
            break

        station = stations[sent % len(stations)]

        if args.mode == "era5" and era5_records:
            payload = generate_era5_payload(era5_records, era5_idx, station)
            rec_num = era5_idx + 1
            total = len(era5_records)
            rain = payload.get("降水强度", "?")
            print(f"  已发送 {sent + 1}/{args.count or '∞'} (ERA5 {rec_num}/{total}): "
                  f"{station} 降水={rain}…", end="")
            era5_idx = (era5_idx + 1) % len(era5_records)
        else:
            payload = generate_random_payload(station)
            rain = payload.get("降水强度", "?")
            print(f"  已发送 {sent + 1}/{args.count or '∞'}: {station} 降水={rain}…", end="")

        ok = send_payload(url, payload)
        print(" ✅" if ok else " ❌")
        sent += 1

        if args.interval > 0:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()