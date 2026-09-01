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
import glob
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
# 图片模式配置
# ============================================================================

# 默认演示图片源（3 张图，按文件名关键词映射 task_type）
DEFAULT_IMAGE_DIR = r"H:\实习\综合演示\1-输入图片"

# 图片文件名关键词 → task_type 映射
IMAGE_KEYWORD_TASK = {
    "洪水": "flood",
    "水位尺": "water_level",
    "道路": "road",
    "滑坡": "landslide",
    "water": "water_level",
    "road": "road",
    "flood": "flood",
    "landslide": "landslide",
}

# 演示图片中的 task_type 标签（按排序后的文件名对应）
DEFAULT_IMAGE_TASKS = [
    ("1-洪水现场.jpg", "flood"),
    ("2-河道水位尺.jpg", "water_level"),
    ("3-道路路面.jpg", "road"),
    ("滑坡-01.jpg", "landslide"),
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


# ============================================================================
# 图片发送
# ============================================================================

def scan_image_dir(image_dir: str) -> list:
    """
    扫描目录，按文件名排序，返回 [(full_path, task_type), ...]。
    自动根据文件名关键词推断 task_type，无法推断的跳过。
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith(valid_exts):
            # 根据文件名关键词推断 task_type
            matched_task = None
            for keyword, task in IMAGE_KEYWORD_TASK.items():
                if keyword in fname:
                    matched_task = task
                    break
            if matched_task is None:
                # 默认 fallback：按轮换分配
                matched_task = "flood"
            full_path = os.path.join(image_dir, fname)
            image_files.append((full_path, matched_task, fname))
    return image_files


def send_image(url: str, image_path: str, station: str, task_type: str) -> bool:
    """以 multipart/form-data 上传一张图片到 ingest_server"""
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"station": station, "task_type": task_type}
            resp = requests.post(url, files=files, data=data, timeout=30)
            if resp.status_code == 200:
                return True
            else:
                print(f"  ⚠️ 图片上传失败: HTTP {resp.status_code} {resp.text}")
                return False
    except requests.exceptions.ConnectionError:
        print(f"  ❌ 连接失败: {url}，请确保 ingest_server 已启动")
        return False
    except FileNotFoundError:
        print(f"  ❌ 图片文件不存在: {image_path}")
        return False
    except Exception as e:
        print(f"  ❌ 图片上传异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="模拟数据流发包程序")
    parser.add_argument("--mode", type=str, default="era5",
                        choices=["era5", "random", "image"],
                        help="数据模式：era5（回放）/ random（随机波动）/ image（图片上传）")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="发送间隔（秒/条，默认 2；图片模式默认 15）")
    parser.add_argument("--count", type=int, default=0,
                        help="发送条数（0=无限循环）")
    parser.add_argument("--port", type=int, default=8502,
                        help="ingest_server 端口（默认 8502）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="ingest_server 地址（默认 127.0.0.1）")
    parser.add_argument("--station", type=str, default=None,
                        help="指定站点（默认轮流使用郑州 12 区）")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="图片模式：图片目录路径（默认使用演示图片）")
    args = parser.parse_args()

    # 图片模式默认 15 秒间隔
    if args.mode == "image" and args.interval == 2.0:
        args.interval = 15.0

    url = f"http://{args.host}:{args.port}/api/ingest"
    image_url = f"http://{args.host}:{args.port}/api/ingest_image"
    print(f"📤 模拟数据流发包程序")
    print(f"   模式: {args.mode}")
    print(f"   间隔: {args.interval} 秒/条")
    print(f"   目标: {url}")
    print(f"   条数: {'无限' if args.count == 0 else args.count}")
    if args.mode == "image":
        image_dir = args.image_dir or DEFAULT_IMAGE_DIR
        print(f"   图片源: {image_dir}")
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

    # 图片模式：扫描图片目录
    image_tasks = []
    if args.mode == "image":
        image_dir = args.image_dir or DEFAULT_IMAGE_DIR
        if not os.path.exists(image_dir):
            print(f"   ❌ 图片目录不存在: {image_dir}")
            sys.exit(1)
        image_tasks = scan_image_dir(image_dir)
        if not image_tasks:
            # 使用默认演示配置
            for img_name, task in DEFAULT_IMAGE_TASKS:
                full_path = os.path.join(image_dir, img_name)
                if os.path.exists(full_path):
                    image_tasks.append((full_path, task, img_name))
        print(f"   ✅ 已扫描图片: {len(image_tasks)} 张")
        for fp, tk, fn in image_tasks:
            print(f"       - {fn} → {tk}")

    # 确定站点列表
    stations = [args.station] if args.station else ZHENGZHOU_DISTRICTS

    # 发送循环
    sent = 0
    era5_idx = 0
    image_idx = 0
    while True:
        if args.count > 0 and sent >= args.count:
            print(f"\n✅ 已发送 {args.count} 条，任务完成")
            break

        if args.mode == "image" and image_tasks:
            # 图片模式：按 station 轮换，每张图对应正确 task_type
            task_idx = image_idx % len(image_tasks)
            image_path, task_type, fname = image_tasks[task_idx]
            station = stations[sent % len(stations)]

            print(f"  已发送图片 {sent + 1}/{args.count or '∞'}: {station} {task_type} {fname}…", end="")
            ok = send_image(image_url, image_path, f"郑州-{station}", task_type)
            print(" ✅" if ok else " ❌")
            image_idx += 1
        else:
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