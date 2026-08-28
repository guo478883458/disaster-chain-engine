"""
数据流接入服务（FastAPI 接收端）
================================
接收模拟/真实传感器上报的气象/水位数据，写入内存队列供 dashboard 轮询。

启动：python scripts/ingest_server.py --port 8502

真实接入时只需改数据格式/来源，接收端不变。
如需持久化，将 queue.Queue 替换为 Redis Stream（代码中已标注 TODO 点）。
"""
import argparse
import json
import os
import queue
import sys
import time
import uuid
from datetime import datetime
from threading import Lock
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

# ── 确保能找到 v2 模块 ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 内存队列（零依赖，如需持久化换 Redis Stream） ──
_data_queue: queue.Queue = queue.Queue()
_data_lock = Lock()
_received_count: int = 0
_last_data: Optional[dict] = None
_last_time: Optional[str] = None

# ── 图片独立队列（避免相互阻塞） ──
_image_queue: queue.Queue = queue.Queue()
_image_lock = Lock()
_image_received_count: int = 0
_last_image: Optional[dict] = None
_last_image_time: Optional[str] = None

# ── 图片存储根目录 ──
STREAM_UPLOADS_ROOT = r"H:\dev\disaster-data\stream_uploads"
os.makedirs(STREAM_UPLOADS_ROOT, exist_ok=True)

# ── 允许的 task_type ──
ALLOWED_TASK_TYPES = ("water_level", "road", "flood")

app = FastAPI(title="灾害链数据流接入服务", version="1.0.0")


# ═══════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════

class SensorData(BaseModel):
    """传感器上报数据模型

    字段说明：
      - timestamp: 数据生成时间（ISO 格式 "2026-08-28 10:00:00"）
      - station: 站点标识，如 "郑州-金水区"
      - 其余字段为气象/水文 BN 节点状态（如 降水强度/风力/河道水位 等）
    """
    timestamp: str = Field(..., description="数据生成时间")
    station: str = Field(..., description="站点标识")
    # 允许额外字段（气象/水文节点状态）
    class Config:
        extra = "allow"


# ═══════════════════════════════════════════════════════════════════
# 接口
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/ingest")
async def ingest(data: SensorData):
    """接收传感器上报数据，写入内存队列"""
    global _received_count, _last_data, _last_time

    # 校验必要字段
    if not data.timestamp or not data.station:
        raise HTTPException(status_code=400, detail="timestamp 和 station 为必填字段")

    payload = data.model_dump()

    with _data_lock:
        _data_queue.put(payload)
        _received_count += 1
        _last_data = payload
        _last_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    return {"status": "ok", "received": _received_count}


# ═══════════════════════════════════════════════════════════════════
# 图片接收端点
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/ingest_image")
async def ingest_image(
    station: str = Form(..., description="区名，如 金水区"),
    task_type: str = Form(..., description="任务类型: water_level / road / flood"),
    file: UploadFile = File(..., description="灾情图片"),
):
    """
    接收图片上传，保存到磁盘，写入独立内存队列（image_queue）。

    校验：
      - task_type 限定 water_level/road/flood
      - 图片非空且可解码（cv2.imread 验证）
    """
    global _image_received_count, _last_image, _last_image_time

    # 校验 task_type
    if task_type not in ALLOWED_TASK_TYPES:
        raise HTTPException(status_code=400,
                            detail=f"task_type 必须为 {ALLOWED_TASK_TYPES}，收到: {task_type}")

    # 校验图片内容
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="图片内容为空")

    # cv2 校验图片是否可解码
    np_arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="图片无法解码，请确认是有效图像文件")

    # 构造保存路径
    station_dir = os.path.join(STREAM_UPLOADS_ROOT, station)
    os.makedirs(station_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    save_filename = f"{timestamp}_{task_type}_{uuid.uuid4().hex[:6]}{ext}"
    save_path = os.path.join(station_dir, save_filename)

    # 写入磁盘
    with open(save_path, "wb") as f:
        f.write(contents)

    # 构造记录
    image_record = {
        "station": station,
        "task_type": task_type,
        "filename": save_filename,
        "save_path": save_path,
        "original_filename": file.filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
    }

    with _image_lock:
        _image_queue.put(image_record)
        _image_received_count += 1
        _last_image = image_record
        _last_image_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    return {
        "status": "ok",
        "image_received_count": _image_received_count,
        "save_path": save_path,
        "station": station,
        "task_type": task_type,
    }


# ═══════════════════════════════════════════════════════════════════
# 状态
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def status():
    """返回接收状态摘要（含图片通道）"""
    with _data_lock, _image_lock:
        return {
            "received_count": _received_count,
            "last_data": _last_data,
            "last_received_time": _last_time,
            "queue_size": _data_queue.qsize(),
            "image_received_count": _image_received_count,
            "last_image": _last_image,
            "last_image_time": _last_image_time,
            "image_queue_size": _image_queue.qsize(),
        }


@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "time": datetime.now().isoformat()}


# ═══════════════════════════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="启动数据流接入服务")
    parser.add_argument("--port", type=int, default=8502, help="监听端口（默认 8502）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    args = parser.parse_args()

    print(f"📡 数据流接入服务启动: http://{args.host}:{args.port}")
    print(f"   POST /api/ingest       — 接收传感器数据")
    print(f"   POST /api/ingest_image — 接收灾情图片（multipart）")
    print(f"   GET  /api/status       — 查询接收状态（含图片通道）")
    print(f"   GET  /api/health       — 健康检查")
    print(f"   内存队列模式，数据不持久化（如需持久化 → Redis Stream）")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()