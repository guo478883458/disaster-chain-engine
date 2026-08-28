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
from datetime import datetime
from threading import Lock
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── 确保能找到 v2 模块 ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 内存队列（零依赖，如需持久化换 Redis Stream） ──
_data_queue: queue.Queue = queue.Queue()
_data_lock = Lock()
_received_count: int = 0
_last_data: Optional[dict] = None
_last_time: Optional[str] = None

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


@app.get("/api/status")
async def status():
    """返回接收状态摘要"""
    with _data_lock:
        return {
            "received_count": _received_count,
            "last_data": _last_data,
            "last_received_time": _last_time,
            "queue_size": _data_queue.qsize(),
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
    print(f"   POST /api/ingest  — 接收传感器数据")
    print(f"   GET  /api/status  — 查询接收状态")
    print(f"   GET  /api/health  — 健康检查")
    print(f"   内存队列模式，数据不持久化（如需持久化 → Redis Stream）")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()