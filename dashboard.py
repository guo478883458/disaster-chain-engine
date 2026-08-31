"""
灾害链推理引擎 v2 - 综合可视化 Dashboard（郑州全景可视化）
=========================================================
用法:
  streamlit run dashboard.py

功能:
  - 郑州地图可视化（各区风险着色 + 文字标注置信度 + hover 详情）
  - 视觉识别置信度透传（置信度水平条）
  - 灵活输入入口（参数修改 + 图片上传）
  - 六阶段结果总结（卡片/表格 + 证据来源标注）
  - 链式递进预测（时间轴推进 + 概率对比 + 递推规则展示）
  - 灾害类型结论模块（主要威胁 + 次生灾害 + 分级结论）
"""

import sys
import os
import json
import traceback as tb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import time
from datetime import datetime

# ── 确保能找到 v2 模块 ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bn_engine import DisasterChainEngine
from region_engine import RegionEngine

# ── 配置文件路径 ──
from path_config import GEOJSON_PATH, GEOJSON_SIMPLIFIED_PATH, ERA5_EVIDENCE_PATH, ERA5_RESULTS_CACHE_PATH, ZHENGZHOU_DIR

FORECAST_PATH = os.path.join(ZHENGZHOU_DIR, "era5_forecasts.json")

V2_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = V2_ROOT / "configs" / "config_40nodes.yaml"
REGIONS_BASE = V2_ROOT / "configs"

# ── 郑州 12 区县（以 GeoJSON 为准） ──
ZHENGZHOU_DISTRICTS = [
    "中原区", "二七区", "管城回族区", "金水区",
    "上街区", "惠济区", "中牟县", "巩义市",
    "荥阳市", "新密市", "新郑市", "登封市",
]

# 演示数据目录
DEMO_DIR = str(ZHENGZHOU_DIR)


# ============================================================================
# 缓存引擎（首屏加载优化）
# ============================================================================

@st.cache_resource(show_spinner="正在加载灾害链推理引擎…")
def load_engine(config_path):
    """缓存 BN 引擎实例，避免每次 rerun 重建"""
    return DisasterChainEngine(str(config_path))


@st.cache_resource(show_spinner="正在加载 GeoJSON 地图数据…")
def load_geojson():
    """缓存 GeoJSON 数据（优先加载简化版，回退到原始版）"""
    path = GEOJSON_SIMPLIFIED_PATH if os.path.exists(GEOJSON_SIMPLIFIED_PATH) else GEOJSON_PATH
    if not os.path.exists(path):
        st.error(f"GeoJSON 文件不存在: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="正在加载 ERA5 逐小时气象证据…")
def load_era5_evidence() -> dict:
    """加载 ERA5 逐小时气象证据时间序列（缓存，避免每次 rerun 重建）"""
    if not os.path.exists(ERA5_EVIDENCE_PATH):
        st.error(f"ERA5 证据文件不存在: {ERA5_EVIDENCE_PATH}\n"
                 f"请先运行 scripts/build_era5_timeseries.py")
        return {"meta": {}, "records": []}
    with open(ERA5_EVIDENCE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="正在加载预测数据…")
def load_era5_forecasts() -> dict:
    """加载 ERA5 预测结果"""
    if not os.path.exists(FORECAST_PATH):
        return {"meta": {"scheme": "未生成"}, "forecasts": {}}
    with open(FORECAST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# ERA5 预计算结果磁盘缓存
# ============================================================================

def _get_demo_params_hash() -> str:
    """计算 demo_params.json 的内容 hash，用于缓存失效检测"""
    import hashlib
    demo_path = os.path.join(os.path.dirname(__file__),
                             "configs", "郑州", "demo_params.json")
    if not os.path.exists(demo_path):
        return ""
    with open(demo_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _get_era5_evidence_hash() -> str:
    """计算 era5_hourly_evidence.json 的内容 hash，用于缓存失效检测"""
    import hashlib
    if not os.path.exists(ERA5_EVIDENCE_PATH):
        return ""
    with open(ERA5_EVIDENCE_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_era5_results_cache() -> dict:
    """从磁盘加载 ERA5 预计算结果缓存，含失效检测

    缓存命中条件（全部满足）：
      1. 缓存文件存在
      2. meta.demo_params_hash 与当前 demo_params.json 一致
      3. meta.evidence_hash 与当前 era5_hourly_evidence.json 一致

    返回: {"valid": True, "results": [...]} 或 {"valid": False, "results": []}
    """
    result = {"valid": False, "results": []}
    if not os.path.exists(ERA5_RESULTS_CACHE_PATH):
        return result

    try:
        with open(ERA5_RESULTS_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

        # 校验 hash
        demo_hash = _get_demo_params_hash()
        ev_hash = _get_era5_evidence_hash()
        meta = cache.get("meta", {})

        if meta.get("demo_params_hash") == demo_hash and meta.get("evidence_hash") == ev_hash:
            result["valid"] = True
            result["results"] = cache.get("results", [])
        return result
    except Exception:
        return result


def save_era5_results_cache(results: list):
    """将 ERA5 预计算结果写入磁盘缓存"""
    import hashlib
    demo_hash = _get_demo_params_hash()
    ev_hash = _get_era5_evidence_hash()

    cache = {
        "meta": {
            "demo_params_hash": demo_hash,
            "evidence_hash": ev_hash,
            "total_hours": len(results),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(ERA5_RESULTS_CACHE_PATH), exist_ok=True)
    with open(ERA5_RESULTS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── 模型权重懒加载（首屏不加载，仅在上传图片时加载） ──
@st.cache_resource
def _get_lazy_models():
    """惰性单例：仅在需要时初始化模型字典"""
    return {}


def _load_model_if_needed(task_type: str):
    """按需加载模型权重（首次调用时加载，后续复用缓存）"""
    models = _get_lazy_models()
    if task_type in models:
        return models[task_type]

    # 首次加载
    from tools.fuse_infer import _load_model
    model = _load_model(task_type)
    models[task_type] = model
    return model


# ============================================================================
# 数据流模式：轮询 ingest_server
# ============================================================================

INGEST_SERVER_URL = "http://127.0.0.1:8502"


def poll_ingest_server() -> dict:
    """轮询 ingest_server 的 /api/status，返回最新状态"""
    import requests
    try:
        resp = requests.get(f"{INGEST_SERVER_URL}/api/status", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"received_count": 0, "last_data": None, "last_received_time": None}


def fetch_ingest_data() -> list:
    """尝试从 ingest_server 获取新数据（通过轮询 /api/status 检测变化）"""
    status_data = poll_ingest_server()
    count = status_data.get("received_count", 0)
    last_data = status_data.get("last_data")
    last_time = status_data.get("last_received_time")
    return [count, last_data, last_time]


# ============================================================================
# 初始化 session_state
# ============================================================================

def init_session_state():
    """初始化所有 session_state 变量"""
    defaults = {
        # 页面导航
        "page": "zhengzhou",
        "zz_tab": "map",

        # 郑州模式
        "zz_evidence": {},             # {district: {param: state, ...}}
        "zz_results": {},              # {district: bn_result}
        "zz_confidence": {},           # {district: confidence_summary}
        "zz_uploaded": {},             # {district: {task: image_path}}
        "zz_recognition": {},          # {district: fuse_result}
        "zz_time_period": 0,           # 当前时段
        "zz_chain_evidence_history": [],  # 链式预测递推历史（仅影响链式模块）
        "zz_chain_results_history": [],   # 链式预测各时段结果
        "zz_chain_initial_evidence": {},  # 链式预测初始证据快照（用于重置）
        "zz_chain_initial_results": {},   # 链式预测初始结果快照
        "zz_fuse_infer_once": False,   # 是否已执行过 fuse_infer
        "zz_infer_source": None,      # 最近一次推演来源: "demo" / "prior" / "manual"

        # 引擎状态
        "_engine_loaded": False,

        # 地图缓存（避免相同结果下重建 fig）
        "_map_fig_cache": {},          # {cache_key: go.Figure}
        "_map_cache_version": 0,       # 每次推理结果变化时递增

        # ERA5 实时监测回放
        "_era5_hour": 0,               # 当前播放小时 (0~143)
        "_era5_playing": False,        # 是否正在自动播放
        "_era5_speed": 1.0,            # 播放速度（秒/步）
        "_era5_results_history": [],   # 各小时推理结果 [{hour, results}, ...]
        "_era5_initialized": False,    # 是否已初始化过 ERA5 推理
        "_era5_forecast_horizon": 0,   # 地图预测时段：0=当前, 1=+1h, 3=+3h

        # 数据流模式
        "_era5_stream_mode": "replay", # 实时监测模式: "replay" / "stream"
        "_era5_stream_last_count": 0,  # 上次轮询时的已接收条数
        "_era5_stream_received_data": [],  # 数据流模式已接收数据列表
        "_era5_stream_results_history": [],  # 数据流模式各点推理结果
        # 数据流模式 - 地图降频
        "_map_last_rebuild_time": 0.0,  # 上次地图重建时间戳（限频用）
        # 数据流模式 - 图片通道
        "_stream_image_last_count": 0,  # 上次轮询时的图片已接收条数
        "_stream_image_records": [],    # 已处理的图片记录 [{station, task_type, save_path, ...}]
        "_stream_image_processing": False,  # 是否正在处理图片识别
        # 数据流模式 - 趋势图降频
        "_stream_trend_last_len": 0,    # 上次趋势图绘制时的 history 长度
        # 数据流模式 - 切换状态
        "_stream_mode_entering": False, # 是否正在进入数据流模式（用于切换提示）
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # 自动加载演示参数（demo_params.json 存在则预填各区证据和图片）
    _try_load_demo_params()


def _try_load_demo_params():
    """尝试加载 demo_params.json，存在则作为各区初始值预填 zz_evidence 和 zz_uploaded"""
    import json, os
    demo_path = os.path.join(os.path.dirname(__file__),
                             "configs", "郑州", "demo_params.json")
    if not os.path.exists(demo_path):
        return
    try:
        with open(demo_path, "r", encoding="utf-8") as f:
            params = json.load(f)
    except Exception as e:
        st.warning(f"⚠️ 演示参数加载失败: {e}")
        return

    # 仅当 zz_evidence 为空（首次启动）时预填
    if not st.session_state.get("zz_evidence"):
        for district, cfg in params.items():
            ev = cfg.get("evidence", {})
            if ev:
                st.session_state["zz_evidence"][district] = dict(ev)
            imgs = cfg.get("images", {})
            if imgs:
                existing = st.session_state["zz_uploaded"].get(district, {})
                existing.update(imgs)
                st.session_state["zz_uploaded"][district] = existing

    # 标记已加载（供侧边栏"清除"按钮判断）
    st.session_state["_demo_params_loaded"] = True


# ============================================================================
# 郑州 40 节点先验证据（7·20 型暴雨场景）
# ============================================================================

def get_default_zz_evidence(district: str) -> dict:
    """获取郑州某区的默认先验证据（基于 Grid500 真实数据构建）"""
    _load = _load_district_evidence()
    return _load.get(district, {}).copy()


@st.cache_resource
def _load_district_evidence() -> dict:
    """加载郑州各区真实数据证据 JSON（缓存，避免每次 rerun 重建）"""
    import json, os
    ev_path = os.path.join(os.path.dirname(__file__),
                           "configs", "郑州", "district_evidence.json")
    if not os.path.exists(ev_path):
        st.warning(f"⚠️ 未找到证据文件 {ev_path}，将使用空证据")
        return {}
    with open(ev_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# 执行郑州单区推理
# ============================================================================

def infer_zz_district(district: str, engine: DisasterChainEngine,
                      override_evidence: dict = None) -> dict:
    """对郑州某区执行 BN 推理"""
    evidence = get_default_zz_evidence(district)
    if override_evidence:
        evidence.update(override_evidence)
    result = engine.infer(evidence)
    return result


# ============================================================================
# 工具：从 BN 结果提取结论
# ============================================================================

def _extract_conclusion(bn_result: dict) -> dict:
    """从 BN 结果中提取结论"""
    conclusion = {}
    for output_name in ("内涝风险", "地质灾害概率"):
        node_res = bn_result.get(output_name, {})
        if "error" in node_res:
            conclusion[output_name] = {"error": node_res["error"]}
            continue
        states = node_res.get("states", [])
        probs = node_res.get("probabilities", [])
        if states and probs:
            max_idx = max(range(len(probs)), key=lambda i: probs[i])
            conclusion[output_name] = {
                "最高风险态": states[max_idx],
                "概率": round(probs[max_idx], 4),
                "全部分布": dict(zip(states, [round(float(p), 4) for p in probs])),
            }
    return conclusion


# ============================================================================
# 灾害类型结论模块（直观结论）
# ============================================================================

def get_disaster_conclusion(bn_result: dict, confidence: dict,
                             evidence_count: int) -> dict:
    """
    基于推理结果生成直观灾害类型结论

    返回:
      {
        "main_threat": "城市内涝" / "山体滑坡" / "综合灾害" / "安全",
        "secondary_threats": ["道路损毁影响救援", ...],
        "level": "安全" / "关注" / "预警" / "危险",
        "level_color": "#2ECC71" / "#F1C40F" / "#F39C12" / "#E74C3C",
        "suggestion": "一句话建议",
        "details": ["细项1", "细项2", ...],
      }
    """
    conclusion = _extract_conclusion(bn_result)
    risk = conclusion.get("内涝风险", {})
    geo = conclusion.get("地质灾害概率", {})

    risk_state = risk.get("最高风险态", "低") if "error" not in risk else "低"
    risk_prob = risk.get("概率", 0.0) if "error" not in risk else 0.0
    geo_state = geo.get("最高风险态", "低") if "error" not in geo else "低"
    geo_prob = geo.get("概率", 0.0) if "error" not in geo else 0.0

    # 道路损毁信息
    road_conf = confidence.get("road_damage", {})
    total_damages = road_conf.get("total", 0)

    # 积水信息
    flood_conf = confidence.get("flood", {})
    area_m2 = flood_conf.get("area_m2")

    # 水位信息
    wl_conf = confidence.get("water_level", {})
    wl_cm = wl_conf.get("value_cm")

    details = []
    secondary_threats = []

    # 主要威胁判定
    if risk_state == "高" and risk_prob >= 0.5:
        main_threat = "城市内涝"
        level = "危险"
        level_color = "#E74C3C"
        suggestion = "该区城市内涝风险极高，建议立即启动应急排水、疏散低洼区域居民"
        details.append(f"内涝风险={risk_state} ({risk_prob*100:.1f}%)")
    elif risk_state == "中" or (risk_state == "高" and risk_prob < 0.5):
        main_threat = "城市内涝"
        level = "预警"
        level_color = "#F39C12"
        suggestion = "该区内涝风险较高，建议关注气象预警、检查排水设施"
        details.append(f"内涝风险={risk_state} ({risk_prob*100:.1f}%)")
    elif geo_state == "高" and geo_prob >= 0.5:
        main_threat = "山体滑坡"
        level = "预警"
        level_color = "#F39C12"
        suggestion = "该区地质灾害风险较高，建议关注坡地稳定性、加强监测"
        details.append(f"地灾概率={geo_state} ({geo_prob*100:.1f}%)")
    else:
        main_threat = "安全"
        level = "安全"
        level_color = "#2ECC71"
        suggestion = "该区当前风险较低，保持常规监测即可"
        details.append("各风险指标均在安全范围内")

    # 次生灾害
    if geo_state == "高" and geo_prob >= 0.5:
        secondary_threats.append("存在山体滑坡/地质灾害风险，建议关注坡地")
        details.append(f"地质灾害={geo_state} ({geo_prob*100:.1f}%)")
    elif geo_state == "高":
        secondary_threats.append("地质灾害风险偏高，建议关注")
        details.append(f"地质灾害={geo_state} ({geo_prob*100:.1f}%)")

    if total_damages >= 3:
        secondary_threats.append("道路桥梁损毁明显，影响救援通行")
        details.append(f"道路损毁={total_damages}处")
    elif total_damages >= 1:
        secondary_threats.append("存在道路损毁情况")
        details.append(f"道路损毁={total_damages}处")

    if area_m2 is not None and area_m2 > 500:
        secondary_threats.append("低洼区域积水严重")
        details.append(f"积水面积={area_m2:.0f}m²")
    elif area_m2 is not None and area_m2 > 100:
        secondary_threats.append("低洼区域存在积水")
        details.append(f"积水面积={area_m2:.0f}m²")

    if wl_cm is not None and wl_cm > 200:
        secondary_threats.append("河道水位超警戒，有溃堤风险")
        details.append(f"水位={wl_cm:.0f}cm（危险）")
    elif wl_cm is not None and wl_cm > 100:
        secondary_threats.append("河道水位偏高")
        details.append(f"水位={wl_cm:.0f}cm（警戒）")

    # 级别调整：同时有多个威胁则升级
    if len(secondary_threats) >= 2 and level == "预警":
        level = "危险"
        level_color = "#E74C3C"
        suggestion = "该区面临多重灾害威胁，建议综合评估、全面响应"

    if not secondary_threats:
        secondary_threats.append("未发现明显次生灾害风险")

    return {
        "main_threat": main_threat,
        "secondary_threats": secondary_threats,
        "level": level,
        "level_color": level_color,
        "suggestion": suggestion,
        "details": details,
        "evidence_count": evidence_count,
    }


# ============================================================================
# 置信度水平条渲染
# ============================================================================

def render_confidence_bar(confidence_pct: float, width: int = 150):
    """渲染一个内联置信度水平条（HTML/CSS）"""
    pct = min(max(confidence_pct, 0), 100)
    color = "#2ECC71" if pct >= 80 else ("#F1C40F" if pct >= 50 else "#E74C3C")
    bar = f"""
    <div style="display:inline-block; width:{width}px; height:16px;
                background:#eee; border-radius:8px; position:relative; vertical-align:middle;">
      <div style="width:{pct}%; height:100%; background:{color};
                  border-radius:8px; transition:width 0.3s;"></div>
    </div>
    <span style="font-size:12px; margin-left:4px; font-weight:bold;">{pct:.0f}%</span>
    """
    return bar


# ============================================================================
# 郑州地图绘制（增强版：文字标注 + 透明度通道 + 灾害结论）
# ============================================================================

def get_district_risk(zz_results: dict, district: str) -> Tuple[str, float, dict]:
    """获取某区风险最高态、置信度、全部分布"""
    result = zz_results.get(district, {})
    risk = result.get("内涝风险", {})
    geo = result.get("地质灾害概率", {})

    if "error" in risk:
        return "无数据", 0.0, {"低": 0.33, "中": 0.33, "高": 0.34}

    states = risk.get("states", [])
    probs = risk.get("probabilities", [])
    if not states or not probs:
        return "无数据", 0.0, {"低": 0.33, "中": 0.33, "高": 0.34}

    max_idx = int(np.argmax(probs))
    top_state = states[max_idx]
    top_prob = round(float(probs[max_idx]), 4)

    dist = dict(zip(states, [round(float(p), 4) for p in probs]))
    return top_state, top_prob, dist


def get_district_geo_risk(zz_results: dict, district: str) -> Tuple[str, float]:
    """获取某区地灾概率"""
    result = zz_results.get(district, {})
    geo = result.get("地质灾害概率", {})
    if "error" in geo or "probabilities" not in geo:
        return "低", 0.0
    states = geo.get("states", [])
    probs = geo.get("probabilities", [])
    if not states or not probs:
        return "低", 0.0
    max_idx = int(np.argmax(probs))
    return states[max_idx], round(float(probs[max_idx]), 4)


def _get_district_center(district: str) -> Tuple[float, float]:
    """获取郑州各区县的近似中心坐标（用于地图标注放置）"""
    centers = {
        "中原区": (34.75, 113.61),
        "二七区": (34.73, 113.64),
        "管城回族区": (34.75, 113.68),
        "金水区": (34.79, 113.66),
        "上街区": (34.80, 113.30),
        "惠济区": (34.84, 113.62),
        "中牟县": (34.72, 113.98),
        "巩义市": (34.75, 113.02),
        "荥阳市": (34.79, 113.39),
        "新密市": (34.54, 113.39),
        "新郑市": (34.40, 113.74),
        "登封市": (34.46, 113.04),
    }
    return centers.get(district, (34.75, 113.65))


def _hex_to_rgba(hex_color: str, opacity: float) -> str:
    """将十六进制颜色转为 rgba 字符串（编码透明度到颜色中）"""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{opacity:.3f})"


def render_zz_map(zz_results: dict, geojson: dict, title: str = "郑州各区内涝风险态势",
                  period: int = 0, height: int = 680) -> go.Figure:
    """
    绘制郑州 Choropleth 地图（单 trace 优化版 + MapLibre 后端）
    ---------------------------------------------------------
    优化点:
      1. 单 trace 渲染（12 区合并为 1 次 add_trace）
      2. 各区独立透明度编码到 RGBA 颜色中
      3. MapLibre 后端（layout.map）替代 Mapbox
      4. 使用简化版 GeoJSON（顶点数减少 80%+）
    """
    # ── 缓存检查：相同版本号和时段下复用已构建的 fig ──
    cache_key = f"map_v{st.session_state.get('_map_cache_version', 0)}_p{period}"
    cached = st.session_state.get("_map_fig_cache", {}).get(cache_key)
    if cached is not None:
        # 只需更新标题（时段可能变化）
        cached.update_layout(title=dict(
            text=f"{title}（时段 {period}）",
            font=dict(size=16, family="Microsoft YaHei"),
        ))
        return cached
    n = len(ZHENGZHOU_DISTRICTS)
    z_values = []
    locations = []
    hover_texts = []
    colorscale = []

    for i, d in enumerate(ZHENGZHOU_DISTRICTS):
        top_state, top_prob, dist = get_district_risk(zz_results, d)
        geo_state, geo_prob = get_district_geo_risk(zz_results, d)
        evidence = st.session_state["zz_evidence"].get(d, {})
        n_evidence = len(evidence)
        bn_result = zz_results.get(d, {})
        confidence = st.session_state["zz_confidence"].get(d, {})
        conclusion = get_disaster_conclusion(bn_result, confidence, n_evidence)

        color_map = {"低": "#2ECC71", "中": "#F1C40F", "高": "#E74C3C"}
        fill_color = color_map.get(top_state, "#95A5A6")

        # 透明度与置信度挂钩（置信度越高越实），编码到 RGBA 颜色中
        opacity = max(0.3, min(1.0, top_prob * 1.2))
        rgba = _hex_to_rgba(fill_color, opacity)

        locations.append(d)
        z_values.append(i)
        hover_texts.append(
            f"<b>{d}</b><br>"
            f"<b>主要威胁: {conclusion['main_threat']}</b><br>"
            f"<span style='color:{conclusion['level_color']}'><b>等级: {conclusion['level']}</b></span><br>"
            f"内涝风险: <b>{top_state}</b> ({top_prob*100:.1f}%)<br>"
            f"内涝分布: {', '.join(f'{k}={v*100:.0f}%' for k, v in dist.items())}<br>"
            f"地灾概率: {geo_state} ({geo_prob*100:.1f}%)<br>"
            f"证据覆盖: {n_evidence}/40 节点<br>"
            f"<i>{conclusion['suggestion']}</i>"
        )

        # Colorscale 每个区一个独立色标（位置 = i/(n-1)）
        pos = i / (n - 1) if n > 1 else 0
        colorscale.append([pos, rgba])

    # ── 单 trace 色块层 ──
    fig = go.Figure()
    fig.add_trace(go.Choroplethmap(
        geojson=geojson,
        locations=locations,
        z=z_values,
        featureidkey="properties.name",
        colorscale=colorscale,
        zmin=0,
        zmax=n - 1,
        marker=dict(line=dict(width=1, color="white")),
        marker_opacity=1.0,  # 透明度已编码到 RGBA 颜色中
        showscale=False,
        hovertext=hover_texts,
        hoverinfo="text",
        name="",
    ))

    # ── 文字标注层（风险态 + 置信度%） ──
    annotations = []
    for d in ZHENGZHOU_DISTRICTS:
        top_state, top_prob, _ = get_district_risk(zz_results, d)
        lat, lon = _get_district_center(d)
        annotations.append(dict(
            x=lon,
            y=lat,
            text=f"<b>{top_state}</b><br>{top_prob * 100:.0f}%",
            showarrow=False,
            font=dict(
                size=12,
                color="black" if top_state != "高" else "white",
                family="Microsoft YaHei",
            ),
            bgcolor="rgba(255,255,255,0.8)" if top_state != "高" else "rgba(231,76,60,0.8)",
            bordercolor=("#2ECC71" if top_state == "低"
                         else "#F1C40F" if top_state == "中"
                         else "#E74C3C"),
            borderwidth=1,
            borderpad=2,
        ))

    fig.update_layout(
        title=dict(
            text=f"{title}（时段 {period}）",
            font=dict(size=16, family="Microsoft YaHei"),
        ),
        font=dict(family="Microsoft YaHei"),
        map=dict(
            style="carto-positron",
            center=dict(lat=34.75, lon=113.65),
            zoom=9.2,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        paper_bgcolor="white",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Microsoft YaHei",
        ),
        annotations=annotations,
    )

    # ── 存入缓存 ──
    if "_map_fig_cache" not in st.session_state:
        st.session_state["_map_fig_cache"] = {}
    st.session_state["_map_fig_cache"][cache_key] = fig

    return fig


# ============================================================================
# 图片上传与识别
# ============================================================================

def run_fuse_infer_for_district(district: str, uploaded_images: dict) -> dict:
    """对某区执行 fuse_infer（图片识别 + BN 推理）"""
    from tools.fuse_infer import fuse_infer

    tasks = []
    for task_type, img_path in uploaded_images.items():
        if img_path and os.path.exists(img_path):
            tasks.append((img_path, task_type))

    if not tasks:
        # 仅用参数推理
        evidence = st.session_state["zz_evidence"].get(district, {})
        engine = load_engine(str(CONFIG_PATH))
        result = engine.infer(evidence)
        return {
            "识别结果明细": [],
            "证据字典": evidence,
            "证据缺失": [],
            "置信度摘要": {},
            "BN推理结果": result,
            "结论": _extract_conclusion(result),
        }

    result = fuse_infer(tasks, config_path=str(CONFIG_PATH))
    return result


# ============================================================================
# 一键推理全部区（支持三种来源）
# ============================================================================

def run_infer_all_districts(engine: DisasterChainEngine, source: str):
    """
    一键推理全部 12 个区。

    参数:
      source: "demo" — 使用 demo_params.json 的证据+图片识别
              "prior" — 使用 get_default_zz_evidence() 纯 BN 推理
              "manual" — 使用 zz_evidence 现有内容推理
    """
    if source == "demo":
        _run_infer_all_demo(engine)
    elif source == "prior":
        _run_infer_all_prior(engine)
    elif source == "manual":
        _run_infer_all_manual(engine)
    st.session_state["zz_infer_source"] = source
    st.session_state["zz_fuse_infer_once"] = True


def _run_infer_all_demo(engine: DisasterChainEngine):
    """演示参数模式：加载 demo_params.json → 图片识别 → BN 推理"""
    import json, os
    demo_path = os.path.join(os.path.dirname(__file__),
                             "configs", "郑州", "demo_params.json")
    if not os.path.exists(demo_path):
        st.error("❌ 未找到演示参数文件 demo_params.json，请先运行 scripts/prepare_demo_params.py")
        return

    with open(demo_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    n = len(ZHENGZHOU_DISTRICTS)
    progress_bar = st.progress(0, text="正在准备演示参数…")
    step = 0

    for d in ZHENGZHOU_DISTRICTS:
        cfg = params.get(d, {})
        # 填入证据
        ev = cfg.get("evidence", {})
        if ev:
            st.session_state["zz_evidence"][d] = dict(ev)

        # 填入图片并执行识别
        imgs = cfg.get("images", {})
        if imgs:
            st.session_state["zz_uploaded"][d] = dict(imgs)
            step += 1
            progress_bar.progress(
                min(step / (n * 2), 0.5),
                text=f"正在识别图片 {step}/{n} 区：{d}…",
            )
            # 执行图片识别（复用现有逻辑）
            st.session_state["zz_recognition"][d] = \
                run_fuse_infer_for_district(d, imgs)

        # BN 推理
        override = st.session_state["zz_evidence"].get(d, {})
        result = engine.infer(override)
        st.session_state["zz_results"][d] = result
        step += 1
        progress_bar.progress(
            step / (n * 2),
            text=f"正在推理第 {step - n}/{n} 区：{d}…",
        )

        # 提取置信度
        recog = st.session_state["zz_recognition"].get(d, {})
        conf = recog.get("置信度摘要", {})
        st.session_state["zz_confidence"][d] = conf

    progress_bar.empty()
    st.success(f"✅ 演示参数推演完成（{n} 个区，含图片识别）")
    st.session_state["_map_cache_version"] += 1


def _run_infer_all_prior(engine: DisasterChainEngine):
    """先验数据模式：用 get_default_zz_evidence() 纯 BN 推理"""
    n = len(ZHENGZHOU_DISTRICTS)
    progress_bar = st.progress(0, text="正在使用先验数据推理全部 12 个区…")
    for i, d in enumerate(ZHENGZHOU_DISTRICTS):
        # 先验模式：用 get_default_zz_evidence，不覆盖 zz_evidence
        evidence = get_default_zz_evidence(d)
        result = engine.infer(evidence)
        st.session_state["zz_results"][d] = result
        progress_bar.progress(
            (i + 1) / n,
            text=f"正在推理第 {i+1}/{n} 区：{d}…",
        )
    progress_bar.empty()
    st.success(f"✅ 先验数据推演完成（{n} 个区，纯 BN 推理，无图片识别）")
    st.session_state["_map_cache_version"] += 1


def _run_infer_all_manual(engine: DisasterChainEngine):
    """当前手动参数模式：用 zz_evidence 现有内容推理"""
    n = len(ZHENGZHOU_DISTRICTS)
    progress_bar = st.progress(0, text="正在使用手动参数推理全部 12 个区…")
    for i, d in enumerate(ZHENGZHOU_DISTRICTS):
        override = st.session_state["zz_evidence"].get(d, {})
        if not override:
            # 若某区无手动参数，跳过
            progress_bar.progress(
                (i + 1) / n,
                text=f"跳过第 {i+1}/{n} 区：{d}（无手动参数）",
            )
            continue
        result = engine.infer(override)
        st.session_state["zz_results"][d] = result
        progress_bar.progress(
            (i + 1) / n,
            text=f"正在推理第 {i+1}/{n} 区：{d}…",
        )
    progress_bar.empty()
    st.success(f"✅ 手动参数推演完成（{n} 个区）")
    st.session_state["_map_cache_version"] += 1


# ============================================================================
# 六阶段结果总结（增强版：置信度条 + 证据来源标注）
# ============================================================================

def render_confidence_bar_html(confidence_pct: float, width: int = 120) -> str:
    """生成置信度水平条 HTML"""
    pct = min(max(confidence_pct, 0), 100)
    color = "#2ECC71" if pct >= 80 else ("#F1C40F" if pct >= 50 else "#E74C3C")
    bar = f"""
    <div style="display:inline-block; width:{width}px; height:14px;
                background:#eee; border-radius:7px; position:relative; vertical-align:middle;">
      <div style="width:{pct}%; height:100%; background:{color};
                  border-radius:7px;"></div>
    </div>
    <span style="font-size:11px; margin-left:3px; font-weight:bold;">{pct:.0f}%</span>
    """
    return bar


def render_six_stage_summary(zz_results: dict, zz_confidence: dict,
                              district: str, zz_recognition: dict):
    """渲染六阶段结果总结卡片（增强版：置信度条 + 灾害结论）"""
    recog = zz_recognition.get(district, {})
    details = recog.get("识别结果明细", [])
    conf = zz_confidence.get(district, {})
    result = zz_results.get(district, {})

    # 先显示灾害结论卡片
    evidence = st.session_state["zz_evidence"].get(district, {})
    n_evidence = len(evidence)
    conclusion = get_disaster_conclusion(result, conf, n_evidence)

    st.markdown("---")
    st.markdown(f"##### 🏆 {district} — 灾害类型结论")

    # 结论卡片
    level_emoji = {"安全": "✅", "关注": "👁️", "预警": "⚠️", "危险": "🚨"}
    emoji = level_emoji.get(conclusion["level"], "❓")
    st.markdown(
        f"""
        <div style="border:2px solid {conclusion['level_color']}; border-radius:10px;
                    padding:12px 16px; background:{conclusion['level_color']}10; margin-bottom:12px;">
          <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
            <span style="font-size:24px;">{emoji}</span>
            <div>
              <span style="font-size:18px; font-weight:bold; color:{conclusion['level_color']};">
                {conclusion['level']} — 主要威胁: {conclusion['main_threat']}
              </span>
              <br>
              <span style="font-size:13px; color:#555;">{conclusion['suggestion']}</span>
            </div>
          </div>
          <div style="margin-top:8px; font-size:12px; color:#666;">
            {" · ".join(conclusion['details'])}
          </div>
          <div style="margin-top:4px; font-size:12px; color:#888;">
            次生灾害: {" · ".join(conclusion['secondary_threats'])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"##### 📋 {district} — 六阶段结果总结")

    # 提取各阶段数据
    stages = []

    # 1. 河道水位
    wl_conf = conf.get("water_level", {})
    wl_cm = wl_conf.get("value_cm")
    wl_confidence = wl_conf.get("confidence", 0.0)
    wl_source = wl_conf.get("source", "先验")
    if wl_cm is not None:
        wl_status = "危险" if wl_cm > 200 else ("警戒" if wl_cm > 100 else "正常")
        stages.append({
            "阶段": "河道水位", "数据来源": "水位识别",
            "输出": f"{wl_cm:.0f} cm",
            "状态": wl_status,
            "置信度条": render_confidence_bar_html(wl_confidence * 100),
            "置信度数值": wl_confidence * 100,
            "证据来源": "真实识别" if wl_source == "visual_recognition" else "先验",
        })
    else:
        stages.append({
            "阶段": "河道水位", "数据来源": "水位识别",
            "输出": "—", "状态": "无数据",
            "置信度条": "—", "置信度数值": 0,
            "证据来源": "先验",
        })

    # 2. 道路桥梁损毁
    road_conf = conf.get("road_damage", {})
    total_damages = road_conf.get("total", 0)
    road_confidence = road_conf.get("avg_confidence", 0.0)
    road_source = road_conf.get("source", "先验")
    per_class = road_conf.get("per_class", {})
    if total_damages > 0:
        class_str = ", ".join(f"{k}={v}" for k, v in per_class.items())
        stages.append({
            "阶段": "道路桥梁损毁", "数据来源": "道路识别",
            "输出": f"共 {total_damages} 处",
            "状态": f"({class_str})",
            "置信度条": render_confidence_bar_html(road_confidence * 100),
            "置信度数值": road_confidence * 100,
            "证据来源": "真实识别" if road_source == "visual_recognition" else "先验",
        })
    else:
        stages.append({
            "阶段": "道路桥梁损毁", "数据来源": "道路识别",
            "输出": "—", "状态": "无数据",
            "置信度条": "—", "置信度数值": 0,
            "证据来源": "先验",
        })

    # 3. 低洼地形积水
    flood_conf = conf.get("flood", {})
    area_m2 = flood_conf.get("area_m2")
    inundation = flood_conf.get("inundation_ratio", 0)
    disaster_level = flood_conf.get("disaster_level", "未知")
    flood_source = flood_conf.get("source", "先验")
    # 积水置信度用洪水分割 mask 面积占比近似
    flood_confidence = min(1.0, inundation * 2) if inundation else 0.0
    if area_m2 is not None:
        stages.append({
            "阶段": "低洼地形积水", "数据来源": "洪水分割",
            "输出": f"{area_m2:.1f} m²",
            "状态": f"淹没占比 {inundation*100:.2f}% · {disaster_level}",
            "置信度条": render_confidence_bar_html(flood_confidence * 100),
            "置信度数值": flood_confidence * 100,
            "证据来源": "真实识别" if flood_source == "visual_recognition" else "先验",
        })
    else:
        stages.append({
            "阶段": "低洼地形积水", "数据来源": "洪水分割",
            "输出": "—", "状态": "无数据",
            "置信度条": "—", "置信度数值": 0,
            "证据来源": "先验",
        })

    # 4. 救助地点损毁（道路模型近似）
    if total_damages > 0:
        rescue_damages = max(1, total_damages // 3)
        stages.append({
            "阶段": "救助地点损毁", "数据来源": "道路模型近似",
            "输出": f"约 {rescue_damages} 处",
            "状态": "近似评估",
            "置信度条": render_confidence_bar_html(road_confidence * 100 * 0.7),
            "置信度数值": road_confidence * 100 * 0.7,
            "证据来源": "近似评估（道路模型）",
        })
    else:
        stages.append({
            "阶段": "救助地点损毁", "数据来源": "道路模型近似",
            "输出": "—", "状态": "无数据",
            "置信度条": "—", "置信度数值": 0,
            "证据来源": "近似评估（道路模型）",
        })

    # 5. 城市内涝（BN 推理）
    risk_conclusion = _extract_conclusion(result).get("内涝风险", {})
    risk_state = risk_conclusion.get("最高风险态", "低")
    risk_prob = risk_conclusion.get("概率", 0.0)
    # 根据推演来源确定证据来源
    _infer = st.session_state.get("zz_infer_source", "prior")
    _has_ev = bool(st.session_state["zz_evidence"].get(district))
    if _infer == "demo":
        risk_source = "演示参数"
    elif _infer == "manual" and _has_ev:
        risk_source = "手动参数"
    else:
        risk_source = "先验"
    stages.append({
        "阶段": "城市内涝", "数据来源": "BN 推理",
        "输出": f"风险态: {risk_state}",
        "状态": f"概率 {risk_prob*100:.1f}%",
        "置信度条": render_confidence_bar_html(risk_prob * 100),
        "置信度数值": risk_prob * 100,
        "证据来源": risk_source,
    })

    # 6. 山体滑坡（BN 推理）
    geo_conclusion = _extract_conclusion(result).get("地质灾害概率", {})
    geo_state = geo_conclusion.get("最高风险态", "低")
    geo_prob = geo_conclusion.get("概率", 0.0)
    if _infer == "demo":
        geo_source = "演示参数"
    elif _infer == "manual" and _has_ev:
        geo_source = "手动参数"
    else:
        geo_source = "先验"
    stages.append({
        "阶段": "山体滑坡", "数据来源": "BN 推理",
        "输出": f"风险态: {geo_state}",
        "状态": f"概率 {geo_prob*100:.1f}%",
        "置信度条": render_confidence_bar_html(geo_prob * 100),
        "置信度数值": geo_prob * 100,
        "证据来源": geo_source,
    })

    # 渲染表格（使用 HTML 以支持置信度条）
    df = pd.DataFrame(stages)
    # 用 HTML 渲染置信度条
    for i, row in df.iterrows():
        df.at[i, "置信度条"] = row["置信度条"]

    st.dataframe(
        df[["阶段", "数据来源", "输出", "状态", "置信度条", "证据来源"]],
        use_container_width=True, hide_index=True,
        column_config={
            "阶段": st.column_config.TextColumn("阶段", width="small"),
            "数据来源": st.column_config.TextColumn("数据来源", width="small"),
            "输出": st.column_config.TextColumn("输出", width="medium"),
            "状态": st.column_config.TextColumn("状态", width="medium"),
            "置信度条": st.column_config.TextColumn("置信度", width="medium"),
            "证据来源": st.column_config.TextColumn("证据来源", width="small"),
        },
    )


# ============================================================================
# 链式递进预测
# ============================================================================

# 递推规则定义（可扩展）
CHAIN_RULES = [
    {
        "id": "rule_1",
        "name": "内涝→土壤含水量",
        "condition": {"内涝风险": "高"},
        "action": {"前期土壤含水量": "高", "径流系数": "大", "河道水位": "危险"},
        "description": "内涝风险高 → 前期土壤含水量=高, 径流系数=大, 河道水位=危险",
    },
    {
        "id": "rule_2",
        "name": "内涝中→土壤含水量",
        "condition": {"内涝风险": "中"},
        "action": {"前期土壤含水量": "中", "径流系数": "大"},
        "description": "内涝风险中 → 前期土壤含水量=中, 径流系数=大",
    },
    {
        "id": "rule_3",
        "name": "地灾→地质易发性",
        "condition": {"地质灾害概率": "高"},
        "action": {"地质易发性": "高", "滑坡历史密度": "高", "土壤渗透性": "差"},
        "description": "地灾概率高 → 地质易发性=高, 滑坡历史密度=高, 土壤渗透性=差",
    },
]


def chain_forward(evidence: dict, current_results: dict) -> dict:
    """
    链式递推：将当前推理结果作为下一时段证据

    递推规则（可扩展）：
      - 内涝风险=高 → 前期土壤含水量=高, 径流系数=大, 河道水位=危险
      - 内涝风险=中 → 前期土壤含水量=中, 径流系数=大
      - 地质灾害概率=高 → 地质易发性=高, 滑坡历史密度=高, 土壤渗透性=差
    """
    next_evidence = dict(evidence)
    rules_triggered = []

    conclusion = _extract_conclusion(current_results)

    risk = conclusion.get("内涝风险", {})
    geo = conclusion.get("地质灾害概率", {})

    risk_state = risk.get("最高风险态", "低") if "error" not in risk else "低"
    geo_state = geo.get("最高风险态", "低") if "error" not in geo else "低"

    for rule in CHAIN_RULES:
        cond = rule["condition"]
        triggered = False
        if "内涝风险" in cond and risk_state == cond["内涝风险"]:
            triggered = True
        if "地质灾害概率" in cond and geo_state == cond["地质灾害概率"]:
            triggered = True
        if triggered:
            next_evidence.update(rule["action"])
            rules_triggered.append(rule["description"])

    return next_evidence, rules_triggered


# ============================================================================
# 郑州模式渲染
# ============================================================================

def render_district_sidebar(engine: DisasterChainEngine, district: str):
    """渲染某区的侧边栏参数编辑和图片上传"""
    st.markdown(f"##### {district} 参数")

    # 图片上传区
    st.markdown("**📷 上传图片（可选）**")
    uploaded = st.session_state["zz_uploaded"].get(district, {})

    img_types = {
        "water_level": "水位尺图片",
        "road": "道路桥梁图片",
        "flood": "洪水现场图片",
    }

    for task_key, task_label in img_types.items():
        uploaded_file = st.file_uploader(
            task_label,
            type=["jpg", "jpeg", "png", "bmp"],
            key=f"zz_upload_{district}_{task_key}",
        )
        if uploaded_file is not None:
            # 保存到临时目录
            save_dir = Path(V2_ROOT) / "temp_uploads" / district
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded[task_key] = str(save_path)
            st.success(f"✅ {task_label} 已上传")
        elif task_key in uploaded and not os.path.exists(uploaded[task_key]):
            del uploaded[task_key]

    st.session_state["zz_uploaded"][district] = uploaded

    if uploaded:
        for task_key, path in uploaded.items():
            if os.path.exists(path):
                st.image(path, caption=f"{task_key}", width=200)

    st.markdown("---")
    st.markdown("**⚙️ 参数编辑**")

    evidence = st.session_state["zz_evidence"].get(district, {})
    current_evidence = dict(evidence) if evidence else {}

    input_params = engine.get_input_params()
    # 按类别分组
    categories = {"气象": [], "水文": [], "地形": [], "地质": [], "城市": []}
    for p in input_params:
        cfg = engine.get_node_config(p)
        cat = cfg.get("category", "其他")
        if cat in categories:
            categories[cat].append(p)
        else:
            categories.setdefault("其他", []).append(p)

    cat_labels = {
        "气象": "🌤️ 气象", "水文": "💧 水文", "地形": "⛰️ 地形",
        "地质": "🪨 地质", "城市": "🏙️ 城市",
    }

    for cat, params in categories.items():
        if not params:
            continue
        with st.expander(f"{cat_labels.get(cat, cat)} ({len(params)})", expanded=False):
            for pname in params:
                cfg = engine.get_node_config(pname)
                states = ["先验/不指定"] + cfg["states"]
                current_val = current_evidence.get(pname, "先验/不指定")
                default_idx = states.index(current_val) if current_val in states else 0
                selected = st.selectbox(
                    f"**{pname}**",
                    states,
                    index=default_idx,
                    key=f"zz_param_{district}_{pname}",
                    label_visibility="collapsed",
                )
                if selected != "先验/不指定":
                    current_evidence[pname] = selected
                elif pname in current_evidence:
                    del current_evidence[pname]

        st.session_state["zz_evidence"][district] = current_evidence

    # 执行推理按钮
    if st.button(f"🚀 推理 {district}", key=f"zz_infer_{district}", use_container_width=True):
        with st.spinner(f"正在识别图像并推理 {district}…"):
            try:
                # 检查是否有上传图片
                uploaded_images = st.session_state["zz_uploaded"].get(district, {})
                if uploaded_images:
                    fuse_result = run_fuse_infer_for_district(district, uploaded_images)
                    st.session_state["zz_recognition"][district] = fuse_result
                    st.session_state["zz_confidence"][district] = fuse_result.get("置信度摘要", {})
                    bn_result = fuse_result.get("BN推理结果", {})
                else:
                    evidence = st.session_state["zz_evidence"].get(district, {})
                    bn_result = engine.infer(evidence)
                    st.session_state["zz_confidence"][district] = {}

                st.session_state["zz_results"][district] = bn_result
                st.session_state["zz_fuse_infer_once"] = True
                st.success(f"✅ {district} 推理完成")
            except Exception as e:
                st.error(f"❌ 推理失败: {str(e)}")


def render_zz_page():
    """渲染郑州全景可视化主页面"""
    # 首屏加载状态
    load_status = st.status("正在加载灾害链引擎…", expanded=False) if not st.session_state.get("_engine_loaded") else None

    geojson = load_geojson()
    if geojson is None:
        st.error("GeoJSON 数据加载失败，请检查文件路径")
        return

    engine = load_engine(str(CONFIG_PATH))

    if load_status:
        load_status.update(label="✅ 引擎就绪", state="complete", expanded=False)
        st.session_state["_engine_loaded"] = True

    st.title("🌊 郑州全景可视化 — 灾害链态势感知")
    st.markdown("---")

    # ── 顶部导航 ──
    col_tabs = st.columns([1, 1, 1, 1, 1, 1])
    tab_labels = ["🗺️ 地图总览", "📋 区详情", "⏱️ 链式预测", "📡 实时监测", "⚙️ 参数", "ℹ️ 帮助"]
    current_tab = st.session_state.get("zz_tab", "map")

    tab_map = {
        col_tabs[0]: "map", col_tabs[1]: "detail",
        col_tabs[2]: "chain", col_tabs[3]: "era5",
        col_tabs[4]: "params", col_tabs[5]: "help",
    }

    for col, tab_name in tab_map.items():
        with col:
            idx = ["map", "detail", "chain", "era5", "params", "help"].index(tab_name)
            if st.button(tab_labels[idx], use_container_width=True,
                         type="primary" if current_tab == tab_name else "secondary"):
                st.session_state["zz_tab"] = tab_name
                st.rerun()

    current_tab = st.session_state.get("zz_tab", "map")

    # ═══════════════════════════════════════════════════════════════
    # 实时监测辅助函数
    # ═══════════════════════════════════════════════════════════════

    def _render_replay_mode(engine, geojson, records):
        """回放模式：ERA5 逐小时时间轴播放"""
        # ── 初始化 ERA5 推理结果（磁盘缓存加速） ──
        if not st.session_state["_era5_initialized"]:
            cache_result = load_era5_results_cache()
            if cache_result["valid"]:
                all_results = cache_result["results"]
                st.session_state["_era5_results_history"] = all_results
                st.session_state["_era5_initialized"] = True
                st.toast("✅ 预计算结果缓存命中，秒开加载", icon="⚡")
            else:
                with st.status("⏳ 正在预计算 144 小时推理，首次约 3 分钟，完成后秒开",
                               expanded=True) as status:
                    all_results = []
                    base_ev = st.session_state.get("zz_evidence", {})
                    total = len(records)
                    for i, rec in enumerate(records):
                        hour_results = {}
                        for d in ZHENGZHOU_DISTRICTS:
                            district_ev = base_ev.get(d, {}).copy()
                            for node, state in rec["evidence"].items():
                                if state != "保持先验":
                                    district_ev[node] = state
                            hour_results[d] = engine.infer(district_ev)
                        pct = int((i + 1) / total * 100)
                        status.update(label=f"⏳ 预计算中 {i+1}/{total} 小时 ({pct}%)…")
                        all_results.append({"hour": i, "results": hour_results})
                    save_era5_results_cache(all_results)
                    st.session_state["_era5_results_history"] = all_results
                    st.session_state["_era5_initialized"] = True
                    status.update(label="✅ 144 小时推理预计算完成（已缓存，下次秒开）",
                                  state="complete")

        # ── 时间轴控制 ──
        col_controls = st.columns([2, 1, 1, 1, 1])
        current_hour = st.session_state["_era5_hour"]
        with col_controls[0]:
            hour = st.slider("时间轴（小时）", min_value=0, max_value=143,
                             value=current_hour, format="%d", key="era5_slider")
            if hour != current_hour:
                st.session_state["_era5_hour"] = hour
                st.session_state["_era5_playing"] = False
                st.rerun()
        with col_controls[1]:
            speed = st.selectbox("播放速度", [1, 2, 5], index=0,
                                 format_func=lambda x: f"{x}秒/步", key="era5_speed_sel")
            st.session_state["_era5_speed"] = float(speed)
        with col_controls[2]:
            play_label = "⏸️ 暂停" if st.session_state["_era5_playing"] else "▶️ 播放"
            if st.button(play_label, use_container_width=True, type="primary"):
                st.session_state["_era5_playing"] = not st.session_state["_era5_playing"]
                st.rerun()
        with col_controls[3]:
            if st.button("⏹️ 停止", use_container_width=True):
                st.session_state["_era5_playing"] = False
                st.session_state["_era5_hour"] = 0
                st.rerun()
        with col_controls[4]:
            if st.button("⏩ 末尾", use_container_width=True):
                st.session_state["_era5_playing"] = False
                st.session_state["_era5_hour"] = 143
                st.rerun()

        if st.session_state["_era5_playing"]:
            current_hour = st.session_state["_era5_hour"]
            if current_hour < 143:
                st.session_state["_era5_hour"] = current_hour + 1
            else:
                st.session_state["_era5_playing"] = False
            time.sleep(st.session_state["_era5_speed"])
            st.rerun()

        current_hour = st.session_state["_era5_hour"]
        current_rec = records[current_hour]
        current_results = st.session_state["_era5_results_history"][current_hour]["results"]
        dt_str = current_rec["datetime"]
        dt_display = dt_str.replace("T", " ")[:16]
        month_day = dt_display[5:10]
        hour_only = dt_display[11:13]
        display_label = f"{month_day} {hour_only}:00"
        ev = current_rec["evidence"]

        col_info = st.columns([1, 1, 1])
        with col_info[0]:
            st.metric("当前时刻", f"7月{int(month_day[3:5])}日 {hour_only}:00",
                      delta=f"第 {current_hour}/143 小时")
        with col_info[1]:
            st.metric("气象摘要", f"降水={ev['降水强度']}, 时长={ev['降水时长']}")
        with col_info[2]:
            st.metric("土壤/气温", f"土壤含水量={ev['前期土壤含水量']}")

        # ── 地图 ──
        map_title = f"郑州实时监测（{display_label}）"
        with st.spinner("⏳ 正在渲染地图…"):
            fig = render_zz_map(current_results, geojson, map_title,
                                period=current_hour, height=480)
        st.plotly_chart(fig, use_container_width=True)

        # ── 地图时段切换（预测） ──
        st.markdown("##### 地图时段切换（预测）")
        fh = st.session_state["_era5_forecast_horizon"]
        col_fh = st.columns([1, 1, 1, 3])
        fh_labels = [("当前", 0), ("+1 小时", 1), ("+3 小时", 3)]
        for ci, (label, val) in enumerate(fh_labels):
            with col_fh[ci]:
                if st.button(label, use_container_width=True,
                             type="primary" if fh == val else "secondary",
                             key=f"fh_{val}"):
                    st.session_state["_era5_forecast_horizon"] = val
                    st.rerun()
        if fh > 0:
            era5_fc = load_era5_forecasts()
            fc_rec = era5_fc.get("forecasts", {}).get(str(current_hour), {})
            if fc_rec and str(fh) in fc_rec.get("horizons", {}):
                fc_ev = fc_rec["horizons"][str(fh)]["evidence"]
                fc_conf = fc_rec["horizons"][str(fh)]["confidence"]
                fc_results = {}
                base_ev = st.session_state.get("zz_evidence", {})
                for d in ZHENGZHOU_DISTRICTS:
                    district_ev = base_ev.get(d, {}).copy()
                    for node, state in fc_ev.items():
                        if state != "保持先验":
                            district_ev[node] = state
                    fc_results[d] = engine.infer(district_ev)
                st.caption(f"📈 预测 {fh} 小时后（置信度：降水={fc_conf.get('降水强度',0.6):.0%}, "
                          f"土壤含水量={fc_conf.get('前期土壤含水量',0.6):.0%}）")
                with st.spinner("⏳ 渲染预测地图…"):
                    st.session_state["_map_cache_version"] += 1
                    fc_fig = render_zz_map(fc_results, geojson,
                                           f"郑州 {fh} 小时后预测（{display_label}）",
                                           period=current_hour + fh * 100, height=420)
                st.plotly_chart(fc_fig, use_container_width=True)

        # ── 预测结论条 ──
        era5_fc = load_era5_forecasts()
        fc_rec = era5_fc.get("forecasts", {}).get(str(current_hour), {})
        if fc_rec and "horizons" in fc_rec:
            h1 = fc_rec["horizons"].get("1", {})
            ev1 = h1.get("evidence", {})
            conf1 = h1.get("confidence", {})
            if ev1:
                st.info(
                    f"📈 **预测未来 1 小时**: 降水强度 **{ev1.get('降水强度','—')}**"
                    f"（置信度 {conf1.get('降水强度',0.6):.0%}），"
                    f"河道水位 **{ev1.get('河道水位','—')}**"
                    f"（置信度 {conf1.get('河道水位',0.8):.0%}）"
                )

        # ── 气象证据详情（折叠） ──
        with st.expander("🌤️ 当前小时气象证据详情", expanded=False):
            ev_cols = st.columns(4)
            ev_items = list(ev.items())
            for i, (node, state) in enumerate(ev_items):
                with ev_cols[i % 4]:
                    if state == "保持先验":
                        st.markdown(f"**{node}**: {state}")
                    else:
                        color = "#2ECC71" if state in ("低", "弱", "短", "好", "小", "低温") else \
                                "#F1C40F" if state in ("中", "适温") else "#E74C3C"
                        st.markdown(f"**{node}**: <span style='color:{color}'>{state}</span>",
                                    unsafe_allow_html=True)

        # ── 风险趋势图 ──
        st.markdown("---")
        st.markdown("##### 📈 风险概率趋势（0~143 小时）")
        trend_hours = list(range(len(st.session_state["_era5_results_history"])))
        flood_high_probs = []
        geo_high_probs = []
        urban_districts = ["中原区", "二七区", "金水区", "管城回族区", "惠济区", "上街区"]
        mountain_districts = ["巩义市", "登封市", "新密市"]

        for h in trend_hours:
            rec = st.session_state["_era5_results_history"][h]
            results = rec["results"]
            f_probs = []
            for d in urban_districts:
                r = results.get(d, {})
                risk = r.get("内涝风险", {})
                if "probabilities" in risk:
                    states = risk["states"]
                    probs = risk["probabilities"]
                    high_idx = states.index("高") if "高" in states else -1
                    f_probs.append(probs[high_idx] if high_idx >= 0 else 0)
            flood_high_probs.append(np.mean(f_probs) if f_probs else 0)
            g_probs = []
            for d in mountain_districts:
                r = results.get(d, {})
                geo = r.get("地质灾害概率", {})
                if "probabilities" in geo:
                    states = geo["states"]
                    probs = geo["probabilities"]
                    high_idx = states.index("高") if "高" in states else -1
                    g_probs.append(probs[high_idx] if high_idx >= 0 else 0)
            geo_high_probs.append(np.mean(g_probs) if g_probs else 0)

        # 预测趋势
        era5_fc = load_era5_forecasts()
        fc_flood_probs = []
        fc_geo_probs = []
        if era5_fc.get("forecasts"):
            base_ev = st.session_state.get("zz_evidence", {})
            for h in trend_hours:
                fc_rec = era5_fc["forecasts"].get(str(h), {})
                h1 = fc_rec.get("horizons", {}).get("1", {})
                fc_ev = h1.get("evidence", {})
                if fc_ev:
                    fc_results = {}
                    for d in ZHENGZHOU_DISTRICTS:
                        district_ev = base_ev.get(d, {}).copy()
                        for node, state in fc_ev.items():
                            if state != "保持先验":
                                district_ev[node] = state
                        fc_results[d] = engine.infer(district_ev)
                    f_probs = []
                    for d in urban_districts:
                        r = fc_results.get(d, {})
                        risk = r.get("内涝风险", {})
                        if "probabilities" in risk:
                            states = risk["states"]
                            probs = risk["probabilities"]
                            high_idx = states.index("高") if "高" in states else -1
                            f_probs.append(probs[high_idx] if high_idx >= 0 else 0)
                    fc_flood_probs.append(np.mean(f_probs) if f_probs else 0)
                    g_probs = []
                    for d in mountain_districts:
                        r = fc_results.get(d, {})
                        geo = r.get("地质灾害概率", {})
                        if "probabilities" in geo:
                            states = geo["states"]
                            probs = geo["probabilities"]
                            high_idx = states.index("高") if "高" in states else -1
                            g_probs.append(probs[high_idx] if high_idx >= 0 else 0)
                    fc_geo_probs.append(np.mean(g_probs) if g_probs else 0)
                else:
                    fc_flood_probs.append(None)
                    fc_geo_probs.append(None)

        key_event_hour = None
        for h in trend_hours:
            if flood_high_probs[h] >= 0.5:
                key_event_hour = h
                break
        fc_key_event_hour = None
        for h in trend_hours:
            if h < len(fc_flood_probs) and fc_flood_probs[h] is not None and fc_flood_probs[h] >= 0.5:
                fc_key_event_hour = h
                break

        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=trend_hours, y=flood_high_probs,
            mode="lines", name="城区内涝 P(高) 实况", line=dict(color="#E74C3C", width=2)))
        trend_fig.add_trace(go.Scatter(x=trend_hours, y=geo_high_probs,
            mode="lines", name="山区地灾 P(高) 实况", line=dict(color="#8E44AD", width=2)))
        valid_fc = [(h, v) for h, v in zip(trend_hours, fc_flood_probs) if v is not None]
        if valid_fc:
            fc_x, fc_y = zip(*valid_fc)
            trend_fig.add_trace(go.Scatter(x=list(fc_x), y=list(fc_y),
                mode="lines", name="城区内涝 P(高) +1h 预测",
                line=dict(color="#E74C3C", width=2, dash="dash"), opacity=0.6))
        valid_fc_geo = [(h, v) for h, v in zip(trend_hours, fc_geo_probs) if v is not None]
        if valid_fc_geo:
            fc_x2, fc_y2 = zip(*valid_fc_geo)
            trend_fig.add_trace(go.Scatter(x=list(fc_x2), y=list(fc_y2),
                mode="lines", name="山区地灾 P(高) +1h 预测",
                line=dict(color="#8E44AD", width=2, dash="dash"), opacity=0.6))
        trend_fig.add_vline(x=current_hour, line_dash="dash", line_color="gray", opacity=0.5)
        if key_event_hour is not None:
            trend_fig.add_annotation(x=key_event_hour, y=0.5,
                text=f"内涝风险>50%<br>(hour {key_event_hour})",
                showarrow=True, arrowhead=1, ax=0, ay=-40,
                font=dict(size=11, color="red"), bgcolor="rgba(255,255,255,0.8)")
        if fc_key_event_hour is not None and fc_key_event_hour != key_event_hour:
            trend_fig.add_annotation(x=fc_key_event_hour, y=0.5,
                text=f"预测>50%<br>(hour {fc_key_event_hour})",
                showarrow=True, arrowhead=1, ax=0, ay=40,
                font=dict(size=10, color="darkorange"), bgcolor="rgba(255,255,255,0.8)")
        trend_fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.3)
        trend_fig.update_layout(
            title=dict(text="风险概率随时间变化（ERA5 回放）", font=dict(size=14)),
            xaxis=dict(title="小时 (0=7/18 00:00)", range=[0, 143]),
            yaxis=dict(title="P(高)", range=[0, 1.05]),
            height=350, margin=dict(l=40, r=20, t=40, b=40),
            hovermode="x unified", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(trend_fig, use_container_width=True)

        # 当前时刻各区数据表
        with st.expander("📋 当前时刻各区风险详情", expanded=False):
            rows = []
            for d in ZHENGZHOU_DISTRICTS:
                r = current_results.get(d, {})
                risk = r.get("内涝风险", {})
                geo = r.get("地质灾害概率", {})
                flood_p = "—"
                if "probabilities" in risk:
                    states = risk["states"]
                    probs = risk["probabilities"]
                    high_idx = states.index("高") if "高" in states else -1
                    flood_p = f"{probs[high_idx]*100:.1f}%" if high_idx >= 0 else "—"
                geo_p = "—"
                if "probabilities" in geo:
                    states = geo["states"]
                    probs = geo["probabilities"]
                    high_idx = states.index("高") if "高" in states else -1
                    geo_p = f"{probs[high_idx]*100:.1f}%" if high_idx >= 0 else "—"
                rows.append({"区县": d, "内涝风险 P(高)": flood_p, "地灾概率 P(高)": geo_p})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


    def _render_stream_mode(engine, geojson):
        """数据流模式：轮询 ingest_server 接收实时数据，自动更新（含图片通道）
        优化：两个独立 fragment——
          Fragment A（指标卡 / run_every=2）：轮询 + 指标卡 + 推理 + 图片识别
          Fragment B（地图+趋势 / run_every=5）：地图 + 趋势图 + 区详情
        """
        # ═══════════════════════════════════════════════════════════
        # Fragment A：指标卡（轻量高频）
        # ═══════════════════════════════════════════════════════════
        @st.fragment(run_every=2)
        def _stream_metrics_fragment():
            import threading
            import requests as _req
            status_data = {"received_count": 0, "last_data": None, "last_received_time": None,
                           "image_received_count": 0, "last_image": None, "last_image_time": None}
            try:
                resp = _req.get(f"{INGEST_SERVER_URL}/api/status", timeout=1)
                if resp.status_code == 200:
                    status_data = resp.json()
            except Exception:
                pass
            count = status_data.get("received_count", 0)
            last_data = status_data.get("last_data")
            last_time = status_data.get("last_received_time")
            image_count = status_data.get("image_received_count", 0)
            last_image = status_data.get("last_image")

            # ── JSON 通道状态 ──
            st.markdown("**📊 JSON 数据通道**")
            col_stream = st.columns([2, 1, 1])
            with col_stream[0]:
                st.metric("已接收数据条数", count)
            with col_stream[1]:
                if last_data:
                    rain = last_data.get("降水强度", "?")
                    station = last_data.get("station", "?")
                    st.metric("最近数据", f"{station} 降水={rain}")
                else:
                    st.metric("最近数据", "—")
            with col_stream[2]:
                if last_time:
                    try:
                        last_dt = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S.%f")
                        ago = (datetime.now() - last_dt).total_seconds()
                        st.metric("接收时间", f"{ago:.0f} 秒前")
                    except Exception:
                        st.metric("接收时间", last_time)
                else:
                    st.metric("接收时间", "—")

            # ── 图片通道状态 ──
            st.markdown("**📷 图片通道**")
            col_img = st.columns([2, 2, 1])
            with col_img[0]:
                st.metric("已接收图片数", image_count)
            with col_img[1]:
                if last_image:
                    img_station = last_image.get("station", "?")
                    img_task = last_image.get("task_type", "?")
                    st.metric("最近图片", f"{img_station} {img_task}")
                else:
                    st.metric("最近图片", "—")
            with col_img[2]:
                img_time = status_data.get("last_image_time")
                if img_time:
                    try:
                        img_dt = datetime.strptime(img_time, "%Y-%m-%d %H:%M:%S.%f")
                        ago = (datetime.now() - img_dt).total_seconds()
                        st.metric("接收时间", f"{ago:.0f} 秒前")
                    except Exception:
                        st.metric("接收时间", img_time)
                else:
                    st.metric("接收时间", "—")

            # ── 检测 JSON 新数据（单区推理，~0.1s） ──
            if count > st.session_state["_era5_stream_last_count"]:
                st.session_state["_era5_stream_last_count"] = count
                if last_data and last_data.get("station"):
                    station = last_data["station"]
                    district = station.replace("郑州-", "").strip()
                    if district in ZHENGZHOU_DISTRICTS:
                        current_ev = st.session_state.get("zz_evidence", {}).get(district, {}).copy()
                        for node, state in last_data.items():
                            if node in ("timestamp", "station"):
                                continue
                            if state != "保持先验":
                                current_ev[node] = state
                        st.session_state["zz_evidence"][district] = current_ev

                        # 单区推理（~0.1s），其余复用
                        prev_results = st.session_state.get("zz_results", {}).copy()
                        prev_results[district] = engine.infer(current_ev)
                        st.session_state["zz_results"] = prev_results

                        # 记录历史
                        history = st.session_state["_era5_stream_received_data"]
                        history.append(last_data)
                        if len(history) > 200:
                            history = history[-200:]
                        st.session_state["_era5_stream_received_data"] = history

                        res_history = st.session_state["_era5_stream_results_history"]
                        res_history.append(prev_results)
                        if len(res_history) > 200:
                            res_history = res_history[-200:]
                        st.session_state["_era5_stream_results_history"] = res_history

            # ── 检测图片新数据 → 启动后台线程，页面不阻塞 ──
            if image_count > st.session_state["_stream_image_last_count"] and not st.session_state.get("_stream_image_processing", False):
                st.session_state["_stream_image_last_count"] = image_count
                st.session_state["_stream_image_processing"] = True

                if last_image and last_image.get("save_path"):
                    img_path = last_image["save_path"]
                    img_station = last_image.get("station", "")
                    img_task = last_image.get("task_type", "")
                    img_file = last_image.get("original_filename", "")
                    district = img_station.replace("郑州-", "").strip()
                    if district not in ZHENGZHOU_DISTRICTS:
                        district = img_station

                    def _bg_recognize(path, stn, task, fname, dist):
                        try:
                            from tools.fuse_infer import TASK_DISPATCH
                            from tools.preprocess_api import map_to_bn_states
                            handler = TASK_DISPATCH.get(task)
                            if handler and dist in ZHENGZHOU_DISTRICTS:
                                result = handler(path)
                                if "error" not in result:
                                    evidence_kwargs = {}
                                    summary_parts = []
                                    if task == "water_level":
                                        wl = result.get("water_level_cm")
                                        if wl is not None:
                                            evidence_kwargs["water_level_cm"] = wl
                                            summary_parts.append(f"水位：{wl}cm")
                                    elif task == "road":
                                        total = result.get("total_damages", 0)
                                        evidence_kwargs["road_damage_counts"] = total
                                        summary_parts.append(f"损毁：{total}处")
                                    elif task == "flood":
                                        area = result.get("积水面积_m2")
                                        if area is not None:
                                            evidence_kwargs["flood_area_m2"] = area
                                            summary_parts.append(f"积水面积：{area:.0f}m²")

                                    bn_evidence = map_to_bn_states(**evidence_kwargs)
                                    current_ev = st.session_state.get("zz_evidence", {}).get(dist, {}).copy()
                                    for node, state in bn_evidence.items():
                                        if state != "保持先验":
                                            current_ev[node] = state
                                    st.session_state["zz_evidence"][dist] = current_ev

                                    prev_results = st.session_state.get("zz_results", {}).copy()
                                    prev_results[dist] = engine.infer(current_ev)
                                    st.session_state["zz_results"] = prev_results

                                    record = {
                                        "station": stn, "task_type": task, "filename": fname,
                                        "save_path": path, "result": result,
                                        "bn_evidence": bn_evidence, "summary": " | ".join(summary_parts),
                                    }
                                    img_records = st.session_state["_stream_image_records"]
                                    img_records.append(record)
                                    if len(img_records) > 50:
                                        img_records = img_records[-50:]
                                    st.session_state["_stream_image_records"] = img_records
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                        finally:
                            st.session_state["_stream_image_processing"] = False

                    threading.Thread(target=_bg_recognize,
                                     args=(img_path, img_station, img_task, img_file, district),
                                     daemon=True).start()

            # 显示"识别中"提示
            if st.session_state.get("_stream_image_processing", False):
                st.info("⏳ 图片识别中…（完成后自动更新）")

            # ── 显示图片识别结果摘要 ──
            img_records = st.session_state.get("_stream_image_records", [])
            if img_records:
                last_img_record = img_records[-1]
                st.markdown(f"📷 **最近识别**: {last_img_record.get('station', '?')} "
                            f"{last_img_record.get('task_type', '?')} — "
                            f"{last_img_record.get('summary', '完成')}")

            # ── 最近接收数据 expander（JSON 明细） ──
            history = st.session_state.get("_era5_stream_received_data", [])
            with st.expander(f"📋 最近接收数据（已接收 {len(history)} 条，显示最近 10 条）", expanded=False):
                if history:
                    recent = history[-10:]
                    rows = []
                    core_fields = ["timestamp", "station", "降水强度", "风力", "气温",
                                   "前期土壤含水量", "河道水位"]
                    for item in recent:
                        row = {}
                        for f in core_fields:
                            row[f] = item.get(f, "—")
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("暂无数据")

            # ── 最近图片识别 expander（图片明细） ──
            with st.expander("📷 最近图片识别（显示最近 5 条）", expanded=False):
                if img_records:
                    recent_img = img_records[-5:]
                    img_rows = []
                    for rec in recent_img:
                        img_rows.append({
                            "站点": rec.get("station", "?"),
                            "任务类型": rec.get("task_type", "?"),
                            "识别结果": rec.get("summary", "完成"),
                        })
                    st.table(pd.DataFrame(img_rows))
                else:
                    st.info("暂无图片识别记录")

        # ═══════════════════════════════════════════════════════════
        # Fragment B：地图 + 趋势图（重量低频）
        # ═══════════════════════════════════════════════════════════
        @st.fragment(run_every=5)
        def _stream_map_fragment():
            import time

            # 地图重建限频（每 5 秒最多 bump 一次）
            def _maybe_bump_map():
                now = time.time()
                last = st.session_state.get("_map_last_rebuild_time", 0.0)
                if now - last >= 5.0:
                    st.session_state["_map_cache_version"] = st.session_state.get("_map_cache_version", 0) + 1
                    st.session_state["_map_last_rebuild_time"] = now

            _maybe_bump_map()

            current_results = st.session_state.get("zz_results", {})

            # ── 地图 ──
            st.caption("🗺️ 地图每 5 秒自动更新")
            if current_results:
                fig = render_zz_map(current_results, geojson,
                                    "郑州实时监测（数据流模式）",
                                    period=st.session_state["_map_cache_version"],
                                    height=480)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("⏳ 等待数据到达… 地图将自动更新")

            # ── 趋势图（降频：数据条数增加≥3 才重建） ──
            history = st.session_state["_era5_stream_received_data"]
            res_history = st.session_state["_era5_stream_results_history"]
            trend_last_len = st.session_state.get("_stream_trend_last_len", 0)
            rebuild_trend = (len(history) - trend_last_len) >= 3
            if rebuild_trend:
                st.session_state["_stream_trend_last_len"] = len(history)

            st.markdown("---")
            st.markdown("##### 📈 风险概率趋势（实时数据流）")
            if history and res_history and rebuild_trend:
                urban_districts = ["中原区", "二七区", "金水区", "管城回族区", "惠济区", "上街区"]
                mountain_districts = ["巩义市", "登封市", "新密市"]
                trend_times = []
                flood_probs = []
                geo_probs = []

                for idx in range(len(history)):
                    data = history[idx]
                    ts = data.get("timestamp", f"#{idx}")
                    trend_times.append(ts[-8:] if len(ts) >= 8 else ts)
                    results = res_history[idx] if idx < len(res_history) else {}
                    f_probs = []
                    for d in urban_districts:
                        r = results.get(d, {})
                        risk = r.get("内涝风险", {})
                        if "probabilities" in risk:
                            states = risk["states"]
                            probs = risk["probabilities"]
                            high_idx = states.index("高") if "高" in states else -1
                            f_probs.append(probs[high_idx] if high_idx >= 0 else 0)
                    flood_probs.append(np.mean(f_probs) if f_probs else 0)
                    g_probs = []
                    for d in mountain_districts:
                        r = results.get(d, {})
                        geo = r.get("地质灾害概率", {})
                        if "probabilities" in geo:
                            states = geo["states"]
                            probs = geo["probabilities"]
                            high_idx = states.index("高") if "高" in states else -1
                            g_probs.append(probs[high_idx] if high_idx >= 0 else 0)
                    geo_probs.append(np.mean(g_probs) if g_probs else 0)

                if trend_times:
                    trend_fig = go.Figure()
                    trend_fig.add_trace(go.Scatter(
                        x=trend_times, y=flood_probs,
                        mode="lines+markers", name="城区内涝 P(高)",
                        line=dict(color="#E74C3C", width=2)))
                    trend_fig.add_trace(go.Scatter(
                        x=trend_times, y=geo_probs,
                        mode="lines+markers", name="山区地灾 P(高)",
                        line=dict(color="#8E44AD", width=2)))
                    trend_fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.3)
                    trend_fig.update_layout(
                        title=dict(text="风险概率实时变化", font=dict(size=14)),
                        xaxis=dict(title="时间"),
                        yaxis=dict(title="P(高)", range=[0, 1.05]),
                        height=350, margin=dict(l=40, r=20, t=40, b=40),
                        hovermode="x unified", legend=dict(orientation="h", y=-0.2))
                    st.plotly_chart(trend_fig, use_container_width=True)
                else:
                    st.info("⏳ 趋势数据加载中…")
            elif history and res_history:
                st.caption(f"趋势图已更新（共 {len(history)} 条数据，每增 3 条刷新）")
            else:
                st.info("⏳ 趋势数据加载中…")

            # ── 各区风险详情 ──
            if current_results:
                with st.expander("📋 当前各区风险详情", expanded=False):
                    rows = []
                    for d in ZHENGZHOU_DISTRICTS:
                        r = current_results.get(d, {})
                        risk = r.get("内涝风险", {})
                        geo = r.get("地质灾害概率", {})
                        flood_p = "—"
                        if "probabilities" in risk:
                            states = risk["states"]
                            probs = risk["probabilities"]
                            high_idx = states.index("高") if "高" in states else -1
                            flood_p = f"{probs[high_idx]*100:.1f}%" if high_idx >= 0 else "—"
                        geo_p = "—"
                        if "probabilities" in geo:
                            states = geo["states"]
                            probs = geo["probabilities"]
                            high_idx = states.index("高") if "高" in states else -1
                            geo_p = f"{probs[high_idx]*100:.1f}%" if high_idx >= 0 else "—"
                        rows.append({"区县": d, "内涝风险 P(高)": flood_p, "地灾概率 P(高)": geo_p})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── 调用两个 fragment ──
        _stream_metrics_fragment()
        _stream_map_fragment()

    # ═══════════════════════════════════════════════════════════════
    # Tab 1: 地图总览
    # ═══════════════════════════════════════════════════════════════
    if current_tab == "map":
        period = st.session_state["zz_time_period"]
        # 链式预测模式下使用链式结果
        chain_results = st.session_state["zz_chain_results_history"]
        if chain_results and period < len(chain_results):
            zz_results = chain_results[period]
        else:
            zz_results = st.session_state["zz_results"]

        if not zz_results:
            st.info("💡 尚未进行推理，请选择数据来源后点击推演按钮。")

            # 推演数据来源选择
            infer_source = st.radio(
                "推演数据来源",
                options=["demo", "prior", "manual"],
                format_func=lambda x: {
                    "demo": "演示参数（demo_params.json：真实数据证据 + 演示图片识别）",
                    "prior": "先验数据（Grid500 真实数据构建的默认证据，无图片）",
                    "manual": "当前手动参数（各区参数页手动修改的证据）",
                }.get(x, x),
                index=0,
                key="zz_map_infer_source",
            )

            # 按钮文案随来源变化
            btn_labels = {
                "demo": "🚀 使用演示参数推演全部 12 个区",
                "prior": "🚀 使用先验数据推演全部 12 个区",
                "manual": "🚀 使用手动参数推演全部 12 个区",
            }
            if st.button(btn_labels[infer_source], use_container_width=True, type="primary"):
                run_infer_all_districts(engine, infer_source)
                st.rerun()

        if zz_results:
            # 地图（带加载提示）
            map_title = "郑州各区内涝风险态势"
            with st.spinner("⏳ 正在渲染郑州地图…"):
                fig = render_zz_map(zz_results, geojson, map_title, period, height=680)
            st.plotly_chart(fig, use_container_width=True)

            # 图例
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🟢 **低风险** — 内涝风险低")
            with col2:
                st.markdown("🟡 **中风险** — 需关注")
            with col3:
                st.markdown("🔴 **高风险** — 紧急响应")

            # 各区数据表
            st.markdown("---")
            st.markdown("##### 各区风险详情")

            # 获取当前推演来源
            current_source = st.session_state.get("zz_infer_source", "prior")
            source_labels = {
                "demo": "演示参数",
                "prior": "先验",
                "manual": "手动参数",
            }
            source_label = source_labels.get(current_source, "先验")

            rows = []
            for d in ZHENGZHOU_DISTRICTS:
                top_state, top_prob, dist = get_district_risk(zz_results, d)
                geo_state, geo_prob = get_district_geo_risk(zz_results, d)
                evidence = st.session_state["zz_evidence"].get(d, {})
                uploaded = st.session_state["zz_uploaded"].get(d, {})
                has_image = "✅" if uploaded else "—"

                # 根据推演来源确定证据来源
                if current_source == "demo":
                    # 演示参数模式：有图片的区显示"图片+参数"，无图片显示"参数"
                    source = "图片+参数" if uploaded else "参数"
                elif current_source == "manual":
                    source = "手动参数" if evidence else "先验"
                else:
                    source = "先验"

                # 灾害结论
                bn_result = zz_results.get(d, {})
                confidence = st.session_state["zz_confidence"].get(d, {})
                conclusion = get_disaster_conclusion(bn_result, confidence, len(evidence))

                rows.append({
                    "区县": d, "内涝风险": f"{top_state} ({top_prob*100:.1f}%)",
                    "置信度": f"{top_prob*100:.1f}%",
                    "地灾概率": f"{geo_state} ({geo_prob*100:.1f}%)",
                    "主要威胁": conclusion["main_threat"],
                    "等级": conclusion["level"],
                    "证据来源": source,
                    "图片": has_image,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════
    # Tab 2: 区详情
    # ═══════════════════════════════════════════════════════════════
    elif current_tab == "detail":
        selected_district = st.selectbox(
            "选择区县", ZHENGZHOU_DISTRICTS,
            key="zz_district_selector",
        )

        # 侧边栏显示该区参数和上传
        with st.sidebar:
            render_district_sidebar(engine, selected_district)

        # 右侧显示详情
        zz_results = st.session_state["zz_results"]
        zz_confidence = st.session_state["zz_confidence"]
        zz_recognition = st.session_state["zz_recognition"]

        if selected_district in zz_results:
            st.markdown(f"### 📊 {selected_district} 推理结果")

            # 输出概率分布
            result = zz_results[selected_district]
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**内涝风险**")
                risk = result.get("内涝风险", {})
                if "probabilities" in risk:
                    states = risk["states"]
                    probs = risk["probabilities"]
                    fig = go.Figure(go.Bar(
                        x=states, y=probs,
                        text=[f"{p*100:.1f}%" for p in probs],
                        textposition="outside",
                        marker_color=["#2ECC71", "#F1C40F", "#E74C3C"],
                    ))
                    fig.update_layout(
                        font=dict(family="Microsoft YaHei"),
                        yaxis=dict(range=[0, 1.2], title="概率"),
                        height=250, margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**地质灾害概率**")
                geo = result.get("地质灾害概率", {})
                if "probabilities" in geo:
                    states = geo["states"]
                    probs = geo["probabilities"]
                    fig = go.Figure(go.Bar(
                        x=states, y=probs,
                        text=[f"{p*100:.1f}%" for p in probs],
                        textposition="outside",
                        marker_color=["#2ECC71", "#E74C3C"],
                    ))
                    fig.update_layout(
                        font=dict(family="Microsoft YaHei"),
                        yaxis=dict(range=[0, 1.2], title="概率"),
                        height=250, margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # 六阶段总结（含灾害结论和置信度条）
            st.markdown("---")
            render_six_stage_summary(
                zz_results, zz_confidence,
                selected_district, zz_recognition,
            )
        else:
            st.info(f"💡 {selected_district} 尚未推理，请点击左侧「推理」按钮")

    # ═══════════════════════════════════════════════════════════════
    # Tab 3: 链式预测
    # ═══════════════════════════════════════════════════════════════
    elif current_tab == "chain":
        st.markdown("### ⏱️ 链式递进预测")
        st.markdown("""
        > 将当前时段推理结果作为下一时段证据，实现灾害链式递进预测。
        > 点击「下一时段」自动应用递推规则并更新所有区概率。
        """)

        # 显示递推规则
        with st.expander("📐 当前递推规则（可扩展）", expanded=False):
            for rule in CHAIN_RULES:
                st.markdown(f"- **{rule['name']}**: {rule['description']}")

        zz_results = st.session_state["zz_results"]
        if not zz_results:
            st.warning("请先在「地图总览」页执行一键推理")
            return

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**当前时段**")
            current_period = st.session_state["zz_time_period"]
            st.markdown(f"### 时段 {current_period}")

            next_disabled = False
            if current_period >= 3:
                st.warning("已达最大递推步数（3 步）")
                next_disabled = True

            if st.button("⏩ 下一时段", type="primary", use_container_width=True,
                         disabled=next_disabled):
                with st.spinner(f"正在推演时段 {current_period + 1}…"):
                    # 初始化链式历史（首次递推时快照当前状态）
                    if not st.session_state["zz_chain_results_history"]:
                        st.session_state["zz_chain_initial_evidence"] = copy.deepcopy(
                            st.session_state["zz_evidence"]
                        )
                        st.session_state["zz_chain_initial_results"] = copy.deepcopy(
                            st.session_state["zz_results"]
                        )
                        st.session_state["zz_chain_results_history"] = [copy.deepcopy(zz_results)]
                        st.session_state["zz_chain_evidence_history"] = [copy.deepcopy(
                            st.session_state["zz_evidence"]
                        )]

                    # 对所有区执行递推
                    new_results = {}
                    new_evidence = {}
                    all_rules_triggered = {}
                    for d in ZHENGZHOU_DISTRICTS:
                        evidence = st.session_state["zz_evidence"].get(d, {})
                        current_result = zz_results.get(d, {})
                        if current_result:
                            next_ev, rules = chain_forward(evidence, current_result)
                            new_evidence[d] = next_ev
                            all_rules_triggered[d] = rules
                            new_result = engine.infer(next_ev)
                            new_results[d] = new_result
                        else:
                            new_results[d] = current_result
                            new_evidence[d] = evidence
                            all_rules_triggered[d] = []

                    # 存储链式历史（仅影响链式模块状态）
                    st.session_state["zz_chain_results_history"].append(copy.deepcopy(new_results))
                    st.session_state["zz_chain_evidence_history"].append(copy.deepcopy(new_evidence))
                    st.session_state["zz_evidence"] = new_evidence
                    st.session_state["zz_results"] = new_results
                    st.session_state["zz_time_period"] += 1
                    st.session_state["_map_cache_version"] += 1
                    st.success(f"✅ 已递推到时段 {st.session_state['zz_time_period']}")
                    st.rerun()

            if st.button("🔄 重置到时段 0", use_container_width=True):
                # 重置只影响链式预测状态，不影响地图总览和区详情
                if st.session_state["zz_chain_initial_evidence"]:
                    st.session_state["zz_evidence"] = copy.deepcopy(
                        st.session_state["zz_chain_initial_evidence"]
                    )
                if st.session_state["zz_chain_initial_results"]:
                    st.session_state["zz_results"] = copy.deepcopy(
                        st.session_state["zz_chain_initial_results"]
                    )
                st.session_state["zz_time_period"] = 0
                st.session_state["zz_chain_results_history"] = []
                st.session_state["zz_chain_evidence_history"] = []
                st.session_state["_map_cache_version"] += 1
                st.rerun()

        with col2:
            # 显示当前时段地图（链式页用较小尺寸）
            with st.spinner("⏳ 正在更新地图…"):
                fig = render_zz_map(zz_results, geojson,
                                    "郑州各区风险态势", current_period,
                                    height=480)
            st.plotly_chart(fig, use_container_width=True)

        # 概率变化对比（修复版：正确展示各时段概率对比）
        st.markdown("---")
        st.markdown("##### 概率变化对比（各时段对比）")

        chain_history = st.session_state["zz_chain_results_history"]

        if current_period > 0 and chain_history:
            # 构建对比表：每个输出节点在各时段的概率
            compare_data = []
            for d in ZHENGZHOU_DISTRICTS:
                row = {"区县": d}
                prev_probs = {}
                for t_idx, results_at_t in enumerate(chain_history):
                    if t_idx > current_period:
                        break
                    result_at_t = results_at_t.get(d, {})
                    risk = result_at_t.get("内涝风险", {})
                    if "probabilities" in risk:
                        states = risk["states"]
                        probs = risk["probabilities"]
                        high_idx = states.index("高") if "高" in states else -1
                        high_prob = probs[high_idx] if high_idx >= 0 else 0
                        prev_probs[t_idx] = high_prob
                        row[f"P(高) 时段{t_idx}"] = f"{high_prob*100:.1f}%"
                    else:
                        prev_probs[t_idx] = 0
                        row[f"P(高) 时段{t_idx}"] = "—"

                # 变化箭头
                if len(prev_probs) >= 2:
                    curr_p = prev_probs.get(current_period, 0)
                    prev_p = prev_probs.get(current_period - 1, 0)
                    diff = curr_p - prev_p
                    if diff > 0.01:
                        arrow = "↑"
                    elif diff < -0.01:
                        arrow = "↓"
                    else:
                        arrow = "→"
                    row["变化"] = f"{arrow} {abs(diff)*100:.1f}%"
                else:
                    row["变化"] = "—"

                compare_data.append(row)

            if compare_data:
                df = pd.DataFrame(compare_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("点击「下一时段」查看概率变化")
        else:
            st.info("点击「下一时段」查看概率变化对比")

        # 显示当前递推链
        if current_period > 0:
            st.markdown("---")
            st.markdown("##### 🔗 当前递推链")
            for t in range(1, current_period + 1):
                ev_history = st.session_state["zz_chain_evidence_history"]
                if t <= len(ev_history):
                    sample_district = ZHENGZHOU_DISTRICTS[0]
                    sample_ev = ev_history[t - 1].get(sample_district, {})
                    st.markdown(f"**时段 {t}**: 证据影响范围 = {len(sample_ev)} 节点")
                    st.markdown(f"  - 内涝风险高 → 前期土壤含水量高, 径流系数大, 河道水位危险")
                    st.markdown(f"  - 地灾概率高 → 地质易发性高, 滑坡历史密度高, 土壤渗透性差")

    # ═══════════════════════════════════════════════════════════════
    # Tab 4 (新): 实时监测（ERA5 回放模拟器 + 数据流模式）
    # ═══════════════════════════════════════════════════════════════
    elif current_tab == "era5":
        st.markdown("### 📡 实时监测")
        st.markdown("""
        > 两种模式：
        > **回放模式** — 以 ERA5 逐小时数据（144 小时）按时间戳回放；
        > **数据流模式** — 模拟数据源持续发包 → FastAPI 接收 → 自动更新。
        """)

        # ── 模式选择 ──
        stream_mode = st.radio(
            "模式选择",
            options=["回放模式", "数据流模式"],
            index=0 if st.session_state["_era5_stream_mode"] == "replay" else 1,
            horizontal=True,
            key="era5_mode_radio",
        )
        if stream_mode == "数据流模式" and st.session_state["_era5_stream_mode"] != "stream":
            st.session_state["_era5_stream_mode"] = "stream"
            st.session_state["_era5_stream_last_count"] = 0
            st.session_state["_stream_mode_entering"] = True
            # 不调用 st.rerun()——radio 自带自动 rerun，避免双重 rerun
        elif stream_mode == "回放模式" and st.session_state["_era5_stream_mode"] != "replay":
            st.session_state["_era5_stream_mode"] = "replay"
            # 不调用 st.rerun()——radio 自带自动 rerun，避免双重 rerun

        # ── 加载 ERA5 证据（回放模式需要） ──
        if st.session_state["_era5_stream_mode"] == "replay":
            era5_data = load_era5_evidence()
            records = era5_data.get("records", [])
            if not records:
                st.warning("ERA5 证据数据为空，请先运行 scripts/build_era5_timeseries.py")
                return

        # ═══════════════════════════════════════════════════════════
        # 回放模式（现有逻辑）
        # ═══════════════════════════════════════════════════════════
        if st.session_state["_era5_stream_mode"] == "replay":
            _render_replay_mode(engine, geojson, records)

        # ═══════════════════════════════════════════════════════════
        # 数据流模式
        # ═══════════════════════════════════════════════════════════
        else:
            # ── 首次进入数据流模式时显示状态提示（页面级，只显示一次） ──
            if st.session_state.get("_stream_mode_entering", False):
                stream_status = st.status("正在进入数据流模式…", expanded=True)
                try:
                    import requests
                    resp = requests.get(f"{INGEST_SERVER_URL}/api/status", timeout=1)
                    if resp.status_code == 200:
                        stream_status.update(label="✅ 数据流模式就绪（已连接接收服务）",
                                             state="complete", expanded=False)
                    else:
                        stream_status.update(label="⚠️ 接收服务响应异常",
                                             state="error", expanded=False)
                except Exception:
                    stream_status.update(
                        label="⚠️ 未检测到接收服务（8502），请先启动 ingest_server.py",
                        state="error", expanded=False)
                st.session_state["_stream_mode_entering"] = False

            _render_stream_mode(engine, geojson)

    # ═══════════════════════════════════════════════════════════════
    # Tab 5: 参数（原 Tab 4）
    # ═══════════════════════════════════════════════════════════════
    elif current_tab == "params":
        st.markdown("### ⚙️ 全局参数管理")

        # 一键重置
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 一键重置所有区为先验", use_container_width=True, type="primary"):
                st.session_state["zz_evidence"] = {}
                st.session_state["zz_uploaded"] = {}
                st.session_state["zz_results"] = {}
                st.session_state["zz_confidence"] = {}
                st.session_state["zz_recognition"] = {}
                st.session_state["zz_time_period"] = 0
                st.session_state["zz_chain_results_history"] = []
                st.session_state["zz_chain_evidence_history"] = []
                st.success("✅ 已重置所有区为先验状态")
                st.rerun()

        with col2:
            # 推演数据来源选择（紧凑型 radio）
            params_infer_source = st.radio(
                "推演数据来源",
                options=["manual", "demo", "prior"],
                format_func=lambda x: {
                    "demo": "演示参数（含图片识别）",
                    "prior": "先验数据（纯 BN 推理）",
                    "manual": "当前手动参数",
                }.get(x, x),
                index=0,
                key="zz_params_infer_source",
                label_visibility="collapsed",
            )
            btn_labels = {
                "demo": "🚀 用演示参数推演全部区",
                "prior": "🚀 用先验数据推演全部区",
                "manual": "🚀 用手动参数推演全部区",
            }
            if st.button(btn_labels[params_infer_source], use_container_width=True, type="secondary"):
                run_infer_all_districts(engine, params_infer_source)
                st.rerun()

        st.markdown("---")

        # 各区参数编辑
        selected_district = st.selectbox(
            "选择区县编辑参数", ZHENGZHOU_DISTRICTS,
            key="zz_param_district",
        )

        evidence = st.session_state["zz_evidence"].get(selected_district, {})
        current_evidence = dict(evidence) if evidence else {}

        input_params = engine.get_input_params()
        # 按类别分组
        categories = {"气象": [], "水文": [], "地形": [], "地质": [], "城市": []}
        for p in input_params:
            cfg = engine.get_node_config(p)
            cat = cfg.get("category", "其他")
            if cat in categories:
                categories[cat].append(p)
            else:
                categories.setdefault("其他", []).append(p)

        cat_labels = {
            "气象": "🌤️ 气象", "水文": "💧 水文", "地形": "⛰️ 地形",
            "地质": "🪨 地质", "城市": "🏙️ 城市",
        }

        for cat, params in categories.items():
            if not params:
                continue
            with st.expander(f"{cat_labels.get(cat, cat)} ({len(params)})", expanded=False):
                cols = st.columns(2)
                for i, pname in enumerate(params):
                    with cols[i % 2]:
                        cfg = engine.get_node_config(pname)
                        states = ["先验/不指定"] + cfg["states"]
                        current_val = current_evidence.get(pname, "先验/不指定")
                        default_idx = states.index(current_val) if current_val in states else 0
                        selected = st.selectbox(
                            f"**{pname}**",
                            states,
                            index=default_idx,
                            key=f"zz_global_{selected_district}_{pname}",
                        )
                        if selected != "先验/不指定":
                            current_evidence[pname] = selected
                        elif pname in current_evidence:
                            del current_evidence[pname]

        st.session_state["zz_evidence"][selected_district] = current_evidence

    # ═══════════════════════════════════════════════════════════════
    # Tab 5: 帮助
    # ═══════════════════════════════════════════════════════════════
    elif current_tab == "help":
        st.markdown("### ℹ️ 使用帮助")
        st.markdown("""
        **郑州全景可视化 Dashboard**

        1. **🗺️ 地图总览** — 查看郑州各区内涝风险态势
           - 色块颜色表示风险等级（绿/黄/红）
           - 色块透明度反映置信度（越实越可信）
           - 地图上标注"风险态+概率%"
           - hover 显示详情（内涝分布、地灾概率、灾害结论）
           - 下方数据表展示各区风险详情

        2. **📋 区详情** — 选择区县，查看详细结果
           - 侧边栏上传图片（水位尺/道路/洪水现场）
           - 编辑 BN 参数（按类别分组）
           - 点击推理按钮执行识别+推理
           - 显示概率分布图 + 灾害结论卡片 + 六阶段结果总结

        3. **⏱️ 链式预测** — 递进式灾害预测
           - 点击「下一时段」将当前推理结果递推为证据
           - 递推规则：内涝高→土壤含水量高→地灾概率上升
           - 概率对比表：各时段概率差值 + 变化箭头（↑/↓/→）
           - 重置只影响链式状态，不影响地图和区详情

        4. **⚙️ 参数管理** — 全局编辑各区参数
           - 一键重置所有区为先验
           - 一键推理全部区（带进度条）
           - 各区参数按类别分组编辑

        **数据来源说明**
        - 🟢 真实识别 → 上传图片后视觉识别结果（带置信度条）
        - 🟡 手动参数 → 手动设定的 BN 参数
        - 🔵 先验 → 默认演示数据（7·20 暴雨场景）

        **技术栈**
        - Streamlit + Plotly (Choropleth 地图)
        - 贝叶斯网络推理引擎 (40 节点)
        - 基础设施灾损视觉识别 (水位/道路/洪水)
        """)


# ============================================================================
# 主入口
# ============================================================================

def main():
    st.set_page_config(
        page_title="灾害链推理引擎 v2 — 郑州全景可视化",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # ── 侧边栏：固定为郑州模式，无模式选择器 ──
    with st.sidebar:
        st.markdown("## 🌊 灾害链推理引擎")
        st.markdown("**郑州全景可视化**")
        st.markdown("---")
        st.markdown("**版本**: v2.0-郑州")
        st.markdown("---")
        st.markdown("**快速操作**")
        if st.button("🗺️ 地图总览", use_container_width=True, key="sb_map"):
            st.session_state["zz_tab"] = "map"
            st.rerun()
        if st.button("📋 区详情", use_container_width=True, key="sb_detail"):
            st.session_state["zz_tab"] = "detail"
            st.rerun()
        if st.button("⏱️ 链式预测", use_container_width=True, key="sb_chain"):
            st.session_state["zz_tab"] = "chain"
            st.rerun()
        if st.button("⚙️ 参数", use_container_width=True, key="sb_params"):
            st.session_state["zz_tab"] = "params"
            st.rerun()

        st.markdown("---")
        st.markdown("**演示控制**")
        if st.session_state.get("_demo_params_loaded"):
            st.info("✅ 演示参数已加载")
            if st.button("🗑️ 清除演示参数", use_container_width=True, key="sb_clear_demo"):
                st.session_state["zz_evidence"] = {}
                st.session_state["zz_uploaded"] = {}
                st.session_state["zz_recognition"] = {}
                st.session_state["zz_results"] = {}
                st.session_state["zz_confidence"] = {}
                st.session_state["_demo_params_loaded"] = False
                st.session_state["_map_cache_version"] += 1
                st.rerun()

    # ── 主页面 ──
    render_zz_page()


if __name__ == "__main__":
    main()