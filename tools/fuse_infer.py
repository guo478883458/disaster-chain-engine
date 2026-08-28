"""
综合推理入口：融合基础设施灾损识别 + 灾害链推理引擎

架构：
  [基础设施项目] process_infra_image(图, task=水位/道路/洪水) → 结构化字段
         ↓  (v2 跨项目调用)
  [v2 项目]  map_to_bn_states(扩展) → 证据字典 → DisasterChainEngine.infer(evidence)
         ↓
   内涝风险 / 地质灾害概率 输出

用法：
    python -m tools.fuse_infer --tasks 水位图.jpg:water_level 道路图.jpg:road 洪水图.jpg:flood
    python -m tools.fuse_infer --config 任务配置.json
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional

# ── 跨项目调用：基础设施灾损识别 ──
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import INFRA_PROJECT_DIR, PACKAGE_ROOT

INFRA_ROOT = INFRA_PROJECT_DIR
if INFRA_ROOT not in sys.path:
    sys.path.insert(0, INFRA_ROOT)

# v2 项目自身
V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if V2_ROOT not in sys.path:
    sys.path.insert(0, V2_ROOT)

from bn_engine import DisasterChainEngine
from tools.preprocess_api import map_to_bn_states


# ==================== 基础设施推理 ====================

def _infer_water_level(image_path: str) -> dict:
    """调用基础设施水位识别"""
    from tools.infer_api import process_infra_image
    try:
        result = process_infra_image(image_path, task="water_level")
        return result
    except Exception as e:
        return {"error": str(e), "water_level_cm": None}


def _infer_road_damage(image_path: str) -> dict:
    """调用基础设施道路损毁检测"""
    from tools.infer_api import process_infra_image
    try:
        result = process_infra_image(image_path, task="road")
        return result
    except Exception as e:
        return {"error": str(e), "total_damages": 0, "class_counts": {}}


def _infer_flood(image_path: str) -> dict:
    """调用基础设施洪水分割"""
    from tools.infer_api import process_infra_image
    try:
        result = process_infra_image(image_path, task="flood")
        return result
    except Exception as e:
        return {"error": str(e), "积水面积_m2": None}


# ==================== 任务调度 ====================

TASK_DISPATCH = {
    "water_level": _infer_water_level,
    "road": _infer_road_damage,
    "road_bridge": _infer_road_damage,
    "flood": _infer_flood,
    "flood_seg": _infer_flood,
}


def run_tasks(tasks: List[Tuple[str, str]]) -> Tuple[list, dict, list, dict]:
    """
    逐张图像执行推理

    Args:
        tasks: [(image_path, task_type), ...]

    Returns:
        (识别明细列表, 收集的证据字段, 缺失证据列表)
    """
    details = []
    evidence_kwargs = {}
    missing_evidence = []

    for img_path, task_type in tasks:
        if not os.path.exists(img_path):
            missing_evidence.append({
                "image_path": img_path,
                "task": task_type,
                "reason": "文件不存在",
            })
            continue

        handler = TASK_DISPATCH.get(task_type)
        if handler is None:
            missing_evidence.append({
                "image_path": img_path,
                "task": task_type,
                "reason": f"不支持的任务类型: {task_type}",
            })
            continue

        result = handler(img_path)
        detail = {
            "image_path": img_path,
            "task": task_type,
        }

        if "error" in result:
            detail["status"] = "failed"
            detail["error"] = result["error"]
            missing_evidence.append({
                "image_path": img_path,
                "task": task_type,
                "reason": result["error"],
            })
        else:
            detail["status"] = "success"
            detail["result"] = {k: v for k, v in result.items()
                                if k not in ("task",)}

            # 收集证据字段
            if task_type == "water_level":
                wl = result.get("water_level_cm")
                if wl is not None:
                    evidence_kwargs["water_level_cm"] = wl
                else:
                    missing_evidence.append({
                        "image_path": img_path,
                        "task": task_type,
                        "reason": "水位识别返回 None",
                    })
            elif task_type in ("road", "road_bridge"):
                total = result.get("total_damages", 0)
                evidence_kwargs["road_damage_counts"] = total
            elif task_type in ("flood", "flood_seg"):
                area = result.get("积水面积_m2")
                if area is not None:
                    evidence_kwargs["flood_area_m2"] = area
                else:
                    missing_evidence.append({
                        "image_path": img_path,
                        "task": task_type,
                        "reason": "洪水分割返回 None",
                    })

        details.append(detail)

    # 提取置信度摘要
    confidence_summary = _extract_confidence(details)

    return details, evidence_kwargs, missing_evidence, confidence_summary


def _extract_confidence(details: list) -> dict:
    """从识别结果中提取置信度摘要"""
    confidence = {}
    for d in details:
        if d.get("status") != "success":
            continue
        task = d["task"]
        res = d.get("result", {})
        if task == "water_level":
            confidence["water_level"] = {
                "value_cm": res.get("water_level_cm"),
                "confidence": res.get("confidence", 0.0),
                "source": "visual_recognition",
            }
        elif task in ("road", "road_bridge"):
            dets = res.get("detections", [])
            avg_conf = round(
                sum(dd.get("confidence", 0.0) for dd in dets) / len(dets), 4
            ) if dets else 0.0
            confidence["road_damage"] = {
                "total": res.get("total_damages", 0),
                "avg_confidence": avg_conf,
                "per_class": {dd["class_name"]: dd.get("confidence", 0.0)
                              for dd in dets[:5]},
                "source": "visual_recognition",
            }
        elif task in ("flood", "flood_seg"):
            confidence["flood"] = {
                "area_m2": res.get("积水面积_m2"),
                "inundation_ratio": res.get("淹没占比"),
                "disaster_level": res.get("灾情等级"),
                "source": "visual_recognition",
            }
    return confidence


# ==================== 综合推理 ====================

def fuse_infer(tasks: List[Tuple[str, str]],
               config_path: Optional[str] = None) -> dict:
    """
    综合推理入口

    Args:
        tasks: [(image_path, task_type), ...]
        config_path: BN 配置文件路径，默认使用 40 节点

    Returns:
        {
            "识别结果明细": [...],
            "证据字典": {...},
            "证据缺失": [...],
            "BN推理结果": {...},
            "结论": {...},
        }
    """
    if config_path is None:
        config_path = os.path.join(V2_ROOT, "configs", "config_40nodes.yaml")

    # 1. 逐张识别
    details, evidence_kwargs, missing_evidence, confidence_summary = run_tasks(tasks)

    # 2. 组装证据
    evidence = map_to_bn_states(**evidence_kwargs)

    # 3. BN 推理
    engine = DisasterChainEngine(config_path)
    bn_result = engine.infer(evidence)

    # 4. 提取结论（风险最高态）
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
                "全部分布": dict(zip(states, [round(p, 4) for p in probs])),
            }

    return {
        "识别结果明细": details,
        "证据字典": evidence,
        "证据缺失": missing_evidence,
        "置信度摘要": confidence_summary,
        "BN推理结果": bn_result,
        "结论": conclusion,
    }


# ==================== CLI ====================

def parse_task_arg(arg: str) -> Tuple[str, str]:
    """解析 '路径:任务类型' 格式"""
    parts = arg.split(":", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return parts[0].strip(), "flood"  # 默认 flood


def main():
    parser = argparse.ArgumentParser(
        description="综合推理：图像识别 + 贝叶斯网络推理"
    )
    parser.add_argument("--tasks", "-t", nargs="+", metavar="路径:任务类型",
                        help="任务列表，如 水位图.jpg:water_level 道路图.jpg:road")
    parser.add_argument("--config", "-c", type=str,
                        default=os.path.join(V2_ROOT, "configs", "config_40nodes.yaml"),
                        help="BN 配置文件路径")
    parser.add_argument("--output", "-o", type=str, help="输出 JSON 文件路径")
    parser.add_argument("--demo", "-d", action="store_true",
                        help="使用内置演示数据运行")

    args = parser.parse_args()

    if args.demo:
        # 内置演示：使用真实测试数据
        tasks = [
            (r"H:\dev\disaster-data\infra_datasets\water_level\extracted\SAM_water_level_Dataset\Staff gauge images\20230527\images\P23052700112410.jpg",
             "water_level"),
            (r"H:\dev\disaster-data\infra_datasets\rdd2020\test1\Czech\images\Czech_000004.jpg",
             "road"),
            (r"H:\dev\disaster-data\image_datasets\floodimg\Flood Images\flood_0.jpg",
             "flood"),
        ]
    elif args.tasks:
        tasks = [parse_task_arg(t) for t in args.tasks]
    else:
        parser.print_help()
        return

    result = fuse_infer(tasks, config_path=args.config)

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()