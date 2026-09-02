"""
统一预处理接口：融合图像预处理 + 文本预处理
提供 process_image() / process_text() 函数接口 + CLI
预处理结果可映射到 BN 节点状态（如 积水面积→内涝深度 状态）

可被 bn_engine/dashboard 调用，不修改现有逻辑
"""

import os
import json
import argparse
from typing import Dict, Optional, List, Union

from tools.image_preprocess import ImageSegmenter, FloodDetector, PIXEL_AREA_M2
from tools.text_preprocess import process_text as _process_text

# ==================== 全局单例 ====================
_flood_segmenter: Optional[ImageSegmenter] = None
_landslide_segmenter: Optional[ImageSegmenter] = None
_flood_detector: Optional[FloodDetector] = None


def _get_flood_segmenter(model_path: Optional[str] = None) -> ImageSegmenter:
    """获取内涝分割器（单例）"""
    global _flood_segmenter
    if _flood_segmenter is None:
        _flood_segmenter = ImageSegmenter(model_path, task="flood")
    return _flood_segmenter


def _get_landslide_segmenter(model_path: Optional[str] = None) -> ImageSegmenter:
    """获取滑坡分割器（单例）"""
    global _landslide_segmenter
    if _landslide_segmenter is None:
        _landslide_segmenter = ImageSegmenter(model_path, task="landslide")
    return _landslide_segmenter


def _get_flood_detector(model_path: Optional[str] = None) -> FloodDetector:
    """获取洪水目标检测器（单例）"""
    global _flood_detector
    if _flood_detector is None:
        _flood_detector = FloodDetector(model_path)
    return _flood_detector


# ==================== 图像预处理接口 ====================

def process_image(image_path: str, task: str = "auto",
                  model_path: Optional[str] = None) -> Dict:
    """
    处理单张灾情图像，返回结构化字段

    Args:
        image_path: 图片路径
        task: "flood" / "landslide" / "flood_detect" / "auto"（自动判断）
        model_path: 模型权重路径，None 使用默认

    Returns:
        task="flood" 或 "landslide":
            {
                "积水面积_m2": float,    # 仅 flood 任务
                "淹没占比": float,
                "滑坡面积_m2": float,    # 仅 landslide 任务
                "灾情等级": str,
                "推理耗时_s": float,
            }
        task="flood_detect":
            {
                "洪水目标数": int,
                "最大洪水框面积比例": float,
                "类别分布": dict,
                "推理耗时_s": float,
            }
    """
    if task == "flood_detect":
        detector = _get_flood_detector(model_path)
        raw = detector.detect(image_path)
        return {
            "洪水目标数": raw["target数"],
            "最大洪水框面积比例": raw["最大框面积比例"],
            "类别分布": raw["类别分布"],
            "推理耗时_s": raw["推理耗时_s"],
        }

    # 自动判断任务类型
    if task == "auto":
        # 根据图片路径关键词判断
        path_lower = image_path.lower()
        if "landslide" in path_lower or "滑坡" in path_lower:
            task = "landslide"
        elif "flood" in path_lower or "floodnet" in path_lower or "积水" in path_lower:
            task = "flood"
        else:
            task = "flood"  # 默认 flood

    if task == "flood":
        segmenter = _get_flood_segmenter(model_path)
    else:
        segmenter = _get_landslide_segmenter(model_path)

    result = segmenter.predict(image_path)
    return result


def process_image_batch(image_paths: List[str], task: str = "auto",
                        model_path: Optional[str] = None) -> List[Dict]:
    """批量处理多张图像"""
    results = []
    for path in image_paths:
        try:
            res = process_image(path, task, model_path)
            results.append(res)
        except Exception as e:
            results.append({"image_path": path, "error": str(e)})
    return results


# ==================== 文本预处理接口 ====================

def process_text(text: str) -> Dict:
    """
    处理单条灾情文本，返回结构化字段

    Args:
        text: 中文文本

    Returns:
        {
            "灾害类型": str,       # 内涝/滑坡/洪水/台风/暴雨/森林火灾/爆炸/地震/其他
            "地点": str or None,
            "时间": str or None,
            "关键词": list[str],
            "置信度": float,
        }
    """
    return _process_text(text)


# ==================== BN 节点状态映射 ====================

def map_to_bn_states(flood_area_m2: Optional[float] = None,
                     landslide_area_m2: Optional[float] = None,
                     disaster_type: Optional[str] = None,
                     flood_detected_objects: Optional[int] = None,
                     water_level_cm: Optional[float] = None,
                     road_damage_counts: Optional[int] = None) -> Dict:
    """
    将预处理结果映射到 BN 节点状态

    映射规则（与 configs/config_40nodes.yaml 状态定义一致）：

      积水面积 → 内涝深度 [无, 浅, 深]：
        < 100 m²  → "无"
        100~2000 m² → "浅"
        ≥ 2000 m² → "深"

      滑坡面积 → 滑坡历史密度 [低, 中, 高]：
        < 40 m² → 不设置（保持先验）
        40~120 m² → "低"
        120~300 m² → "中"
        ≥ 300 m² → "高"

      水位 cm → 河道水位 [正常, 警戒, 危险]（阈值可调）：
        < 120 cm → "正常"
        120~150 cm → "警戒"
        ≥ 150 cm → "危险"

      道路损毁总数 → 道路积水历史频率 [低, 中, 高]：
        0 处 → "低"
        1~3 处 → "中"
        ≥ 4 处 → "高"

      道路损毁总数 → 管网排水能力 [强, 弱]：
        ≥ 1 处 → "弱"（否则保持先验）

      灾害类型 → 灾害类型节点：
        内涝/洪水/暴雨 → "洪涝"
        滑坡 → "滑坡"
        台风 → "台风"
        森林火灾 → "火灾"
        爆炸 → "爆炸"
        其他 → "无"

      flood_detect 目标数 → 现场证据补充

    Args:
        flood_area_m2: 积水面积（m²），来自 flood 分割任务
        landslide_area_m2: 滑坡面积（m²），来自 landslide 分割任务
        disaster_type: 文本识别灾害类型
        flood_detected_objects: 洪水目标检测数
        water_level_cm: 水位（cm），来自基础设施水位识别
        road_damage_counts: 道路损毁总数（D00~D40 合计），来自基础设施道路检测

    Returns:
        dict of {node_name: state_value}
    """
    states = {}

    # 内涝深度 [无, 浅, 深]（config_40nodes.yaml 3态）
    if flood_area_m2 is not None:
        if flood_area_m2 < 100:
            states["内涝深度"] = "无"
        elif flood_area_m2 < 2000:
            states["内涝深度"] = "浅"
        else:
            states["内涝深度"] = "深"

    # 滑坡历史密度 [低, 中, 高]（config_40nodes.yaml 3态，根节点）
    if landslide_area_m2 is not None:
        if landslide_area_m2 < 40:
            pass  # 保持先验
        elif landslide_area_m2 < 120:
            states["滑坡历史密度"] = "低"
        elif landslide_area_m2 < 300:
            states["滑坡历史密度"] = "中"
        else:
            states["滑坡历史密度"] = "高"

    # 河道水位 [正常, 警戒, 危险]（config_40nodes.yaml 3态）
    # 阈值后续可通过参数调整
    if water_level_cm is not None:
        if water_level_cm < 120:
            states["河道水位"] = "正常"
        elif water_level_cm < 150:
            states["河道水位"] = "警戒"
        else:
            states["河道水位"] = "危险"

    # 道路积水历史频率 [低, 中, 高]（config_40nodes.yaml 3态）
    if road_damage_counts is not None:
        if road_damage_counts == 0:
            states["道路积水历史频率"] = "低"
        elif road_damage_counts <= 3:
            states["道路积水历史频率"] = "中"
        else:
            states["道路积水历史频率"] = "高"

        # 管网排水能力 [强, 弱]（config_40nodes.yaml 2态）
        if road_damage_counts >= 1:
            states["管网排水能力"] = "弱"
        # 否则保持先验

    # 灾害类型
    if disaster_type:
        type_map = {
            "内涝": "洪涝",
            "洪水": "洪涝",
            "暴雨": "洪涝",
            "滑坡": "滑坡",
            "台风": "台风",
            "森林火灾": "火灾",
            "爆炸": "爆炸",
            "地震": "地震",
            "其他": "无",
        }
        states["灾害类型"] = type_map.get(disaster_type, "无")

    # 洪水目标检测（flood_detect）→ 现场证据补充
    if flood_detected_objects is not None:
        if flood_detected_objects > 5:
            states["现场证据"] = "有多处灾害迹象"
        elif flood_detected_objects > 0:
            states["现场证据"] = "有灾害迹象"
        else:
            states["现场证据"] = "无"

    return states


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="灾害链预处理统一接口")
    parser.add_argument("--mode", choices=["image", "text", "batch_image", "batch_text"],
                        default="image", help="处理模式")
    parser.add_argument("--image", type=str, help="图片路径")
    parser.add_argument("--text", type=str, help="文本内容")
    parser.add_argument("--task", choices=["flood", "landslide", "flood_detect", "auto"], default="auto",
                        help="图像任务类型")
    parser.add_argument("--model", type=str, help="模型权重路径")
    parser.add_argument("--images", type=str, nargs="+", help="批量图片路径")
    parser.add_argument("--output", type=str, help="输出 JSON 文件路径")
    args = parser.parse_args()

    if args.mode == "image":
        if not args.image:
            print("请指定 --image 图片路径")
            return
        result = process_image(args.image, args.task, args.model)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # 输出 BN 映射
        if "积水面积_m2" in result:
            bn_states = map_to_bn_states(flood_area_m2=result.get("积水面积_m2"))
        elif "滑坡面积_m2" in result:
            bn_states = map_to_bn_states(landslide_area_m2=result.get("滑坡面积_m2"))
        elif "洪水目标数" in result:
            bn_states = map_to_bn_states(flood_detected_objects=result.get("洪水目标数"))
        else:
            bn_states = {}
        if bn_states:
            print("\n→ BN 节点状态映射:")
            print(json.dumps(bn_states, ensure_ascii=False, indent=2))

    elif args.mode == "text":
        if not args.text:
            # 交互式
            print("输入文本（输入空行退出）：")
            while True:
                text = input("> ").strip()
                if not text:
                    break
                result = process_text(text)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                # BN 映射
                bn_states = map_to_bn_states(disaster_type=result.get("灾害类型"))
                if bn_states:
                    print("→ BN 节点状态映射:", json.dumps(bn_states, ensure_ascii=False))
                print()
        else:
            result = process_text(args.text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            bn_states = map_to_bn_states(disaster_type=result.get("灾害类型"))
            if bn_states:
                print("\n→ BN 节点状态映射:")
                print(json.dumps(bn_states, ensure_ascii=False, indent=2))

    elif args.mode == "batch_image":
        if not args.images:
            print("请指定 --images 图片路径列表")
            return
        results = process_image_batch(args.images, args.task, args.model)
        output = json.dumps(results, ensure_ascii=False, indent=2)
        print(output)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"结果已保存到: {args.output}")

    elif args.mode == "batch_text":
        from tools.text_preprocess import batch_process
        output_path = args.output or os.path.join(
            os.path.dirname(__file__), "..", "output", "text_preprocess_result.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        batch_process(output_path)


if __name__ == "__main__":
    main()