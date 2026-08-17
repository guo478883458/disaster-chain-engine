# -*- coding: utf-8 -*-
"""
郑州 7·20 灾害链情景推理演示脚本
==================================
用法（无需任何参数）：
    D:\\ana\\envs\\disasterlex\\python.exe demo_infer.py

内置 4 个场景对比（正常天气 / 郑州暴雨 / 内涝最不利 / 地灾场景），
再加一个"无证据"场景展示弹性输入（先验输出）。
输出：每个场景下链条各环节的概率分布。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bn_engine import DisasterChainEngine

ENGINE = DisasterChainEngine("configs/config_40nodes.yaml")

SCENES = [
    ("① 无证据（弹性输入：什么都不提供）", {}),
    ("② 正常天气", {"降水强度": "低", "风力": "弱"}),
    ("③ 郑州 7·20 暴雨（降水高+风强+低洼+排水慢）",
     {"降水强度": "高", "风力": "强", "海拔": "低", "历史排水时间": "慢"}),
    ("④ 暴雨 + 陡坡 + 植被差（地灾场景）",
     {"降水强度": "高", "坡度": "陡", "植被覆盖": "差"}),
]

OUTPUT_NODES = ["内涝风险", "地质灾害概率"]
CHAIN_NODES = ["暴雨强度", "内涝深度", "地质易发性"]


def fmt_dist(dist: dict) -> str:
    """把 {states, probabilities} 格式化输出"""
    if "error" in dist:
        return f"（证据节点，跳过: {dist['error'][:20]}…）"
    pairs = zip(dist.get("states", []), dist.get("probabilities", []))
    return "  ".join(f"{k}={v * 100:.1f}%" for k, v in pairs)


def main() -> None:
    print("=" * 70)
    print("  灾害链推理引擎 v2（40 节点贝叶斯网络）— 郑州 7·20 情景演示")
    print("=" * 70)
    for name, evidence in SCENES:
        print(f"\n【{name}】")
        print(f"  证据: {evidence if evidence else '（无，全部保持先验）'}")
        result = ENGINE.infer(evidence)
        print(f"  链条环节: {fmt_dist(result['暴雨强度'])}")
        print(f"           {fmt_dist(result['内涝深度'])}")
        print(f"           {fmt_dist(result['地质易发性'])}")
        print(f"  输出风险: {fmt_dist(result['内涝风险'])}")
        print(f"           {fmt_dist(result['地质灾害概率'])}")


if __name__ == "__main__":
    main()
