"""
验证脚本：用 8 节点基线配置跑 4 个场景，与实验1 结果对比
"""
import json
from bn_engine import DisasterChainEngine

engine = DisasterChainEngine("configs/config_8nodes.yaml")

# 实验1 的 4 个场景
scenarios = [
    ("正常天气", {"降水强度": "低", "风力": "弱"}),
    ("暴雨（高降水+强风）", {"降水强度": "高", "风力": "强", "风向": "向岸"}),
    ("暴雨 + 低洼 + 排水慢（最不利）", {"降水强度": "高", "风力": "强", "海拔": "低", "历史排水时间": "慢"}),
    ("暴雨 + 陡坡 + 植被差（地灾场景）", {"降水强度": "高", "坡度": "陡", "植被覆盖": "差", "土壤渗透性": "差"}),
]

# 实验1 参考结果（来自 basic_inference.txt）
expected = {
    "正常天气": {"内涝风险": 0.489, "地质灾害概率": 0.552},
    "暴雨（高降水+强风）": {"内涝风险": 0.799, "地质灾害概率": 0.706},
    "暴雨 + 低洼 + 排水慢（最不利）": {"内涝风险": 0.806, "地质灾害概率": 0.663},
    "暴雨 + 陡坡 + 植被差（地灾场景）": {"内涝风险": 0.799, "地质灾害概率": 0.849},
}

print("=" * 80)
print(f"{'场景':<30s} {'输出':<12s} {'实验1':>8s} {'v2引擎':>8s} {'Δ':>8s} {'匹配':>6s}")
print("=" * 80)

all_match = True
for name, evidence in scenarios:
    result = engine.infer(evidence)
    e = expected[name]

    for output_name in ["内涝风险", "地质灾害概率"]:
        # 找 "高" 状态的概率
        node_res = result[output_name]
        for s, p in zip(node_res["states"], node_res["probabilities"]):
            if s == "高":
                v2_p = p
                break
        exp_p = e[output_name]

        delta = round(v2_p - exp_p, 4)
        match = "✓" if abs(delta) < 0.01 else "✗"
        if match == "✗":
            all_match = False
        print(f"{name:<30s} {output_name:<12s} {exp_p:>8.3f} {v2_p:>8.3f} {delta:>+8.4f} {match:>6s}")

print("=" * 80)
print(f"整体结果: {'✅ 全部匹配' if all_match else '❌ 存在差异'}")
print()

# 输出详细结果
if not all_match:
    print("详细差异分析:")
    for name, evidence in scenarios:
        result = engine.infer(evidence)
        print(f"\n[{name}]")
        print(f"  证据: {json.dumps(evidence, ensure_ascii=False)}")
        for output_name in ["内涝风险", "地质灾害概率"]:
            node_res = result[output_name]
            for s, p in zip(node_res["states"], node_res["probabilities"]):
                if s == "高":
                    print(f"  P({output_name}=高) = {p:.4f}")