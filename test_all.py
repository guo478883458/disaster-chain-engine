"""
综合测试：弹性输入 + 敏感性分析 + JSON 导出
"""
import json
from bn_engine import DisasterChainEngine

# 1. 20节点 - 弹性输入测试
print("=" * 70)
print("测试1: 20节点 - 弹性输入（仅传入降水强度=高）")
print("=" * 70)
engine = DisasterChainEngine("configs/config_20nodes.yaml")
result = engine.infer({"降水强度": "高"})
meta = result.pop("_meta")
print(f"证据覆盖度: {meta['evidence_coverage']*100:.1f}% ({meta['evidence_provided']}/{meta['evidence_expected']})")
print(f"未提供参数: {', '.join(meta['missing_params'])}")
for node in ["暴雨强度", "内涝深度", "地质易发性", "内涝风险", "地质灾害概率"]:
    r = result[node]
    top_idx = r["probabilities"].index(max(r["probabilities"]))
    print(f"  {node}: {r['states'][top_idx]} ({r['probabilities'][top_idx]*100:.1f}%)")

# 2. 敏感性分析
print("\n" + "=" * 70)
print("测试2: 20节点 - 敏感性分析（目标: 内涝风险）")
print("=" * 70)
df = engine.sensitivity_analysis("内涝风险")
if not df.empty:
    # 按参数聚合，显示最大影响
    summary = df.groupby("parameter").agg({"avg_|ΔP|": "max"}).sort_values("avg_|ΔP|", ascending=False)
    for param, row in summary.iterrows():
        print(f"  {param:10s}: |ΔP|_avg = {row['avg_|ΔP|']:.4f}")
    engine.plot_sensitivity(df, "内涝风险", "output/sensitivity_20nodes.png")
    df.to_csv("output/sensitivity_20nodes.csv", index=False, encoding="utf-8-sig")
    print("\n敏感性数据已导出: output/sensitivity_20nodes.csv")

# 3. 导出 JSON 结果
print("\n" + "=" * 70)
print("测试3: 导出完整推理结果到 JSON")
print("=" * 70)
full_result = engine.infer({"降水强度": "高", "坡度": "陡", "植被覆盖": "差"})
with open("output/result_full.json", "w", encoding="utf-8") as f:
    json.dump(full_result, f, ensure_ascii=False, indent=2)
print("已导出: output/result_full.json")

# 4. 验证 - 8节点 vs 20节点 部分结果对比
print("\n" + "=" * 70)
print("测试4: 8节点 vs 20节点 对比（相同证据场景）")
print("=" * 70)
engine_8 = DisasterChainEngine("configs/config_8nodes.yaml")
engine_20 = DisasterChainEngine("configs/config_20nodes.yaml")

scenarios = [
    ("正常", {"降水强度": "低", "风力": "弱"}),
    ("暴雨", {"降水强度": "高", "风力": "强", "风向": "向岸"}),
    ("最不利", {"降水强度": "高", "风力": "强", "海拔": "低", "历史排水时间": "慢"}),
]

for name, ev in scenarios:
    r8 = engine_8.infer(ev)
    r20 = engine_20.infer(ev)
    print(f"\n  [{name}] 证据: {json.dumps(ev, ensure_ascii=False)}")
    for out in ["内涝风险", "地质灾害概率"]:
        p8 = [p for s, p in zip(r8[out]["states"], r8[out]["probabilities"]) if s == "高"][0]
        p20 = [p for s, p in zip(r20[out]["states"], r20[out]["probabilities"]) if s == "高"][0]
        delta = p20 - p8
        print(f"    {out}(高): 8节点={p8:.3f}  20节点={p20:.3f}  Δ={delta:+.3f}")

print("\n" + "=" * 70)
print("全部测试完成")
print("=" * 70)