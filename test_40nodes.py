"""
40节点模型测试：弹性输入 + 全参数敏感性分析 + JSON 输出
"""
import json
import os
import sys
from bn_engine import DisasterChainEngine

CONFIG = "configs/config_40nodes.yaml"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 测试1：弹性输入 - 只给 1 个参数
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("测试1: 40节点 - 弹性输入（仅传入降水强度=高）")
print("=" * 70)
engine = DisasterChainEngine(CONFIG)
result = engine.infer({"降水强度": "高"})
meta = result.pop("_meta")

print(f"证据覆盖度: {meta['evidence_coverage']*100:.1f}% "
      f"({meta['evidence_provided']}/{meta['evidence_expected']})")
print(f"未提供参数 ({len(meta['missing_params'])} 个): "
      f"{', '.join(meta['missing_params'][:10])}...")
assert meta["evidence_provided"] == 1, f"应只提供1个证据，实际 {meta['evidence_provided']}"
assert len(meta["missing_params"]) == 39, f"应有39个缺失参数，实际 {len(meta['missing_params'])}"
print("  ✓ 弹性输入语义正确：未提供参数保持先验")

# 检查输出节点概率
for node in ["暴雨强度", "内涝深度", "地质易发性", "内涝风险", "地质灾害概率"]:
    if node in result:
        r = result[node]
        if "error" not in r:
            top_idx = r["probabilities"].index(max(r["probabilities"]))
            print(f"  {node}: {r['states'][top_idx]} ({r['probabilities'][top_idx]*100:.1f}%)")
        else:
            print(f"  {node}: 错误 - {r['error']}")
            assert False, f"{node} 推理失败: {r['error']}"

print("  ✓ 弹性输入测试通过\n")

# ═══════════════════════════════════════════════════════════════════════════
# 测试2：全参数敏感性分析（目标: 内涝风险）
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("测试2: 40节点 - 全参数敏感性分析（目标: 内涝风险）")
print("=" * 70)
df = engine.sensitivity_analysis("内涝风险")
assert not df.empty, "敏感性分析结果不应为空"
print(f"  敏感性分析遍历参数数: {df['parameter'].nunique()}")
print(f"  总数据行数: {len(df)}")

# 按参数聚合，显示 top 10 最大影响参数
summary = df.groupby("parameter").agg({"avg_|ΔP|": "max"}).sort_values("avg_|ΔP|", ascending=False)
print(f"\n  Top 10 敏感参数:")
for i, (param, row) in enumerate(summary.head(10).iterrows(), 1):
    print(f"    {i:2d}. {param:12s}: |ΔP|_avg = {row['avg_|ΔP|']:.4f}")

# 导出敏感性图和数据
engine.plot_sensitivity(df, "内涝风险", os.path.join(OUTPUT_DIR, "sensitivity_40nodes.png"))
df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_40nodes.csv"), index=False, encoding="utf-8-sig")
print(f"\n  敏感性数据已导出: {OUTPUT_DIR}/sensitivity_40nodes.csv")
print("  ✓ 全参数敏感性分析测试通过\n")

# ═══════════════════════════════════════════════════════════════════════════
# 测试3：输出 JSON 结果
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("测试3: 导出完整推理结果到 JSON")
print("=" * 70)
full_result = engine.infer({
    "降水强度": "高", "降水时长": "长", "风力": "强",
    "前期土壤含水量": "高", "海拔": "低", "坡度": "陡",
    "植被覆盖": "差", "管网排水能力": "弱",
})
json_path = os.path.join(OUTPUT_DIR, "result_40nodes.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(full_result, f, ensure_ascii=False, indent=2)

# 校验 JSON 结构
assert "_meta" in full_result, "JSON 应包含 _meta 字段"
assert "内涝风险" in full_result, "JSON 应包含 内涝风险 节点"
assert "地质灾害概率" in full_result, "JSON 应包含 地质灾害概率 节点"
print(f"  已导出: {json_path}")

# 打印输出结果
meta = full_result["_meta"]
print(f"  证据覆盖度: {meta['evidence_coverage']*100:.1f}% "
      f"({meta['evidence_provided']}/{meta['evidence_expected']})")
for out_name in ["内涝风险", "地质灾害概率"]:
    r = full_result[out_name]
    for s, p in zip(r["states"], r["probabilities"]):
        print(f"  P({out_name}={s}) = {p:.4f}")
print("  ✓ JSON 输出测试通过\n")

# ═══════════════════════════════════════════════════════════════════════════
# 测试4：模型摘要验证
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("测试4: 40节点模型摘要验证")
print("=" * 70)
info = engine.summary()
assert info["total_nodes"] == 49, f"应有49个节点，实际 {info['total_nodes']}"
assert len(info["input_params"]) == 40, f"应有40个输入参数，实际 {len(info['input_params'])}"
assert len(info["outputs"]) == 2, f"应有2个输出，实际 {len(info['outputs'])}"
print(f"  总节点数: {info['total_nodes']} ✓")
print(f"  输入参数: {len(info['input_params'])} ✓")
print(f"  输出节点: {len(info['outputs'])} ✓")
for cat, nodes in info["nodes_by_category"].items():
    print(f"  [{cat}] {len(nodes)} 个节点")
print("  ✓ 模型摘要验证通过\n")

# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("全部 40 节点测试完成 ✓")
print("=" * 70)