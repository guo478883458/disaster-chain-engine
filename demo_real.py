# -*- coding: utf-8 -*-
"""
郑州 7·20 真实数据演示脚本（暴雨网格案例）
================================================
从 653 个真实网格样本中筛选"降水强度=高"（暴雨）的网格，
选取 3 个代表性案例：
  A. 实际内涝 + 模型预测高（命中）
  B. 实际无内涝 + 模型预测低（命中）
  C. 实际无内涝 + 模型预测高（局限：假阳性，展示 83.9% 之外的 16%）

用法：
    D:\\ana\\envs\\disasterlex\\python.exe demo_real.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bn_engine import DisasterChainEngine

ENGINE = DisasterChainEngine("configs/config_40nodes.yaml")
SAMPLES = Path(__file__).parent / "output" / "validation" / "samples_zhengzhou.csv"
DETAILS = Path(__file__).parent / "output" / "validation" / "validation_details.csv"
PRIOR = "保持先验"


def fmt_dist(dist: dict) -> str:
    if "error" in dist:
        return f"（{dist['error'][:30]}…）"
    return "  ".join(f"{k}={v * 100:.1f}%" for k, v in
                     zip(dist.get("states", []), dist.get("probabilities", [])))


def infer_sample(row: pd.Series) -> tuple[dict, dict]:
    evidence = {}
    for node in ENGINE.get_input_params():
        val = row.get(node)
        if val not in (PRIOR, "", None) and pd.notna(val):
            evidence[node] = str(val)
    return ENGINE.infer(evidence), evidence


def pick_cases(df: pd.DataFrame, ph: pd.Series) -> list[tuple[str, pd.Series]]:
    """按演示策略选 3 个案例：正样本P高最高 / 负样本P高最低 / 假阳性P高最低"""
    storm = df[df["降水强度"] == "高"].copy()
    storm["P高"] = ph.values[storm.index]
    pos = storm[(storm["flood_label"] == 1)].sort_values("P高", ascending=False)
    neg = storm[(storm["flood_label"] == 0) & (storm["P高"] < 0.5)].sort_values("P高", ascending=True)
    fp = storm[(storm["flood_label"] == 0) & (storm["P高"] >= 0.5)].sort_values("P高", ascending=True)
    cases = []
    if len(pos): cases.append(("A. 实际内涝 → 模型预测高（命中）", pos.iloc[0]))
    if len(neg): cases.append(("B. 实际无内涝 → 模型预测低（命中）", neg.iloc[0]))
    if len(fp): cases.append(("C. 实际无内涝 → 模型预测高（假阳性，模型局限）", fp.iloc[0]))
    return cases


def main() -> None:
    df = pd.read_csv(SAMPLES, encoding="utf-8-sig")
    details = pd.read_csv(DETAILS, encoding="utf-8-sig")
    ph = details["P(高)"]
    storm_n = (df["降水强度"] == "高").sum()
    print(f"样本总数: {len(df)} | 其中降水强度=高（暴雨网格）: {storm_n}")

    print("=" * 72)
    print("  郑州 7·20 真实暴雨网格推理演示（预测概率 vs 真实结果）")
    print("=" * 72)
    for title, row in pick_cases(df, ph):
        result, evidence = infer_sample(row)
        p_high = result["内涝风险"]["probabilities"][2]
        print(f"\n【{title}】")
        print(f"  真实标签: {'内涝' if row['flood_label'] == 1 else '无内涝'}"
              f" | 证据覆盖度 {row['证据覆盖度'] * 100:.1f}%")
        key = {k: v for k, v in evidence.items() if k in
               ("降水强度", "坡度", "海拔", "历史排水时间", "下垫面硬化率")}
        print(f"  关键证据: " + "、".join(f"{k}={v}" for k, v in key.items()))
        print(f"  链条环节: 内涝深度 {fmt_dist(result['内涝深度'])}")
        print(f"  模型预测: 内涝风险[高]={p_high * 100:.1f}%"
              f"  地灾概率[高]={result['地质灾害概率']['probabilities'][1] * 100:.1f}%")
        correct = (p_high > 0.5) == (row["flood_label"] == 1)
        print(f"  判定（阈值 0.5）: {'✅ 与真实结果一致' if correct else '❌ 与真实结果相反（模型局限）'}")


if __name__ == "__main__":
    main()
