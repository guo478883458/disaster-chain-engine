"""
验证脚本：用 bn_engine 对每个样本推理，与实际标签对比，输出验证报告
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

warnings.filterwarnings("ignore")

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bn_engine import DisasterChainEngine

CONFIG = os.path.join(ROOT, "configs", "config_40nodes.yaml")
SAMPLES = os.path.join(ROOT, "output", "validation", "samples_zhengzhou.csv")
OUTPUT_DIR = os.path.join(ROOT, "output", "validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 40 个节点名称（与 prepare_validation_data.py 一致）
NODE_NAMES = [
    "降水强度", "降水时长", "风力", "风向", "湿度", "气压", "气温", "露点温度",
    "对流有效位能CAPE", "垂直风切变",
    "前期土壤含水量", "径流系数", "河道水位", "地下水埋深", "湖泊调蓄能力",
    "潮汐影响", "蒸发量",
    "海拔", "坡度", "坡向", "地形起伏度", "汇流面积", "河网密度",
    "地表粗糙度", "沟谷密度",
    "植被覆盖", "土壤渗透性", "岩性", "距断层距离", "地震烈度",
    "风化程度", "节理发育程度", "滑坡历史密度",
    "历史排水时间", "管网排水能力", "下垫面硬化率", "道路积水历史频率",
    "建筑密度", "绿地率", "应急排水能力",
]


def build_evidence(row):
    """从样本行构建证据字典（跳过保持先验的节点）"""
    evidence = {}
    for n in NODE_NAMES:
        v = row.get(n)
        if v not in (None, "保持先验", ""):
            evidence[n] = v
    return evidence


def calc_metrics(y_true, y_pred, y_prob=None):
    """计算准确率、精确率、召回率、F1、混淆矩阵"""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def plot_confusion_matrix(cm, title, ax, labels=("无内涝", "内涝")):
    """绘制混淆矩阵"""
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("实际")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > np.array(cm).max() / 2 else "black")
    return im


def main():
    print("=" * 60)
    print("验证：贝叶斯网络推理 vs 实际标签")
    print("=" * 60)

    # ── 1. 加载模型 ──
    print(f"\n[1] 加载模型: {CONFIG}")
    engine = DisasterChainEngine(CONFIG)
    engine.print_summary()

    # ── 2. 加载样本 ──
    print(f"\n[2] 加载样本: {SAMPLES}")
    df = pd.read_csv(SAMPLES, encoding="utf-8-sig")
    print(f"    样本数: {len(df)}")
    print(f"    正样本(Y=1): {(df['flood_label']==1).sum()}, "
          f"负样本(N=0): {(df['flood_label']==0).sum()}")

    # ── 3. 逐样本推理 ──
    print(f"\n[3] 逐样本推理 (共 {len(df)} 样本)...")
    predictions = []
    probabilities = []
    coverage_list = []
    details = []

    for idx, (_, row) in enumerate(df.iterrows()):
        evidence = build_evidence(row)
        try:
            result = engine.infer(evidence)
            risk = result["内涝风险"]
            probs = risk["probabilities"]  # [P(低), P(中), P(高)]
            states = risk["states"]  # [低, 中, 高]

            # 找到高状态索引
            high_idx = states.index("高")
            p_flood = probs[high_idx]

            # 预测：P(高) > 0.50 → 内涝
            # 模型先验 P(高)=0.50（加权 CPT 设计导致），故以 0.50 为决策阈值
            pred = 1 if p_flood > 0.50 else 0

            coverage = row.get("证据覆盖度", 0)
            proxy_conf = row.get("代理置信度", "高")

            predictions.append(pred)
            probabilities.append(p_flood)
            coverage_list.append(coverage)
            details.append({
                "idx": idx,
                "evidence_count": len(evidence),
                "evidence_coverage": coverage,
                "代理置信度": proxy_conf,
                "P(高)": p_flood,
                "pred": pred,
                "label": int(row["flood_label"]),
            })

            if (idx + 1) % 100 == 0:
                print(f"    已处理 {idx+1}/{len(df)} 样本...")

        except Exception as e:
            print(f"    样本 {idx} 推理失败: {e}")
            predictions.append(0)
            probabilities.append(0.0)
            coverage_list.append(row.get("证据覆盖度", 0))
            details.append({
                "idx": idx,
                "evidence_count": 0,
                "evidence_coverage": row.get("证据覆盖度", 0),
                "代理置信度": row.get("代理置信度", "高"),
                "P(高)": 0.0,
                "pred": 0,
                "label": int(row["flood_label"]),
                "error": str(e),
            })

    # 转换为 DataFrame
    det_df = pd.DataFrame(details)
    y_true = det_df["label"].values
    y_pred = det_df["pred"].values
    y_prob = det_df["P(高)"].values

    # ── 4. 总体指标 ──
    print(f"\n[4] 总体指标")
    overall = calc_metrics(y_true, y_pred)
    print(f"    准确率:  {overall['accuracy']*100:.2f}%")
    print(f"    F1 分数:  {overall['f1']:.4f}")
    print(f"    精确率:   {overall['precision']:.4f}")
    print(f"    召回率:   {overall['recall']:.4f}")
    print(f"    混淆矩阵:")
    print(f"             预测无内涝  预测内涝")
    print(f"    实际无内涝  {overall['confusion_matrix'][0][0]:>6d}  {overall['confusion_matrix'][0][1]:>6d}")
    print(f"    实际内涝    {overall['confusion_matrix'][1][0]:>6d}  {overall['confusion_matrix'][1][1]:>6d}")

    # ── 5. 按证据覆盖度分组统计 ──
    print(f"\n[5] 按证据覆盖度分组统计")

    # 定义分组
    # 全因素：覆盖度 >= 0.5（因为有 23/40 可映射，全因素约 57.5%）
    # 部分因素：覆盖度 < 0.5
    det_df["coverage_group"] = det_df["evidence_coverage"].apply(
        lambda x: "全因素" if x >= 0.5 else "部分因素"
    )

    group_results = {}
    for group_name, group_df in det_df.groupby("coverage_group"):
        if len(group_df) < 5:
            print(f"    {group_name}: 样本不足 ({len(group_df)})，跳过")
            continue
        metrics = calc_metrics(group_df["label"].values, group_df["pred"].values)
        group_results[group_name] = metrics
        print(f"    [{group_name}] 样本数={metrics['total']}, "
              f"准确率={metrics['accuracy']*100:.2f}%, "
              f"F1={metrics['f1']:.4f}")
        print(f"      混淆矩阵: TN={metrics['tn']} FP={metrics['fp']} "
              f"FN={metrics['fn']} TP={metrics['tp']}")

    # 细化分组：按覆盖度区间
    bins = [0, 0.3, 0.4, 0.5, 0.6, 1.0]
    labels = ["0-30%", "30-40%", "40-50%", "50-60%", "60-100%"]
    det_df["coverage_bin"] = pd.cut(det_df["evidence_coverage"], bins=bins,
                                    labels=labels, include_lowest=True)
    print(f"\n    [细化分组 - 按覆盖度区间]")
    for label, group_df in det_df.groupby("coverage_bin", observed=True):
        if len(group_df) < 3:
            continue
        metrics = calc_metrics(group_df["label"].values, group_df["pred"].values)
        print(f"      覆盖度 {label}: n={metrics['total']}, "
              f"准确率={metrics['accuracy']*100:.2f}%, "
              f"F1={metrics['f1']:.4f}")

    # ── 6. 按代理置信度分组统计 ──
    print(f"\n[6] 按代理置信度分组统计")
    for conf, group_df in det_df.groupby("代理置信度"):
        if len(group_df) < 5:
            continue
        metrics = calc_metrics(group_df["label"].values, group_df["pred"].values)
        print(f"    [{conf}] n={metrics['total']}, "
              f"准确率={metrics['accuracy']*100:.2f}%, "
              f"F1={metrics['f1']:.4f}")

    # ── 7. 输出验证报告 ──
    report_path = os.path.join(OUTPUT_DIR, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("灾害链推理引擎 v2 - 验证报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"模型: {engine.config['model']['name']}\n")
        f.write(f"配置文件: {CONFIG}\n")
        f.write(f"样本集: {SAMPLES}\n")
        f.write(f"样本数: {len(det_df)}\n")
        f.write(f"  - 正样本(Y=1): {int((y_true==1).sum())}\n")
        f.write(f"  - 负样本(N=0): {int((y_true==0).sum())}\n\n")

        f.write("-" * 70 + "\n")
        f.write("1. 总体指标\n")
        f.write("-" * 70 + "\n")
        f.write(f"  准确率 (Accuracy):  {overall['accuracy']*100:.2f}%\n")
        f.write(f"  F1 分数:            {overall['f1']:.4f}\n")
        f.write(f"  精确率 (Precision): {overall['precision']:.4f}\n")
        f.write(f"  召回率 (Recall):    {overall['recall']:.4f}\n\n")
        f.write(f"  混淆矩阵:\n")
        f.write(f"                   预测无内涝  预测内涝\n")
        f.write(f"  实际无内涝        {overall['confusion_matrix'][0][0]:>6d}  {overall['confusion_matrix'][0][1]:>6d}\n")
        f.write(f"  实际内涝          {overall['confusion_matrix'][1][0]:>6d}  {overall['confusion_matrix'][1][1]:>6d}\n\n")

        f.write("-" * 70 + "\n")
        f.write("2. 按证据覆盖度分组统计\n")
        f.write("-" * 70 + "\n")
        for group_name, metrics in group_results.items():
            f.write(f"  [{group_name}]\n")
            f.write(f"    样本数: {metrics['total']}\n")
            f.write(f"    准确率: {metrics['accuracy']*100:.2f}%\n")
            f.write(f"    F1:     {metrics['f1']:.4f}\n")
            f.write(f"    TN={metrics['tn']} FP={metrics['fp']} "
                    f"FN={metrics['fn']} TP={metrics['tp']}\n\n")

        f.write("-" * 70 + "\n")
        f.write("3. 按覆盖度区间细化统计\n")
        f.write("-" * 70 + "\n")
        for label, group_df in det_df.groupby("coverage_bin", observed=True):
            if len(group_df) < 3:
                continue
            metrics = calc_metrics(group_df["label"].values, group_df["pred"].values)
            f.write(f"  覆盖度 {label}: n={metrics['total']}, "
                    f"准确率={metrics['accuracy']*100:.2f}%, "
                    f"F1={metrics['f1']:.4f}\n")
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("4. 按代理置信度分组统计\n")
        f.write("-" * 70 + "\n")
        for conf, group_df in det_df.groupby("代理置信度"):
            if len(group_df) < 5:
                continue
            metrics = calc_metrics(group_df["label"].values, group_df["pred"].values)
            f.write(f"  [{conf}] n={metrics['total']}, "
                    f"准确率={metrics['accuracy']*100:.2f}%, "
                    f"F1={metrics['f1']:.4f}\n")
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("5. 证据覆盖度分布\n")
        f.write("-" * 70 + "\n")
        f.write(f"  平均覆盖度: {det_df['evidence_coverage'].mean()*100:.1f}%\n")
        f.write(f"  中位覆盖度: {det_df['evidence_coverage'].median()*100:.1f}%\n")
        f.write(f"  < 30%: {int((det_df['evidence_coverage']<0.3).sum())} 样本\n")
        f.write(f"  30-50%: {int(((det_df['evidence_coverage']>=0.3)&(det_df['evidence_coverage']<0.5)).sum())} 样本\n")
        f.write(f"  ≥ 50%: {int((det_df['evidence_coverage']>=0.5).sum())} 样本\n")
        f.write(f"  ≥ 60%: {int((det_df['evidence_coverage']>=0.6).sum())} 样本\n\n")

        f.write("=" * 70 + "\n")
        f.write("报告结束\n")
        f.write("=" * 70 + "\n")

    print(f"\n[7] 验证报告已输出: {report_path}")

    # ── 8. 绘制混淆矩阵图 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 总体混淆矩阵
    cm = overall["confusion_matrix"]
    plot_confusion_matrix(cm, "总体混淆矩阵\n(准确率={:.1%}, F1={:.3f})".format(
        overall["accuracy"], overall["f1"]), axes[0])

    # 按覆盖度分组混淆矩阵（取样本数最多的组）
    if group_results:
        main_group = list(group_results.keys())[0]
        cm_group = group_results[main_group]["confusion_matrix"]
        plot_confusion_matrix(
            cm_group,
            f"{main_group}混淆矩阵\n"
            f"(准确率={group_results[main_group]['accuracy']*100:.1f}%, "
            f"F1={group_results[main_group]['f1']:.3f})",
            axes[1])

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    混淆矩阵图已保存: {cm_path}")

    # ── 9. 输出详细结果 CSV ──
    detail_path = os.path.join(OUTPUT_DIR, "validation_details.csv")
    det_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"    详细结果已保存: {detail_path}")

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()