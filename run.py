"""
灾害链推理引擎 v2 - CLI 入口
================================
用法:
  python run.py --config configs/config_8nodes.yaml                                # 显示模型摘要
  python run.py --config configs/config_8nodes.yaml --infer '{"降水强度":"高"}'       # 部分证据推理
  python run.py --config configs/config_8nodes.yaml --sensitivity 内涝风险           # 敏感性分析
  python run.py --config configs/config_8nodes.yaml --infer '{}' --output result.json
"""

import argparse
import json
import sys
from pathlib import Path
from bn_engine import DisasterChainEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="灾害链贝叶斯网络推理引擎 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --config configs/config_8nodes.yaml
  %(prog)s --config configs/config_8nodes.yaml --infer '{"降水强度":"高","坡度":"陡"}'
  %(prog)s --config configs/config_8nodes.yaml --infer '{}' --output result.json
  %(prog)s --config configs/config_8nodes.yaml --sensitivity 内涝风险
  %(prog)s --config configs/config_8nodes.yaml --sensitivity 地质灾害概率 --output sensitivity.csv
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="配置文件路径 (YAML)",
    )
    parser.add_argument(
        "--infer", "-i",
        type=str,
        default=None,
        help='推理证据，JSON 格式字符串，如 \'{"降水强度":"高"}\'。空对象 {} 表示无证据先验推理',
    )
    parser.add_argument(
        "--sensitivity", "-s",
        type=str,
        default=None,
        metavar="TARGET_NODE",
        help="对目标节点执行全参数敏感性分析",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径 (JSON 或 CSV，根据扩展名自动判断)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="生成敏感性分析柱状图（需配合 --sensitivity）",
    )

    return parser.parse_args()


def print_inference_result(result: dict):
    """格式化打印推理结果"""
    meta = result.pop("_meta", {})
    print("\n" + "=" * 60)
    print(f"  模型: {meta.get('model', '未知')}")
    print(f"  证据覆盖度: {meta.get('evidence_coverage', 0)*100:.1f}% "
          f"({meta.get('evidence_provided', 0)}/{meta.get('evidence_expected', 0)})")
    missing = meta.get("missing_params", [])
    if missing:
        print(f"  未提供参数: {', '.join(missing)}")
    print("=" * 60)

    for node_name, node_result in result.items():
        if "error" in node_result:
            print(f"  ❌ {node_name}: {node_result['error']}")
            continue
        states = node_result["states"]
        probs = node_result["probabilities"]
        # 找出最大概率状态
        max_idx = probs.index(max(probs))
        max_state = states[max_idx]
        max_prob = probs[max_idx]

        # 格式化概率条
        bars = []
        for s, p in zip(states, probs):
            bar_len = int(p * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            marker = " ←" if s == max_state else ""
            bars.append(f"    {s:4s} |{bar}| {p*100:5.1f}%{marker}")

        print(f"\n  [{node_name}]")
        print("\n".join(bars))

    print()


def main():
    args = parse_args()

    # 初始化引擎
    try:
        engine = DisasterChainEngine(args.config)
        engine.print_summary()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)

    # ── 推理模式 ──
    if args.infer is not None:
        try:
            evidence = json.loads(args.infer)
        except json.JSONDecodeError as e:
            print(f"❌ 证据 JSON 解析失败: {e}")
            sys.exit(1)

        # 校验证据：检查节点名称和状态
        for key, val in evidence.items():
            try:
                node_cfg = engine.get_node_config(key)
                if val not in node_cfg["states"]:
                    print(f"⚠️  警告: '{key}' 的状态 '{val}' 不在定义中 "
                          f"({node_cfg['states']})，推理可能失败")
            except KeyError:
                print(f"⚠️  警告: 节点 '{key}' 未在配置中定义，将被忽略")
                del evidence[key]

        result = engine.infer(evidence)
        print_inference_result(result)

        if args.output:
            engine.export_results(result, args.output)

    # ── 敏感性分析模式 ──
    if args.sensitivity:
        target = args.sensitivity
        # 校验目标节点
        try:
            engine.get_node_config(target)
        except KeyError:
            print(f"❌ 目标节点 '{target}' 未在配置中定义")
            sys.exit(1)

        print(f"\n🔍 执行敏感性分析: 目标 = {target}")
        df = engine.sensitivity_analysis(target)

        if df.empty:
            print("(无有效敏感性数据)")
        else:
            # 打印摘要
            delta_cols = [c for c in df.columns if c.startswith("ΔP(")]
            print(f"\n{'='*70}")
            print(f"  参数敏感性分析 — 目标: {target}")
            print(f"{'='*70}")
            for _, row in df.iterrows():
                deltas = ", ".join(
                    f"{c}: {row[c]:+.4f}" for c in delta_cols
                )
                avg = row.get("avg_|ΔP|", 0)
                print(f"  {row['parameter']:10s} | {row['state']:4s} | {deltas} | |ΔP|_avg={avg:.4f}")

            # 输出到文件
            if args.output:
                df.to_csv(args.output, index=False, encoding="utf-8-sig")
                print(f"\n敏感性数据已导出: {args.output}")

            # 绘图
            if args.plot:
                plot_path = args.output.replace(".csv", ".png") if args.output else None
                if not plot_path:
                    plot_path = f"sensitivity_{target}.png"
                engine.plot_sensitivity(df, target, output_path=plot_path)


if __name__ == "__main__":
    main()