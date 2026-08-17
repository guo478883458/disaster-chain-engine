"""
灾害链推理引擎 v2 - 可视化工具 CLI
===================================
用法:
  python tools/visualize.py --config configs/config_40nodes.yaml --network
  python tools/visualize.py --config configs/config_40nodes.yaml --infer '{"降水强度":"高","坡度":"陡"}'
  python tools/visualize.py --config configs/config_40nodes.yaml --sensitivity 内涝风险
  python tools/visualize.py --config configs/config_40nodes.yaml --compare
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# ── 确保能找到 bn_engine ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bn_engine import DisasterChainEngine

# matplotlib 配置（必须在 import pyplot 之前设置字体）
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "visualize"

# ============================================================================
# 配色方案
# ============================================================================
CATEGORY_COLORS = {
    "气象": "#FF8C00",       # 橙
    "水文": "#FF8C00",       # 橙（输入层统一）
    "地形": "#FF8C00",
    "地质": "#FF8C00",
    "城市": "#FF8C00",
    "子节点": "#3498DB",     # 蓝（中间层）
    "中间环节": "#3498DB",
    "输出": "#2ECC71",       # 绿（输出层）
}

# 输入层类别（统一橙色）
INPUT_CATEGORIES = ["气象", "水文", "地形", "地质", "城市"]
MIDDLE_CATEGORIES = ["子节点", "中间环节"]
OUTPUT_CATEGORIES = ["输出"]

# 每个输入类别在 network 图中的列偏移
CATEGORY_COL_OFFSET = {
    "气象": 0, "水文": 1, "地形": 2, "地质": 3, "城市": 4,
    "子节点": 0, "中间环节": 0, "输出": 0,
}

# ============================================================================
# 场景定义（用于 --compare）
# ============================================================================
SCENARIOS = [
    {
        "name": "正常天气",
        "evidence": {},
        "desc": "无任何输入证据，先验推理",
    },
    {
        "name": "暴雨",
        "evidence": {"降水强度": "高", "风力": "强"},
        "desc": "高降水强度 + 强风",
    },
    {
        "name": "最不利内涝",
        "evidence": {
            "降水强度": "高", "降水时长": "长", "风力": "强",
            "海拔": "低", "城市排水能力": "弱",
        },
        "desc": "高降水 + 长时长 + 低洼 + 排水弱",
    },
    {
        "name": "地灾场景",
        "evidence": {
            "降水强度": "高", "坡度": "陡",
            "植被覆盖": "差", "土壤渗透性": "差",
        },
        "desc": "高降水 + 陡坡 + 植被差 + 渗透差",
    },
]


# ============================================================================
# 辅助函数
# ============================================================================

def _ensure_output_dir():
    """确保输出目录存在"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_category_color(category):
    """获取类别颜色"""
    if category in INPUT_CATEGORIES:
        return CATEGORY_COLORS["气象"]  # 输入层统一橙色
    return CATEGORY_COLORS.get(category, "#95A5A6")


def _get_node_layer(category):
    """获取节点所在网络层（用于 DAG 布局）"""
    if category in INPUT_CATEGORIES:
        return 0  # 输入层
    if category == "子节点":
        return 1  # 聚合层
    if category == "中间环节":
        return 2  # 中间层
    if category == "输出":
        return 3  # 输出层
    return 0


# ============================================================================
# 1. 网络结构图
# ============================================================================

def plot_network(engine, output_path):
    """绘制 DAG 网络结构图，按类别分层布局"""
    try:
        import networkx as nx
    except ImportError:
        print("❌ 需要 networkx 库: pip install networkx")
        sys.exit(1)

    nodes = engine.config["nodes"]
    edges = engine.config["edges"]

    # 构建有向图
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["name"], category=n.get("category", "未分类"), states=n["states"])
    for src, dst in edges:
        G.add_edge(src, dst)

    # ── 计算布局 ──
    # 按层 + 类别分配位置
    pos = {}
    layer_nodes = {0: {}, 1: {}, 2: {}, 3: {}}  # layer -> {category: [node_list]}

    for n in nodes:
        cat = n.get("category", "未分类")
        layer = _get_node_layer(cat)
        if cat not in layer_nodes[layer]:
            layer_nodes[layer][cat] = []
        layer_nodes[layer][cat].append(n["name"])

    # 水平间距和垂直间距
    h_spacing = 2.5
    v_spacing = 3.0

    for layer, cat_dict in layer_nodes.items():
        # 计算该层所有节点总数，用于居中
        all_nodes_in_layer = []
        cat_starts = {}
        cat_count = len(cat_dict)
        cat_idx = 0
        for cat, node_list in cat_dict.items():
            cat_starts[cat] = cat_idx
            all_nodes_in_layer.extend(node_list)
            cat_idx += len(node_list)

        total = len(all_nodes_in_layer)
        start_x = - (total - 1) * h_spacing / 2

        for cat, node_list in cat_dict.items():
            for i, name in enumerate(node_list):
                x = start_x + (cat_starts[cat] + i) * h_spacing
                # 输入层（layer=0）按类别分组，中间留空隙
                if layer == 0:
                    # 在类别之间增加空隙
                    gap = cat_starts[cat] * 0.5
                    x = start_x + (cat_starts[cat] + i) * h_spacing + gap
                y = -layer * v_spacing
                pos[name] = (x, y)

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_aspect("equal")

    # 绘制边
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#AAAAAA",
        arrows=True,
        arrowsize=15,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.05",
        min_source_margin=20,
        min_target_margin=20,
    )

    # 绘制节点
    for n in nodes:
        name = n["name"]
        cat = n.get("category", "未分类")
        color = _get_category_color(cat)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=[name],
            node_color=color,
            node_size=1800,
            edgecolors="white",
            linewidths=1.5,
        )

    # 绘制标签（节点名称 + 状态）
    for n in nodes:
        name = n["name"]
        x, y = pos[name]
        states_str = " [" + "/".join(n["states"]) + "]"
        ax.text(
            x, y - 0.35, states_str,
            ha="center", va="top", fontsize=6, color="#555555",
        )

    # 节点名称
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7,
        font_weight="bold",
        font_family="Microsoft YaHei",
    )

    # ── 图例 ──
    legend_elements = [
        mpatches.Patch(color="#FF8C00", label="输入层 (气象/水文/地形/地质/城市)"),
        mpatches.Patch(color="#3498DB", label="中间层 (子节点/中间环节)"),
        mpatches.Patch(color="#2ECC71", label="输出层 (内涝风险/地质灾害概率)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    ax.set_title(
        f"灾害链贝叶斯网络结构图\n{engine.config['model']['name']}",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 网络结构图已保存: {output_path}")


# ============================================================================
# 2. 推理结果可视化
# ============================================================================

def plot_infer_result(engine, evidence, output_path):
    """绘制推理结果双面板图"""
    # 执行推理
    result = engine.infer(evidence)
    meta = result.pop("_meta", {})

    # 获取所有输入节点和输出节点
    input_params = engine.get_input_params()
    output_nodes = engine.get_outputs()

    # ── 创建图形 ──
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(
        f"推理结果可视化\n证据覆盖度: {meta.get('evidence_coverage', 0)*100:.1f}% "
        f"({meta.get('evidence_provided', 0)}/{meta.get('evidence_expected', 0)} 参数)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ═══════════════════════════════════════════
    # 左面板：证据输入状态
    # ═══════════════════════════════════════════
    ax_left.set_title("输入参数状态", fontsize=12, fontweight="bold", pad=10)

    # 按类别分组
    cat_params = {}
    for p in input_params:
        try:
            cfg = engine.get_node_config(p)
            cat = cfg.get("category", "未分类")
            if cat not in cat_params:
                cat_params[cat] = []
            cat_params[cat].append((p, p in evidence))
        except KeyError:
            pass

    y_pos = 0
    cat_colors_input = {
        "气象": "#FF8C00", "水文": "#E67E22", "地形": "#D35400",
        "地质": "#8E44AD", "城市": "#C0392B",
    }

    for cat, params in cat_params.items():
        # 类别标题
        ax_left.text(
            -0.5, y_pos, f"─ {cat} ─",
            fontsize=9, fontweight="bold", color=cat_colors_input.get(cat, "#333"),
            va="center",
        )
        y_pos += 1

        for p_name, is_provided in params:
            if is_provided:
                color = "#27AE60"  # 绿色：已提供证据
                marker = f"● {evidence[p_name]}"
            else:
                color = "#95A5A6"  # 灰色：先验
                marker = "○ 先验"
            ax_left.text(
                -0.5, y_pos, f"  {p_name:12s}  {marker}",
                fontsize=8, color=color, va="center",
            )
            y_pos += 1

        y_pos += 0.5  # 类别间间距

    ax_left.set_xlim(-1, 1)
    ax_left.set_ylim(-1, y_pos + 1)
    ax_left.invert_yaxis()
    ax_left.axis("off")

    # ═══════════════════════════════════════════
    # 右面板：输出概率条形图
    # ═══════════════════════════════════════════
    ax_right.set_title("输出节点概率分布", fontsize=12, fontweight="bold", pad=10)

    output_probs = {}
    for on in output_nodes:
        if on in result and "probabilities" in result[on]:
            output_probs[on] = {
                "states": result[on]["states"],
                "probs": result[on]["probabilities"],
            }

    if not output_probs:
        ax_right.text(0.5, 0.5, "无输出节点数据", ha="center", va="center", fontsize=12)
    else:
        n_outputs = len(output_probs)
        colors_bar = ["#3498DB", "#F39C12", "#E74C3C"]

        for idx, (oname, data) in enumerate(output_probs.items()):
            states = data["states"]
            probs = data["probs"]

            # 计算条形图位置
            n_states = len(states)
            bar_width = 0.6
            x = np.arange(n_states) + idx * (n_states + 1)

            bars = ax_right.bar(
                x, probs, bar_width,
                color=colors_bar[:n_states],
                edgecolor="white", linewidth=1.2,
                alpha=0.85,
            )

            # 标注最大概率状态
            max_idx = np.argmax(probs)
            ax_right.text(
                x[max_idx], probs[max_idx] + 0.02,
                f"{states[max_idx]}\n{probs[max_idx]*100:.1f}%",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#333",
            )

            # 概率值标签
            for i, (bar, p) in enumerate(zip(bars, probs)):
                ax_right.text(
                    bar.get_x() + bar.get_width() / 2, p + 0.01,
                    f"{p*100:.1f}%",
                    ha="center", va="bottom", fontsize=7,
                    color="#555",
                )

            # 节点名称标签
            for i, s in enumerate(states):
                ax_right.text(
                    x[i], -0.05, s,
                    ha="center", va="top", fontsize=8,
                    color="#555",
                )

            # 节点标题
            mid_x = x[len(x) // 2] if len(x) > 1 else x[0]
            ax_right.text(
                mid_x, 1.05, oname,
                ha="center", va="bottom", fontsize=10,
                fontweight="bold",
            )

        ax_right.set_xlim(-1, x[-1] + 1)
        ax_right.set_ylim(0, 1.15)
        ax_right.set_ylabel("概率", fontsize=10)
        ax_right.set_xticks([])
        ax_right.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 推理结果图已保存: {output_path}")
    result["_meta"] = meta  # 恢复


# ============================================================================
# 3. 敏感性分析 Top10
# ============================================================================

def plot_sensitivity_top10(engine, target_node, output_path):
    """绘制敏感性分析 Top10 横向条形图"""
    print(f"🔍 执行敏感性分析: 目标 = {target_node}")
    df = engine.sensitivity_analysis(target_node)

    if df.empty:
        print("⚠️ 敏感性分析数据为空，无法绘图")
        return

    # 计算每个参数的最大 |ΔP|
    delta_cols = [c for c in df.columns if c.startswith("ΔP(")]
    if not delta_cols:
        print("⚠️ 无 ΔP 列，无法绘图")
        return

    # 按参数分组，取最大 |ΔP|
    param_impact = []
    for param in df["parameter"].unique():
        param_df = df[df["parameter"] == param]
        max_abs_delta = param_df[delta_cols].abs().max().max()
        # 找出对应的最佳状态
        best_row = param_df.loc[param_df[delta_cols].abs().max(axis=1).idxmax()]
        param_impact.append({
            "parameter": param,
            "max_|ΔP|": round(max_abs_delta, 4),
            "best_state": best_row["state"],
            "best_delta": round(best_row[delta_cols].max(), 4),
        })

    param_impact.sort(key=lambda x: x["max_|ΔP|"], reverse=True)

    # Top 10
    top10 = param_impact[:10]

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [f"{p['parameter']} ({p['best_state']})" for p in top10][::-1]
    values = [p["max_|ΔP|"] for p in top10][::-1]

    # 颜色渐变
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(values)))

    bars = ax.barh(names, values, color=colors, edgecolor="gray", linewidth=0.8)

    # 数值标签
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            ha="left", va="center", fontsize=9,
        )

    ax.set_xlabel("最大 |ΔP|", fontsize=11)
    ax.set_title(f"参数敏感性分析 Top 10 — 目标: {target_node}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(values) * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 敏感性 Top10 图已保存: {output_path}")

    # 打印文本表格
    print(f"\n{'='*60}")
    print(f"  参数敏感性 Top 10 — 目标: {target_node}")
    print(f"{'='*60}")
    print(f"  {'参数':12s} {'状态':6s} {'|ΔP|':>8s}")
    print(f"  {'-'*28}")
    for p in param_impact[:10]:
        print(f"  {p['parameter']:12s} {p['best_state']:6s} {p['max_|ΔP|']:8.4f}")
    print(f"{'='*60}\n")


# ============================================================================
# 4. 场景对比图
# ============================================================================

def plot_scene_compare(engine, output_path):
    """绘制 4 个典型场景的对比分组条形图"""
    output_nodes = engine.get_outputs()
    if not output_nodes:
        print("⚠️ 配置中未定义输出节点，无法进行场景对比")
        return

    # 执行每个场景的推理
    scene_results = []
    for scene in SCENARIOS:
        result = engine.infer(scene["evidence"])
        meta = result.pop("_meta", {})

        scene_data = {"name": scene["name"], "desc": scene["desc"], "outputs": {}}
        for on in output_nodes:
            if on in result and "probabilities" in result[on]:
                states = result[on]["states"]
                probs = result[on]["probabilities"]
                # 找到"高"状态的概率（或最后一个状态）
                high_state = "高" if "高" in states else states[-1]
                high_idx = states.index(high_state)
                scene_data["outputs"][on] = {
                    "high_prob": probs[high_idx],
                    "high_state": high_state,
                }
        scene_results.append(scene_data)
        result["_meta"] = meta

    # ── 绘图 ──
    n_scenes = len(scene_results)
    n_outputs = len(output_nodes)
    bar_width = 0.3
    x = np.arange(n_scenes)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors_out = ["#E74C3C", "#8E44AD", "#3498DB", "#2ECC71"]

    for i, oname in enumerate(output_nodes):
        probs = [sr["outputs"][oname]["high_prob"] for sr in scene_results]
        bars = ax.bar(
            x + i * bar_width - (n_outputs - 1) * bar_width / 2,
            probs, bar_width,
            label=oname,
            color=colors_out[i % len(colors_out)],
            edgecolor="white", linewidth=1.2,
            alpha=0.85,
        )
        # 数值标签
        for bar, p in zip(bars, probs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{p*100:.1f}%",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([s["name"] for s in scene_results], fontsize=11)
    ax.set_ylabel("P(高)", fontsize=12)
    ax.set_title("典型场景下输出节点概率对比", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # 在底部加场景描述
    desc_text = "  |  ".join(
        f"{s['name']}: {s['desc']}" for s in scene_results
    )
    ax.text(
        0.5, -0.15, desc_text,
        ha="center", va="top", fontsize=8, color="#555",
        transform=ax.transAxes,
        wrap=True,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 场景对比图已保存: {output_path}")

    # 打印文本表格
    print(f"\n{'='*70}")
    print(f"  典型场景对比")
    print(f"{'='*70}")
    header = f"  {'场景':12s} {'描述':20s}"
    for on in output_nodes:
        header += f" {on:12s}"
    print(header)
    print(f"  {'-'*60}")
    for sr in scene_results:
        row = f"  {sr['name']:12s} {sr['desc']:20s}"
        for on in output_nodes:
            p = sr["outputs"][on]["high_prob"]
            row += f" {p*100:8.1f}%   "
        print(row)
    print(f"{'='*70}\n")


# ============================================================================
# CLI 入口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="灾害链推理引擎 v2 - 可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子功能:
  --network                 绘制网络结构图 (DAG)
  --infer '{"key":"val"}'   推理结果双面板可视化
  --sensitivity TARGET      敏感性分析 Top10 图
  --compare                 典型场景对比图

示例:
  %(prog)s --config configs/config_40nodes.yaml --network
  %(prog)s --config configs/config_40nodes.yaml --infer '{"降水强度":"高","坡度":"陡"}'
  %(prog)s --config configs/config_40nodes.yaml --sensitivity 内涝风险
  %(prog)s --config configs/config_40nodes.yaml --compare
        """,
    )
    parser.add_argument("--config", "-c", type=str, required=True, help="配置文件路径")
    parser.add_argument("--network", action="store_true", help="绘制网络结构图")
    parser.add_argument("--infer", type=str, default=None, help='推理证据，JSON 格式')
    parser.add_argument("--sensitivity", type=str, default=None, metavar="TARGET", help="敏感性分析目标节点")
    parser.add_argument("--compare", action="store_true", help="场景对比图")
    return parser.parse_args()


def main():
    args = parse_args()
    _ensure_output_dir()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    # 初始化引擎
    try:
        engine = DisasterChainEngine(str(config_path))
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)

    # ── 网络结构图 ──
    if args.network:
        output_path = OUTPUT_DIR / "network.png"
        plot_network(engine, str(output_path))

    # ── 推理结果可视化 ──
    if args.infer is not None:
        try:
            evidence = json.loads(args.infer)
        except json.JSONDecodeError as e:
            print(f"❌ 证据 JSON 解析失败: {e}")
            sys.exit(1)

        # 校验证据
        for key, val in list(evidence.items()):
            try:
                node_cfg = engine.get_node_config(key)
                if val not in node_cfg["states"]:
                    print(f"⚠️  警告: '{key}' 状态 '{val}' 不在定义中 ({node_cfg['states']})")
            except KeyError:
                print(f"⚠️  警告: 节点 '{key}' 未定义，已忽略")
                del evidence[key]

        output_path = OUTPUT_DIR / "infer_result.png"
        plot_infer_result(engine, evidence, str(output_path))

    # ── 敏感性分析 ──
    if args.sensitivity:
        # 校验目标节点
        try:
            engine.get_node_config(args.sensitivity)
        except KeyError:
            print(f"❌ 目标节点 '{args.sensitivity}' 未在配置中定义")
            sys.exit(1)

        output_path = OUTPUT_DIR / "sensitivity_top10.png"
        plot_sensitivity_top10(engine, args.sensitivity, str(output_path))

    # ── 场景对比 ──
    if args.compare:
        output_path = OUTPUT_DIR / "scene_compare.png"
        plot_scene_compare(engine, str(output_path))

    if not any([args.network, args.infer, args.sensitivity, args.compare]):
        print("⚠️  未指定任何可视化子功能，请使用 --network / --infer / --sensitivity / --compare")
        parser = parse_args()
        parser.print_help()


if __name__ == "__main__":
    main()