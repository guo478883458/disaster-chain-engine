"""
灾害链推理引擎 v2 - Streamlit 交互式演示面板（重构版）
=============================================
用法:
  streamlit run dashboard.py

功能：
  - 两阶段分页：参数选择页 → 结果页
  - Plotly 矢量交互图表（网络图/推理概率/敏感性/场景对比）
  - 弹性输入：未指定参数保持先验，不报错
  - "开始预测"按钮，避免实时联动
"""

import sys
import traceback as tb
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ── 确保能找到 bn_engine ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bn_engine import DisasterChainEngine

# ── 配置文件路径 ──
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config_40nodes.yaml"


# ============================================================================
# 缓存引擎（避免重复加载）
# ============================================================================

@st.cache_resource
def load_engine(config_path):
    return DisasterChainEngine(str(config_path))


# ============================================================================
# 初始化 session_state
# ============================================================================

def init_session_state():
    """初始化所有 session_state 变量（不含 param_* —— 它们在 render_param_page 中初始化）"""
    if "page" not in st.session_state:
        st.session_state["page"] = "params"  # "params" | "results"
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "meta" not in st.session_state:
        st.session_state["meta"] = None
    if "evidence_used" not in st.session_state:
        st.session_state["evidence_used"] = {}
    if "params_modified" not in st.session_state:
        st.session_state["params_modified"] = False
    if "error_msg" not in st.session_state:
        st.session_state["error_msg"] = None


# ============================================================================
# 页面跳转辅助
# ============================================================================

def go_to_page(page_name):
    st.session_state["page"] = page_name
    st.rerun()


# ============================================================================
# PLOTLY 图表函数
# ============================================================================

def plotly_network(engine):
    """绘制 Plotly 交互式网络结构图"""
    try:
        import networkx as nx
    except ImportError:
        fig = go.Figure()
        fig.add_annotation(text="需要 networkx 库", showarrow=False)
        return fig

    nodes = engine.config["nodes"]
    edges = engine.config["edges"]

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["name"], category=n.get("category", "未分类"), states=n["states"])
    for src, dst in edges:
        G.add_edge(src, dst)

    # ── 分层布局 ──
    def _get_node_layer(cat):
        input_cats = ["气象", "水文", "地形", "地质", "城市"]
        if cat in input_cats:
            return 0
        if cat == "子节点":
            return 1
        if cat == "中间环节":
            return 2
        if cat == "输出":
            return 3
        return 0

    pos = {}
    layer_nodes = {0: {}, 1: {}, 2: {}, 3: {}}
    for n in nodes:
        cat = n.get("category", "未分类")
        layer = _get_node_layer(cat)
        if cat not in layer_nodes[layer]:
            layer_nodes[layer][cat] = []
        layer_nodes[layer][cat].append(n["name"])

    h_spacing = 2.5
    v_spacing = 3.0
    for layer, cat_dict in layer_nodes.items():
        all_nodes = []
        cat_starts = {}
        ci = 0
        for cat, nl in cat_dict.items():
            cat_starts[cat] = ci
            all_nodes.extend(nl)
            ci += len(nl)
        total = len(all_nodes)
        start_x = -(total - 1) * h_spacing / 2
        for cat, nl in cat_dict.items():
            for i, name in enumerate(nl):
                x = start_x + (cat_starts[cat] + i) * h_spacing
                if layer == 0:
                    x = start_x + (cat_starts[cat] + i) * h_spacing + cat_starts[cat] * 0.5
                y = -layer * v_spacing
                pos[name] = (x, y)

    # ── 颜色映射 ──
    color_map = {
        "气象": "#FF8C00", "水文": "#FF8C00", "地形": "#FF8C00",
        "地质": "#FF8C00", "城市": "#FF8C00",
        "子节点": "#3498DB", "中间环节": "#3498DB",
        "输出": "#2ECC71",
    }

    # ── 构建 Plotly 图 ──
    edge_traces = []
    for src, dst in edges:
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_traces.append(
            go.Scatter(
                x=[x0, (x0 + x1) / 2, x1],
                y=[y0, (y0 + y1) / 2, y1],
                mode="lines",
                line=dict(color="#AAAAAA", width=1.5),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # 节点
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_colors = []
    node_sizes = []

    for n in nodes:
        name = n["name"]
        if name not in pos:
            continue
        x, y = pos[name]
        node_x.append(x)
        node_y.append(y)
        cat = n.get("category", "未分类")
        states_str = "[" + "/".join(n["states"]) + "]"
        node_text.append(name)
        node_hover.append(
            f"<b>{name}</b><br>"
            f"类别: {cat}<br>"
            f"状态: {states_str}<br>"
        )
        node_colors.append(color_map.get(cat, "#95A5A6"))
        node_sizes.append(22)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color="white", width=1.5),
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=9, color="white", family="Microsoft YaHei"),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure()
    for t in edge_traces:
        fig.add_trace(t)
    fig.add_trace(node_trace)

    fig.update_layout(
        title=dict(
            text=f"灾害链贝叶斯网络结构图<br><sub>{engine.config['model']['name']}</sub>",
            font=dict(size=16, family="Microsoft YaHei"),
        ),
        font=dict(family="Microsoft YaHei"),
        hovermode="closest",
        showlegend=False,
        width=1000,
        height=600,
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plotly_output_probs(result, output_nodes):
    """Plotly 分组条形图 — 输出节点概率分布"""
    fig = go.Figure()

    colors = ["#3498DB", "#F39C12", "#E74C3C"]
    for idx, oname in enumerate(output_nodes):
        if oname not in result or "probabilities" not in result[oname]:
            continue
        states = result[oname]["states"]
        probs = result[oname]["probabilities"]
        max_idx = int(np.argmax(probs))

        text_labels = [f"{p*100:.1f}%" for p in probs]
        if max_idx >= 0:
            text_labels[max_idx] = f"<b>{text_labels[max_idx]}</b>"

        fig.add_trace(go.Bar(
            name=oname,
            x=states,
            y=probs,
            text=text_labels,
            textposition="outside",
            textfont=dict(size=12, family="Microsoft YaHei"),
            marker_color=colors[:len(states)],
            marker_line=dict(color="white", width=1),
            hovertemplate=f"<b>{oname}</b><br>状态: %{{x}}<br>概率: %{{y:.1%}}<extra></extra>",
            offsetgroup=idx,
        ))

    fig.update_layout(
        font=dict(family="Microsoft YaHei"),
        yaxis=dict(title="概率", range=[0, 1.2], dtick=0.2),
        barmode="group",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#EEEEEE")
    return fig


def plotly_sensitivity_top10(engine, target_node):
    """Plotly 横向条形图 — 敏感性 Top10"""
    df = engine.sensitivity_analysis(target_node)
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="无敏感性数据", showarrow=False, font=dict(size=14))
        return fig

    delta_cols = [c for c in df.columns if c.startswith("ΔP(")]
    if not delta_cols:
        fig = go.Figure()
        return fig

    # 按参数取最大 |ΔP|
    param_impact = []
    for param in df["parameter"].unique():
        sub = df[df["parameter"] == param]
        max_abs = sub[delta_cols].abs().max().max()
        best_row = sub.loc[sub[delta_cols].abs().max(axis=1).idxmax()]
        param_impact.append({
            "parameter": param,
            "max_|ΔP|": round(max_abs, 4),
            "best_state": best_row["state"],
            "best_delta": round(best_row[delta_cols].max(), 4),
        })

    param_impact.sort(key=lambda x: x["max_|ΔP|"], reverse=True)
    top10 = param_impact[:10]

    names = [f"{p['parameter']} ({p['best_state']})" for p in top10][::-1]
    values = [p["max_|ΔP|"] for p in top10][::-1]

    colors = px.colors.sample_colorscale("oranges", [i / max(len(values), 1) for i in range(len(values))])

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        marker_line=dict(color="gray", width=0.5),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, family="Microsoft YaHei"),
        hovertemplate="参数: %{y}<br>最大 |ΔP|: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"参数敏感性分析 Top 10 — 目标: {target_node}",
            font=dict(size=14, family="Microsoft YaHei"),
        ),
        font=dict(family="Microsoft YaHei"),
        xaxis=dict(title="最大 |ΔP|", gridcolor="#EEEEEE"),
        yaxis=dict(title=""),
        margin=dict(l=20, r=80, t=50, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
    )
    return fig


def plotly_scene_compare(engine, output_nodes):
    """Plotly 分组条形图 — 场景对比"""
    scenes = [
        {"name": "正常天气", "evidence": {}, "desc": "先验推理"},
        {"name": "暴雨", "evidence": {"降水强度": "高", "风力": "强"}, "desc": "高降水+强风"},
        {"name": "最不利内涝", "evidence": {"降水强度": "高", "降水时长": "长", "风力": "强", "海拔": "低", "城市排水能力": "弱"}, "desc": "高降水+长时长+低洼+排水弱"},
        {"name": "地灾场景", "evidence": {"降水强度": "高", "坡度": "陡", "植被覆盖": "差", "土壤渗透性": "差"}, "desc": "高降水+陡坡+植被差+渗透差"},
    ]

    scene_results = []
    for scene in scenes:
        r = engine.infer(scene["evidence"])
        scene_results.append(r)

    colors_out = {"内涝风险": "#E74C3C", "地质灾害概率": "#8E44AD"}

    fig = go.Figure()
    for oname in output_nodes:
        probs = []
        for sr in scene_results:
            if oname in sr and "probabilities" in sr[oname]:
                states = sr[oname]["states"]
                high_state = "高" if "高" in states else states[-1]
                high_idx = states.index(high_state)
                probs.append(sr[oname]["probabilities"][high_idx])
            else:
                probs.append(0.0)

        fig.add_trace(go.Bar(
            name=oname,
            x=[s["name"] for s in scenes],
            y=probs,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(size=12, family="Microsoft YaHei"),
            marker_color=colors_out.get(oname, "#3498DB"),
            marker_line=dict(color="white", width=1),
            hovertemplate=f"<b>{oname}</b><br>场景: %{{x}}<br>P(高): %{{y:.1%}}<extra></extra>",
            offsetgroup=oname,
        ))

    fig.update_layout(
        title=dict(
            text="典型场景下输出概率对比",
            font=dict(size=14, family="Microsoft YaHei"),
        ),
        font=dict(family="Microsoft YaHei"),
        yaxis=dict(title="P(高)", range=[0, 1.2], dtick=0.2),
        barmode="group",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=80),
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
    )
    fig.update_yaxes(gridcolor="#EEEEEE")

    desc_text = "  |  ".join(f"{s['name']}: {s['desc']}" for s in scenes)
    fig.add_annotation(
        x=0.5, y=-0.25,
        text=desc_text,
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=10, color="#555", family="Microsoft YaHei"),
    )
    return fig


# ============================================================================
# 页面 1: 参数选择页
# ============================================================================

# ── 预设场景定义 ──
SCENES = [
    {"name": "正常天气", "evidence": {}},
    {"name": "暴雨", "evidence": {"降水强度": "高", "风力": "强"}},
    {"name": "最不利内涝", "evidence": {"降水强度": "高", "降水时长": "长", "风力": "强", "海拔": "低", "城市排水能力": "弱"}},
    {"name": "地灾场景", "evidence": {"降水强度": "高", "坡度": "陡", "植被覆盖": "差", "土壤渗透性": "差"}},
]


def _init_param_session_state(input_params):
    """在 widget 渲染之前，一次性初始化所有 param_* session_state（只做一次）"""
    for pname in input_params:
        key = f"param_{pname}"
        if key not in st.session_state:
            st.session_state[key] = "先验/不指定"


def _render_quick_scene_buttons(input_params):
    """快速场景按钮：放在所有 selectbox 之前，写入 session_state 是安全的"""
    st.markdown("##### 快速场景")
    cols = st.columns(4)
    for ci, sc in enumerate(SCENES):
        with cols[ci]:
            if st.button(f"⚡ {sc['name']}", use_container_width=True, key=f"scene_{ci}"):
                # 先全部重置为先验
                for pname in input_params:
                    st.session_state[f"param_{pname}"] = "先验/不指定"
                # 再设置场景参数
                for pname, state in sc["evidence"].items():
                    st.session_state[f"param_{pname}"] = state
                st.rerun()


def render_param_page(engine):
    """渲染参数选择页面"""
    input_params = engine.get_input_params()

    st.title("🌊 灾害链推理引擎 v2")
    st.markdown("##### 选择输入参数，点击「开始预测」查看推理结果")

    # ── 第 1 步：初始化 param_* session_state（必须在任何 widget 之前）──
    _init_param_session_state(input_params)

    # ── 第 2 步：参数已修改提示 ──
    if st.session_state["results"] is not None and st.session_state["params_modified"]:
        st.info("⚠️ 参数已修改，请重新点击「开始预测」", icon="💡")

    # ── 第 3 步：快速场景按钮（在 selectbox 之前，写入 session_state 安全）──
    st.markdown("---")
    _render_quick_scene_buttons(input_params)
    st.markdown("---")

    # ── 第 4 步：按类别分组的参数选择器 ──
    categories = {"气象": [], "水文": [], "地形": [], "地质": [], "城市": []}
    for p in input_params:
        try:
            cfg = engine.get_node_config(p)
            cat = cfg.get("category", "其他")
            if cat in categories:
                categories[cat].append(p)
            else:
                categories.setdefault("其他", []).append(p)
        except KeyError:
            pass

    cat_labels = {
        "气象": "🌤️ 气象 (10)",
        "水文": "💧 水文 (7)",
        "地形": "⛰️ 地形 (8)",
        "地质": "🪨 地质 (8)",
        "城市": "🏙️ 城市 (7)",
    }

    cat_tabs = st.tabs([cat_labels.get(c, c) for c in categories.keys() if categories[c]])

    for tab_idx, (cat, params) in enumerate([(c, p) for c, p in categories.items() if p]):
        with cat_tabs[tab_idx]:
            cols = st.columns(2)
            for i, pname in enumerate(params):
                with cols[i % 2]:
                    try:
                        cfg = engine.get_node_config(pname)
                    except KeyError:
                        continue
                    states = ["先验/不指定"] + cfg["states"]
                    # 用 index=0 控制默认值，不手动写 session_state
                    st.selectbox(
                        f"**{pname}**",
                        states,
                        key=f"param_{pname}",
                    )

    # ── 第 5 步：开始预测按钮 ──
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button("🚀 开始预测", type="primary", use_container_width=True)

    if predict_clicked:
        # 只收集当前 widget 值，不写 param_* 状态
        evidence = {}
        validation_errors = []
        for pname in input_params:
            selected = st.session_state.get(f"param_{pname}", "先验/不指定")
            if selected == "先验/不指定":
                continue
            try:
                cfg = engine.get_node_config(pname)
                if selected not in cfg["states"]:
                    validation_errors.append(
                        f"参数「{pname}」: 状态「{selected}」不在定义范围内 "
                        f"({cfg['states']})"
                    )
                    continue
            except KeyError:
                validation_errors.append(f"参数「{pname}」: 未在配置中定义")
                continue
            evidence[pname] = selected

        if validation_errors:
            st.error("❌ 以下参数存在非法值：\n" + "\n".join(validation_errors))
            st.session_state["error_msg"] = "非法值"
        else:
            with st.spinner("推理中，请稍候..."):
                try:
                    result = engine.infer(evidence)
                    meta = result.pop("_meta", {})
                    result["_meta"] = meta

                    st.session_state["results"] = result
                    st.session_state["meta"] = meta
                    st.session_state["evidence_used"] = dict(evidence)
                    st.session_state["params_modified"] = False
                    st.session_state["error_msg"] = None

                    go_to_page("results")
                except Exception as e:
                    tb.print_exc()
                    st.error(f"❌ 推理过程出错：{str(e)}")
                    st.session_state["error_msg"] = str(e)


# ============================================================================
# 页面 2: 结果页
# ============================================================================

def render_result_page(engine):
    """渲染结果页面"""
    result = st.session_state["results"]
    evidence_used = st.session_state["evidence_used"]
    input_params = engine.get_input_params()
    output_nodes = engine.get_outputs()

    if result is None:
        st.warning("尚无结果，请先执行预测")
        if st.button("← 返回参数选择"):
            go_to_page("params")
        return

    # ── 顶部栏 ──
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← 返回修改参数"):
            st.session_state["params_modified"] = True
            go_to_page("params")
    with col3:
        st.button("🔄 重新预测", type="primary", on_click=lambda: go_to_page("params"))

    # ── 证据摘要 ──
    st.markdown("---")
    ev_col1, ev_col2 = st.columns([2, 1])
    with ev_col1:
        n_provided = len(evidence_used)
        n_expected = len(input_params)
        coverage = n_provided / n_expected if n_expected > 0 else 0.0
        st.markdown(f"##### 证据摘要：已提供 **{n_provided}/{n_expected}** 个参数")
        st.progress(coverage, text=f"证据覆盖度 {coverage*100:.1f}%")
        if evidence_used:
            ev_str = " | ".join(f"{k}={v}" for k, v in evidence_used.items())
            st.caption(f"证据详情：{ev_str}")
        else:
            st.caption("无任何证据参数，全部使用先验推理")
    with ev_col2:
        for on in output_nodes:
            if on in result and "probabilities" in result[on]:
                states = result[on]["states"]
                probs = result[on]["probabilities"]
                max_idx = np.argmax(probs)
                st.metric(on, f"{states[max_idx]}", f"{probs[max_idx]*100:.1f}%")

    # ── 4 个 Tab ──
    tab_network, tab_infer, tab_sensitivity, tab_compare = st.tabs([
        "📊 网络结构",
        "📈 推理结果",
        "🔬 敏感性分析",
        "📋 场景对比",
    ])

    with tab_network:
        st.subheader("贝叶斯网络结构图")
        fig = plotly_network(engine)
        st.plotly_chart(fig, use_container_width=True)

    with tab_infer:
        st.subheader("输出节点概率分布")
        fig = plotly_output_probs(result, output_nodes)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("查看全部节点概率分布"):
            node_list = [n for n in result.keys() if n != "_meta"]
            for node_name in node_list:
                nd = result[node_name]
                if "error" in nd:
                    st.error(f"{node_name}: {nd['error']}")
                    continue
                states = nd["states"]
                probs = nd["probabilities"]
                max_idx = np.argmax(probs)
                prob_str = "  |  ".join(
                    f"{s}: {p*100:.1f}%" for s, p in zip(states, probs)
                )
                st.markdown(f"**{node_name}**")
                st.markdown(f"_{prob_str}_")
                st.progress(probs[max_idx], text=f"最大概率: {states[max_idx]} ({probs[max_idx]*100:.1f}%)")

    with tab_sensitivity:
        st.subheader("全参数敏感性分析")
        target = st.selectbox(
            "选择目标节点",
            output_nodes + ["内涝深度", "地质易发性"],
            index=0,
            key="sensitivity_target",
        )
        with st.spinner(f"计算 {target} 的敏感性..."):
            fig = plotly_sensitivity_top10(engine, target)
            st.plotly_chart(fig, use_container_width=True)

    with tab_compare:
        st.subheader("典型场景对比")
        fig = plotly_scene_compare(engine, output_nodes)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("查看详细数据"):
            table_data = []
            for sc in SCENES:
                r = engine.infer(sc["evidence"])
                row = {"场景": sc["name"]}
                for on in output_nodes:
                    if on in r and "probabilities" in r[on]:
                        states = r[on]["states"]
                        high_state = "高" if "高" in states else states[-1]
                        high_idx = states.index(high_state)
                        row[f"{on} P({high_state})"] = f"{r[on]['probabilities'][high_idx]*100:.1f}%"
                table_data.append(row)
            st.dataframe(table_data, use_container_width=True)


# ============================================================================
# 主入口
# ============================================================================

def main():
    try:
        st.set_page_config(
            page_title="灾害链推理引擎 v2 - 交互演示",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

        init_session_state()

        if not CONFIG_PATH.exists():
            st.error(f"配置文件不存在: {CONFIG_PATH}")
            st.stop()
        engine = load_engine(str(CONFIG_PATH))

        with st.sidebar:
            st.markdown("### 关于本引擎")
            st.markdown(
                """
**灾害链推理引擎 v2** 是一个基于贝叶斯网络的灾害链推理模型。

- **40 个输入参数**（气象/水文/地形/地质/城市）
- **5 个子节点** 聚合输入
- **2 个中间节点** 连接因果链
- **2 个输出节点**（内涝风险/地质灾害概率）

**使用方法**：
1. 选择任意参数的状态
2. 点击"开始预测"
3. 查看推理结果

**提示**：未指定的参数将使用先验概率，体现"弹性输入"特性。
            """
            )
            st.markdown("---")
            st.markdown("模型版本: v2.0-full-40nodes")
            st.markdown("输入参数: 40 | 总节点: 49")

        # 页面路由
        if st.session_state["page"] == "params":
            render_param_page(engine)
        else:
            render_result_page(engine)

    except Exception:
        tb.print_exc()
        raise


if __name__ == "__main__":
    main()