"""
灾害链贝叶斯网络推理引擎 v2
================================
配置驱动的通用灾害链推理引擎，支持：
- 从 YAML 配置构建贝叶斯网络
- 弹性输入（部分证据 + 覆盖度提示）
- 敏感性分析
- 数据学习接口（fit_from_data）
"""

import itertools
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 配置 matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator


# ============================================================================
# 工具函数
# ============================================================================

def _cartesian_product(lists):
    """笛卡尔积，用于展开多父节点 CPT 组合"""
    return list(itertools.product(*lists))


def _normalize(probs):
    """归一化概率向量"""
    s = sum(probs)
    if s <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [round(p / s, 4) for p in probs]


# ============================================================================
# CPT 生成器
# ============================================================================

def _gen_uniform_cpd(child_states, num_combinations):
    """均匀先验 CPT"""
    k = len(child_states)
    probs = [1.0 / k] * k
    return [[probs[s]] * num_combinations for s in range(k)]


def _gen_explicit_cpd(child_states, parent_states_list, values):
    """
    显式 CPT。
    values: list of lists, 每个子状态一行，每列对应一个父状态组合
    """
    num_combinations = 1
    for st in parent_states_list:
        num_combinations *= len(st)

    # 校验维度
    expected_rows = len(child_states)
    assert len(values) == expected_rows, (
        f"CPT 行数 {len(values)} 不匹配子状态数 {expected_rows}"
    )
    for i, row in enumerate(values):
        assert len(row) == num_combinations, (
            f"CPT 第 {i} 行列数 {len(row)} 不匹配组合数 {num_combinations}"
        )

    # 确保每列归一化
    result = []
    for s_idx in range(len(child_states)):
        result.append([round(float(v), 4) for v in values[s_idx]])

    # 归一化检查
    for col in range(num_combinations):
        col_sum = sum(result[row][col] for row in range(len(child_states)))
        if abs(col_sum - 1.0) > 0.01:
            # 自动归一化
            norm = [result[row][col] / col_sum for row in range(len(child_states))]
            for row in range(len(child_states)):
                result[row][col] = round(norm[row], 4)

    return result


def _gen_weighted_sum_cpd(child_states, parent_states_list, parent_names, config):
    """
    加权和 CPT（2 状态节点专用）。
    P(target_state) = min(sum(weight_for_each_parent_state), max_prob)
    P(other_state) = 1 - P(target_state)
    """
    state_names = [st for st in parent_states_list]
    target_state = config["target_state"]
    max_prob = config.get("max_prob", 1.0)
    weights_cfg = config["weights"]

    # 找到 target_state 的索引
    target_idx = child_states.index(target_state)
    other_idx = 1 - target_idx

    # 计算每个父状态组合的权重和
    num_combos = 1
    for st in parent_states_list:
        num_combos *= len(st)

    result = [[0.0] * num_combos for _ in range(len(child_states))]

    for combo_idx, combo in enumerate(itertools.product(*[range(len(st)) for st in parent_states_list])):
        weight_sum = 0.0
        for p_idx, p_name in enumerate(parent_names):
            state_value = parent_states_list[p_idx][combo[p_idx]]
            w = weights_cfg.get(p_name, {}).get(state_value, 0.0)
            weight_sum += w
        prob_target = min(weight_sum, max_prob)
        prob_target = round(prob_target, 4)
        prob_other = round(1.0 - prob_target, 4)
        result[target_idx][combo_idx] = prob_target
        result[other_idx][combo_idx] = prob_other

    return result


def _gen_multi_weighted_cpd(child_states, parent_states_list, parent_names, config):
    """
    多状态加权和 CPT。
    支持三种子规则：
      - type: weighted_sum  (默认): 权重累加，max_prob 封顶
      - type: scaled_residual: P = min(base, (1 - P(reference)) * scale)
      - type: residual: P = max(1 - sum(已定义states), 0)
    """
    num_combos = 1
    for st in parent_states_list:
        num_combos *= len(st)

    state_defs = config["states"]
    result = {}

    for combo_idx, combo in enumerate(itertools.product(*[range(len(st)) for st in parent_states_list])):
        # 先计算所有 state 的原始值
        raw = {}
        for sd in state_defs:
            name = sd["name"]
            stype = sd.get("type", "weighted_sum")

            if stype == "residual":
                # 依赖其他已计算状态
                subtract = sd.get("subtract", [])
                subtract_names = [s["name"] for s in state_defs if s.get("type") == "residual"]
                # 自动推断：所有非 residual 的已定义状态
                defined = [s["name"] for s in state_defs if s["name"] != name
                           and s.get("type") != "residual"]
                used = [raw[d] for d in defined]
                val = max(1.0 - sum(used), 0.0)
                raw[name] = val

            elif stype == "scaled_residual":
                ref_name = sd["residual_of"]
                base = sd["base"]
                scale = sd["scale"]
                ref_val = raw.get(ref_name, 0.0)
                val = min(base, (1.0 - ref_val) * scale)
                raw[name] = val

            else:  # weighted_sum (default)
                weights_cfg = sd.get("weights", {})
                max_prob = sd.get("max_prob", 1.0)
                weight_sum = 0.0
                for p_idx, p_name in enumerate(parent_names):
                    state_value = parent_states_list[p_idx][combo[p_idx]]
                    w = weights_cfg.get(p_name, {}).get(state_value, 0.0)
                    weight_sum += w
                val = min(weight_sum, max_prob)
                raw[name] = val

        # 归一化
        vals = [raw[sd["name"]] for sd in state_defs]
        total = sum(vals)
        if total > 0:
            vals = [v / total for v in vals]
        vals = [round(v, 4) for v in vals]

        for s_idx, sd in enumerate(state_defs):
            name = sd["name"]
            if name not in result:
                result[name] = [0.0] * num_combos
            result[name][combo_idx] = vals[s_idx]

    # 按 child_states 顺序输出
    return [[result[s][c] for c in range(num_combos)] for s in child_states]


# CPT 生成器注册表
CPT_GENERATORS = {
    "uniform": _gen_uniform_cpd,
    "explicit": _gen_explicit_cpd,
    "weighted_sum": _gen_weighted_sum_cpd,
    "multi_weighted": _gen_multi_weighted_cpd,
}


# ============================================================================
# 核心引擎
# ============================================================================

class DisasterChainEngine:
    """灾害链贝叶斯网络推理引擎"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self.inference = None
        self._build_model()

    # ── 配置加载 ──

    def _load_config(self):
        """加载并校验 YAML 配置"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 校验必要字段
        assert "nodes" in cfg, "配置缺少 nodes 字段"
        assert "edges" in cfg, "配置缺少 edges 字段"

        # 建立节点索引
        node_names = {n["name"] for n in cfg["nodes"]}
        for edge in cfg["edges"]:
            assert len(edge) == 2, f"边定义错误: {edge}"
            assert edge[0] in node_names, f"边起点 {edge[0]} 未定义"
            assert edge[1] in node_names, f"边终点 {edge[1]} 未定义"

        # 给节点添加默认 cpd 类型
        for n in cfg["nodes"]:
            if "cpd" not in n:
                n["cpd"] = {"type": "uniform"}

        return cfg

    def get_node_config(self, name: str) -> dict:
        """按名称获取节点配置"""
        for n in self.config["nodes"]:
            if n["name"] == name:
                return n
        raise KeyError(f"节点 '{name}' 未找到")

    def get_node_names(self) -> list[str]:
        """获取所有节点名称"""
        return [n["name"] for n in self.config["nodes"]]

    def get_input_params(self) -> list[str]:
        """获取输入参数列表"""
        return self.config.get("input_params", [])

    def get_outputs(self) -> list[str]:
        """获取输出节点列表"""
        return self.config.get("outputs", [])

    # ── 模型构建 ──

    def _build_model(self):
        """根据配置构建贝叶斯网络"""
        node_names = self.get_node_names()
        edges = self.config["edges"]

        # 1. 构建 DAG
        self.model = BayesianNetwork(edges)

        # 2. 为每个节点生成 CPT
        cpds = []
        for node_cfg in self.config["nodes"]:
            name = node_cfg["name"]
            states = node_cfg["states"]
            cpd_cfg = node_cfg["cpd"]
            cpd_type = cpd_cfg.get("type", "uniform")

            # 确定父节点
            parents = cpd_cfg.get("parents", [])
            parent_states = []
            for p in parents:
                p_cfg = self.get_node_config(p)
                parent_states.append(p_cfg["states"])

            # 生成 CPT 值
            generator = CPT_GENERATORS.get(cpd_type)
            if generator is None:
                raise ValueError(f"未知的 CPT 类型: {cpd_type} (节点 {name})")

            if cpd_type == "uniform":
                num_combos = 1
                for st in parent_states:
                    num_combos *= len(st)
                values = generator(states, num_combos)
            elif cpd_type == "explicit":
                values = generator(states, parent_states, cpd_cfg["values"])
            elif cpd_type == "weighted_sum":
                values = generator(states, parent_states, parents, cpd_cfg)
            elif cpd_type == "multi_weighted":
                values = generator(states, parent_states, parents, cpd_cfg)
            else:
                raise ValueError(f"未处理的 CPT 类型: {cpd_type}")

            # 构建 TabularCPD
            evidence_card = [len(s) for s in parent_states]
            # 构建 state_names 字典（包含父节点）
            state_names_dict = {name: states}
            for p in parents:
                p_cfg = self.get_node_config(p)
                state_names_dict[p] = p_cfg["states"]
            cpd = TabularCPD(
                variable=name,
                variable_card=len(states),
                values=values,
                evidence=parents if parents else None,
                evidence_card=evidence_card if evidence_card else None,
                state_names=state_names_dict,
            )
            cpds.append(cpd)

        # 3. 添加所有 CPT 到模型
        self.model.add_cpds(*cpds)

        # 4. 校验模型
        assert self.model.check_model(), "模型校验失败"

        # 5. 创建推理器
        self.inference = VariableElimination(self.model)

    # ── 推理 ──

    def infer(self, evidence: dict[str, str] = None) -> dict:
        """
        执行推理。

        Args:
            evidence: 证据字典，如 {"降水强度": "高", "坡度": "陡"}

        Returns:
            推理结果 dict，包含各节点概率分布 + 证据覆盖度
        """
        evidence = evidence or {}
        result = {}

        # 计算证据覆盖度
        input_params = self.get_input_params()
        provided = set(evidence.keys())
        expected = set(input_params)
        covered = provided & expected
        missing = expected - provided
        coverage = len(covered) / len(expected) if expected else 1.0

        result["_meta"] = {
            "model": self.config["model"]["name"],
            "evidence_provided": len(provided),
            "evidence_expected": len(expected),
            "evidence_coverage": round(coverage, 4),
            "missing_params": sorted(missing),
        }

        # 执行推理
        for node in self.get_node_names():
            try:
                query = self.inference.query([node], evidence=evidence)
                probs = query.values
                states = self.get_node_config(node)["states"]
                result[node] = {
                    "states": states,
                    "probabilities": [round(float(p), 4) for p in probs],
                }
            except Exception as e:
                result[node] = {"error": str(e)}

        return result

    # ── 敏感性分析 ──

    def sensitivity_analysis(self, target_node: str, evidence: dict = None) -> pd.DataFrame:
        """
        全参数敏感性分析：逐个变化每个输入参数的所有状态，观察对目标节点的影响。

        Returns:
            DataFrame: 每行 = 一个参数的一个状态，包含原始概率和 ΔP
        """
        evidence = evidence or {}
        input_params = self.get_input_params()

        # 基础推理（无证据或给定证据）
        base_result = self.infer(evidence)
        base_probs = base_result[target_node]["probabilities"]
        base_states = base_result[target_node]["states"]

        rows = []
        for param in input_params:
            if param in evidence:
                continue  # 跳过已提供证据的参数
            node_cfg = self.get_node_config(param)
            for state in node_cfg["states"]:
                test_ev = {**evidence, param: state}
                try:
                    test_result = self.infer(test_ev)
                    test_probs = test_result[target_node]["probabilities"]
                    for s_idx, s_name in enumerate(base_states):
                        delta = round(test_probs[s_idx] - base_probs[s_idx], 4)
                        rows.append({
                            "parameter": param,
                            "state": state,
                            "target": target_node,
                            f"P({s_name})_base": base_probs[s_idx],
                            f"P({s_name})_test": test_probs[s_idx],
                            f"ΔP({s_name})": delta,
                        })
                except Exception:
                    pass

        df = pd.DataFrame(rows)

        # 计算每个参数的平均绝对影响
        if not df.empty:
            delta_cols = [c for c in df.columns if c.startswith("ΔP(")]
            if delta_cols:
                df["avg_|ΔP|"] = df[delta_cols].abs().mean(axis=1)
                df = df.sort_values("avg_|ΔP|", ascending=False)

        return df

    def plot_sensitivity(self, df: pd.DataFrame, target_node: str, output_path: str = None):
        """绘制敏感性分析柱状图"""
        if df.empty:
            print("敏感性分析数据为空，无法绘图")
            return

        delta_cols = [c for c in df.columns if c.startswith("ΔP(")]
        if not delta_cols:
            return

        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))

        # 为每个参数取最大 |ΔP|
        plot_data = df.groupby("parameter").apply(
            lambda g: g.loc[g["avg_|ΔP|"].idxmax()]
        ).reset_index(drop=True)
        plot_data = plot_data.sort_values("avg_|ΔP|", ascending=True)

        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in plot_data["avg_|ΔP|"]]

        ax.barh(plot_data["parameter"], plot_data["avg_|ΔP|"], color=colors, edgecolor="gray")
        ax.set_xlabel("平均 |ΔP|")
        ax.set_title(f"参数敏感性分析 — 目标: {target_node}")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"敏感性图已保存: {output_path}")
        plt.close()

    # ── 数据学习接口 ──

    def fit_from_data(self, data: pd.DataFrame, estimator: str = "mle"):
        """
        从数据学习 CPT，替换专家 CPT。

        Args:
            data: DataFrame，列名为节点名称，值为状态名
            estimator: 估计器类型，支持 "mle" (最大似然) 或 "bayes" (贝叶斯)
        """
        if estimator == "mle":
            est = MaximumLikelihoodEstimator
        else:
            raise ValueError(f"不支持的估计器: {estimator}")

        # 检查数据列
        node_names = self.get_node_names()
        missing_cols = [n for n in node_names if n not in data.columns]
        if missing_cols:
            print(f"警告: 数据中缺少以下节点列: {missing_cols}")

        self.model.fit(data, estimator=est)
        self.inference = VariableElimination(self.model)
        print(f"模型已从数据学习（{estimator}），CPT 已更新")

    # ── 导出 ──

    def export_results(self, results: dict, output_path: str):
        """导出推理结果到 JSON 文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已导出: {output_path}")

    # ── 模型摘要 ──

    def summary(self) -> dict:
        """打印模型摘要"""
        nodes = self.config["nodes"]
        edges = self.config["edges"]

        info = {
            "model": self.config["model"]["name"],
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "input_params": self.get_input_params(),
            "outputs": self.get_outputs(),
            "nodes_by_category": {},
        }

        for n in nodes:
            cat = n.get("category", "未分类")
            if cat not in info["nodes_by_category"]:
                info["nodes_by_category"][cat] = []
            info["nodes_by_category"][cat].append({
                "name": n["name"],
                "states": n["states"],
                "cpd_type": n["cpd"]["type"],
            })

        return info

    def print_summary(self):
        """打印模型摘要"""
        info = self.summary()
        print("=" * 60)
        print(f"  模型: {info['model']}")
        print(f"  节点数: {info['total_nodes']}  |  边数: {info['total_edges']}")
        print(f"  输入参数: {len(info['input_params'])}  |  输出: {len(info['outputs'])}")
        print("-" * 60)
        for cat, nodes in info["nodes_by_category"].items():
            print(f"  [{cat}]")
            for n in nodes:
                states_str = "/".join(n["states"])
                print(f"    {n['name']}  ({states_str})  [{n['cpd_type']}]")
        print("=" * 60)