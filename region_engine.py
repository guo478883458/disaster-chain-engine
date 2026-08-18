"""
区域化引擎 — 多区域加载 + 并行推理 + 区域间耦合
=============================================
用法：
  from region_engine import RegionEngine
  re = RegionEngine()
  re.load_region("北京")          # 加载 configs/北京/ 配置包
  subs = re.get_region_list("北京")  # 获取北京16行政区列表
  result = re.infer_region("北京", "海淀区", {"降水强度": "高"})
  results = re.infer_all("北京", {"降水强度": "高"})
  coupled = re.infer_coupled("北京", "昌平区", {"降水强度": "高"})
"""

import copy
import json
import logging
import yaml
from pathlib import Path
from typing import Optional

from bn_engine import DisasterChainEngine

logger = logging.getLogger(__name__)


class RegionEngine:
    """区域化推理引擎，管理多地区配置 + 多区域并行推理 + 耦合传播"""

    def __init__(self, regions_base: str = "configs"):
        self.regions_base = Path(regions_base)
        # 大区名 -> DisasterChainEngine 实例
        self.region_engines: dict[str, DisasterChainEngine] = {}
        # 大区名 -> 子区域列表（从 regions.yaml 加载）
        self.region_lists: dict[str, list[dict]] = {}
        # 大区名 -> 耦合边列表（从 coupling.yaml 加载）
        self.coupling_edges: dict[str, list[dict]] = {}
        # 大区名 -> 原始配置字典
        self.region_configs: dict[str, dict] = {}

    # ── 加载接口 ──

    def load_region(self, region_name: str):
        """
        加载一个地区的配置包（configs/<region_name>/ 下的三件套）。

        Args:
            region_name: 地区目录名，如 "北京"
        """
        region_dir = self.regions_base / region_name
        if not region_dir.exists():
            raise FileNotFoundError(f"地区配置目录不存在: {region_dir}")

        model_path = region_dir / "model.yaml"
        regions_path = region_dir / "regions.yaml"
        coupling_path = region_dir / "coupling.yaml"

        # 1. 加载模型引擎
        if not model_path.exists():
            raise FileNotFoundError(f"模型配置文件不存在: {model_path}")
        self.region_engines[region_name] = DisasterChainEngine(str(model_path))

        # 2. 加载子区域定义
        if regions_path.exists():
            with open(regions_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.region_lists[region_name] = cfg.get("regions", [])
        else:
            self.region_lists[region_name] = []

        # 3. 加载耦合边
        if coupling_path.exists():
            with open(coupling_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.coupling_edges[region_name] = cfg.get("coupling_edges", [])
        else:
            self.coupling_edges[region_name] = []

        # 4. 缓存原始配置
        self.region_configs[region_name] = {
            "model": str(model_path),
            "regions": str(regions_path),
            "coupling": str(coupling_path),
        }

        return self

    def is_loaded(self, region_name: str) -> bool:
        """检查地区是否已加载"""
        return region_name in self.region_engines

    def get_loaded_regions(self) -> list[str]:
        """获取已加载的地区列表"""
        return list(self.region_engines.keys())

    # ── 子区域查询 ──

    def get_region_list(self, region_name: str) -> list[dict]:
        """获取地区下的子区域列表（含属性）"""
        if region_name not in self.region_lists:
            raise KeyError(f"地区 '{region_name}' 未加载，请先调用 load_region()")
        return self.region_lists[region_name]

    def get_subregion_names(self, region_name: str) -> list[str]:
        """获取子区域名称列表"""
        return [r["name"] for r in self.get_region_list(region_name)]

    def get_engine(self, region_name: str) -> DisasterChainEngine:
        """获取地区的引擎实例"""
        if region_name not in self.region_engines:
            raise KeyError(f"地区 '{region_name}' 未加载，请先调用 load_region()")
        return self.region_engines[region_name]

    # ── 单区域推理 ──

    def infer_region(self, region_name: str, evidence: dict = None) -> dict:
        """
        对单个地区执行推理（标准推理，无耦合）。

        Args:
            region_name: 地区名
            evidence: 证据字典，如 {"降水强度": "高", "风力": "强"}

        Returns:
            dict: 推理结果（含 _meta）
        """
        engine = self.get_engine(region_name)
        return engine.infer(evidence or {})

    # ── 多区域并行推理 ──

    def infer_all(self, region_name: str, evidence: dict = None) -> dict[str, dict]:
        """
        对所有子区域执行独立推理（共享同一证据）。

        Args:
            region_name: 地区名
            evidence: 证据字典

        Returns:
            dict: 子区域名 -> 推理结果
        """
        evidence = evidence or {}
        results = {}
        for sub in self.get_region_list(region_name):
            sub_name = sub["name"]
            try:
                # 使用子区域名称作为引擎标识（目前所有子区域共享同一模型）
                # 若将来各子区域有独立模型，可在此扩展
                result = self.infer_region(region_name, evidence)
                results[sub_name] = result
            except Exception as e:
                results[sub_name] = {"_meta": {"error": str(e)}}
        return results

    # ── 区域耦合推理 ──

    def infer_coupled(self, region_name: str, source_subregion: str,
                      evidence: dict = None) -> dict:
        """
        带区域间耦合的级联推理。

        步骤：
          1. 对源子区域执行标准推理（给定证据）
          2. 沿耦合边传播：源子区域输出 → 影响下游子区域输入
          3. 对受影响的下游子区域重新推理

        耦合机制（修复后）：
          - 每条耦合边定义 source_state / target_state 显式映射
          - 源节点最可能状态 == source_state 时，将 target_state 作为证据传入目标区域
          - 权重 w > 0.5 的边强耦合（设置证据）；w ≤ 0.5 的边弱耦合（不设置证据，仅报告）
          - 级联 Δ 按 w 缩放（delta_weighted = delta_raw * w），体现权重梯度
          - 推理异常抛出日志，禁止静默 fallback 成 0

        Args:
            region_name: 地区名
            source_subregion: 源子区域名称（如 "昌平区"）
            evidence: 作用于源子区域的证据

        Returns:
            dict: {
                "source": 源子区域名,
                "source_result": 源子区域推理结果,
                "downstream": {下游子区域名: 推理结果},
                "cascade": [级联影响列表],
            }
        """
        evidence = evidence or {}
        engine = self.get_engine(region_name)
        sub_names = self.get_subregion_names(region_name)
        edges = self.coupling_edges.get(region_name, [])

        # 1. 源子区域独立推理
        source_result = engine.infer(evidence)
        cascade = []

        # 2. 第一遍遍历：收集所有下游证据，按目标合并
        #    target_evidence_map[target] = {target_node: target_state, …}
        #    edge_info_list 保留每条边的元信息用于后续级联计算
        target_evidence_map = {}  # target -> {target_node: state}
        edge_info_list = []       # 保存每条边的元信息

        for edge in edges:
            if edge["source"] != source_subregion:
                continue
            target = edge["target"]
            if target not in sub_names:
                continue

            source_node = edge.get("source_node", "暴雨强度")
            target_node = edge.get("target_node", "暴雨强度")
            weight = edge.get("weight", 0.5)
            source_state = edge.get("source_state", "强")
            target_state = edge.get("target_state", "强")

            # 获取源节点最可能状态
            if source_node not in source_result or "probabilities" not in source_result[source_node]:
                logger.warning("耦合边 %s→%s: 源节点 %s 无概率结果", source_subregion, target, source_node)
                continue
            src_probs = source_result[source_node]["probabilities"]
            src_states = source_result[source_node]["states"]
            src_max_prob = max(src_probs)
            src_max_state = src_states[src_probs.index(src_max_prob)]

            # 判断是否激活
            active = False
            if src_max_state == source_state:
                if weight > 0.5:
                    active = True
                    # 强耦合：合并证据到目标
                    if target not in target_evidence_map:
                        target_evidence_map[target] = {}
                    target_evidence_map[target][target_node] = target_state
                    logger.info("耦合边 %s→%s (%s, w=%.2f): %s=%s → %s=%s",
                                source_subregion, target, edge.get("channel", "?"),
                                weight, source_node, src_max_state, target_node, target_state)
                else:
                    logger.info("耦合边 %s→%s (%s, w=%.2f): 弱耦合，不设置证据",
                                source_subregion, target, edge.get("channel", "?"), weight)
            else:
                logger.info("耦合边 %s→%s: 源节点最可能状态 %s ≠ %s，不触发",
                            source_subregion, target, src_max_state, source_state)

            edge_info_list.append({
                "target": target,
                "channel": edge.get("channel", "unknown"),
                "weight": weight,
                "source_node": source_node,
                "target_node": target_node,
                "active": active,
                "src_max_state": src_max_state,
                "src_max_prob": src_max_prob,
            })

        # 3. 先验推理（无证据），用于后续级联 Δ 计算
        prior_result = engine.infer({})

        # 4. 按目标执行耦合推理（每个目标只算一次，合并所有证据）
        downstream_results = {}
        for target, combined_evidence in target_evidence_map.items():
            try:
                target_result = engine.infer(combined_evidence)
                downstream_results[target] = target_result
                logger.info("下游推理 %s: 证据=%s", target, combined_evidence)
            except Exception as e:
                logger.error("下游推理 %s 失败: %s", target, e)
                downstream_results[target] = {"_meta": {"error": str(e)}}

        # 5. 计算级联概率变化（每条边一条记录）
        def _get_high_prob(result_dict, node_name):
            """获取节点"高"状态的概率（若节点无"高"状态则取最后一个）"""
            if node_name not in result_dict or "probabilities" not in result_dict[node_name]:
                return 0.0
            states = result_dict[node_name]["states"]
            probs = result_dict[node_name]["probabilities"]
            high_state = "高" if "高" in states else states[-1]
            idx = states.index(high_state)
            return probs[idx]

        for info in edge_info_list:
            target = info["target"]
            target_result = downstream_results.get(target, {})
            if "_meta" in target_result and "error" in target_result["_meta"]:
                continue

            for output_node in ["内涝风险", "地质灾害概率", "内涝深度", "地质易发性"]:
                if output_node not in target_result or "probabilities" not in target_result.get(output_node, {}):
                    continue
                if output_node not in prior_result or "probabilities" not in prior_result.get(output_node, {}):
                    continue
                post_prob = _get_high_prob(target_result, output_node)
                prior_prob = _get_high_prob(prior_result, output_node)
                delta = round(post_prob - prior_prob, 4)
                weighted_delta = round(delta * info["weight"], 4)

                cascade.append({
                    "source": source_subregion,
                    "target": target,
                    "channel": info["channel"],
                    "weight": info["weight"],
                    "source_node": info["source_node"],
                    "target_node": info["target_node"],
                    "output_node": output_node,
                    f"P({output_node}=高)_prior": round(prior_prob, 4),
                    f"P({output_node}=高)_coupled": round(post_prob, 4),
                    "delta_raw": delta,
                    "delta_weighted": weighted_delta,
                })

        return {
            "source": source_subregion,
            "source_result": source_result,
            "downstream": downstream_results,
            "cascade": cascade,
        }

    # ── 全区域耦合推理（按拓扑顺序传播） ──

    def infer_coupled_all(self, region_name: str,
                          evidence_dict: dict[str, dict] = None) -> dict:
        """
        全区域耦合推理：为多个子区域提供证据，沿耦合边拓扑传播。

        Args:
            region_name: 地区名
            evidence_dict: 子区域 -> 证据字典，如 {"昌平区": {"降水强度": "高"}}

        Returns:
            dict: 各子区域推理结果 + 级联影响列表
        """
        evidence_dict = evidence_dict or {}
        engine = self.get_engine(region_name)
        sub_names = self.get_subregion_names(region_name)
        edges = self.coupling_edges.get(region_name, [])

        # 1. 构建拓扑顺序
        # 计算每个节点的入度
        in_degree = {s: 0 for s in sub_names}
        adj = {s: [] for s in sub_names}
        for e in edges:
            src, tgt = e["source"], e["target"]
            if src in sub_names and tgt in sub_names:
                adj[src].append(e)
                in_degree[tgt] = in_degree.get(tgt, 0) + 1

        # 拓扑排序（Kahn 算法）
        from collections import deque
        q = deque()
        for s in sub_names:
            if in_degree.get(s, 0) == 0:
                q.append(s)

        topo_order = []
        while q:
            node = q.popleft()
            topo_order.append(node)
            for e in adj[node]:
                tgt = e["target"]
                in_degree[tgt] -= 1
                if in_degree[tgt] == 0:
                    q.append(tgt)

        # 将有耦合边的子区域加入顺序（未在拓扑中的也加入末尾）
        for s in sub_names:
            if s not in topo_order:
                topo_order.append(s)

        # 2. 按拓扑顺序推理，累积耦合影响
        results = {}
        cascade = []

        # 每个子区域累积的额外证据（来自上游耦合）
        extra_evidence = {s: {} for s in sub_names}

        for sub in topo_order:
            # 合并基础证据 + 上游耦合证据
            combined_ev = {}
            combined_ev.update(extra_evidence.get(sub, {}))
            combined_ev.update(evidence_dict.get(sub, {}))

            try:
                result = engine.infer(combined_ev)
                results[sub] = result
            except Exception as e:
                results[sub] = {"_meta": {"error": str(e)}}
                continue

            # 跳过出错结果
            if "_meta" in results[sub] and "error" in results[sub]["_meta"]:
                continue

            # 沿出边传播
            for e in adj.get(sub, []):
                target = e["target"]
                source_node = e.get("source_node", "暴雨强度")
                target_node = e.get("target_node", "暴雨强度")
                weight = e.get("weight", 0.5)
                source_state = e.get("source_state", "强")
                target_state = e.get("target_state", "强")

                # 获取源节点最可能状态
                if source_node not in results[sub] or "probabilities" not in results[sub][source_node]:
                    continue
                src_probs = results[sub][source_node]["probabilities"]
                src_states = results[sub][source_node]["states"]
                src_max_prob = max(src_probs)
                src_max_state = src_states[src_probs.index(src_max_prob)]

                # 用显式映射判断是否传播耦合证据
                if src_max_state == source_state and weight > 0.5:
                    extra_evidence[target][target_node] = target_state
                    logger.info("全拓扑传播: %s→%s (%s, w=%.2f): %s=%s → %s=%s",
                                sub, target, e.get("channel", "?"), weight,
                                source_node, src_max_state, target_node, target_state)

                # 记录级联影响
                prior_result = engine.infer(evidence_dict.get(sub, {}))
                for output_node in ["内涝风险", "地质灾害概率"]:
                    if output_node in results[sub]:
                        states = results[sub][output_node]["states"]
                        probs = results[sub][output_node]["probabilities"]
                        high_state = "高" if "高" in states else states[-1]
                        post_prob = probs[states.index(high_state)]

                        prior_states = prior_result[output_node]["states"]
                        prior_probs = prior_result[output_node]["probabilities"]
                        prior_prob = prior_probs[prior_states.index(high_state)]

                        delta = round(post_prob - prior_prob, 4)
                        if abs(delta) > 0.001:
                            cascade.append({
                                "source": sub,
                                "target": target,
                                "channel": e.get("channel", "unknown"),
                                "weight": weight,
                                "output_node": output_node,
                                f"P({output_node}=高)_prior": round(prior_prob, 4),
                                f"P({output_node}=高)_coupled": round(post_prob, 4),
                                "delta": delta,
                                "delta_weighted": round(delta * weight, 4),
                            })

        return {
            "results": results,
            "cascade": cascade,
            "topo_order": topo_order,
        }

    # ── 导出 ──

    def export_results(self, results: dict, output_path: str):
        """导出推理结果到 JSON 文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已导出: {output_path}")

    # ── 摘要信息 ──

    def summary(self, region_name: str) -> dict:
        """获取地区的摘要信息"""
        engine = self.get_engine(region_name)
        sub_list = self.get_region_list(region_name)
        edges = self.coupling_edges.get(region_name, [])

        return {
            "region": region_name,
            "model": engine.config["model"]["name"],
            "total_nodes": len(engine.get_node_names()),
            "input_params": len(engine.get_input_params()),
            "outputs": engine.get_outputs(),
            "subregions": len(sub_list),
            "subregion_names": [r["name"] for r in sub_list],
            "coupling_edges": len(edges),
            "coupling_summary": {
                "meteorological": len([e for e in edges if e.get("channel") == "meteorological"]),
                "hydrological": len([e for e in edges if e.get("channel") == "hydrological"]),
            },
        }