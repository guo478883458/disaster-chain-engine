# -*- coding: utf-8 -*-
"""
灾害链推理引擎 v2 - 交互式演示（现场改参数）
================================================
用法：
    D:\\ana\\envs\\disasterlex\\python.exe demo_interactive.py

操作：全程输入数字即可（避免中文输入/编码问题）。
  1. 修改参数：选择参数编号 → 选择状态编号 → 自动重新推理
  2. 推理显示：手动触发（也可在改完参数后自动显示）
  3. 清空证据 / 4. 退出
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bn_engine import DisasterChainEngine

ENGINE = DisasterChainEngine("configs/config_40nodes.yaml")

# 现场常用演示参数（编号 → 节点名）
QUICK_PARAMS = [
    "降水强度", "风力", "海拔", "坡度",
    "历史排水时间", "下垫面硬化率", "植被覆盖", "土壤渗透性",
    "前期土壤含水量", "气温", "湿度", "气压",
]

SHOW_NODES = ["内涝深度", "内涝风险", "地质灾害概率"]


def fmt_dist(dist: dict) -> str:
    if "error" in dist:
        return f"（{dist['error'][:25]}…）"
    return "  ".join(f"{k}={v * 100:.1f}%" for k, v in
                     zip(dist.get("states", []), dist.get("probabilities", [])))


def show_result(evidence: dict) -> None:
    print("\n" + "=" * 56)
    print(f"  推理结果（证据 {len(evidence)}/40 个参数）")
    print("=" * 56)
    result = ENGINE.infer(evidence)
    for node in SHOW_NODES:
        print(f"  {node:8s} {fmt_dist(result[node])}")
    print("=" * 56 + "\n")


def pick_state(node: str) -> str:
    states = ENGINE.get_node_states(node) if hasattr(ENGINE, "get_node_states") else None
    if states is None:
        import yaml
        cfg = yaml.safe_load(open("configs/config_40nodes.yaml", encoding="utf-8"))
        states = next(n["states"] for n in cfg["nodes"] if n["name"] == node)
    print(f"  [{node}] 可选状态:")
    for i, s in enumerate(states, 1):
        print(f"    {i}. {s}")
    while True:
        try:
            idx = int(input("  输入状态编号: "))
            if 1 <= idx <= len(states):
                return states[idx - 1]
        except (ValueError, EOFError):
            pass
        print("  ⚠ 编号无效，请重试")


def main() -> None:
    evidence: dict = {}
    print("=" * 56)
    print("  灾害链推理引擎 v2 交互演示（输入数字操作）")
    print("=" * 56)
    while True:
        print("\n当前证据: " + ("、".join(f"{k}={v}" for k, v in evidence.items()) or "（无）"))
        print("-" * 56)
        print("  1. 修改参数    2. 推理显示    3. 清空证据    4. 退出")
        try:
            choice = input("  请选择: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if choice == "1":
            print("\n  常用参数:")
            for i, p in enumerate(QUICK_PARAMS, 1):
                cur = f"（当前: {evidence[p]}）" if p in evidence else ""
                print(f"    {i:2d}. {p} {cur}")
            print(f"    {len(QUICK_PARAMS) + 1:2d}. 全部 40 个参数（输入编号查询）")
            try:
                pidx = int(input("  选择参数编号: "))
            except ValueError:
                continue
            if pidx == len(QUICK_PARAMS) + 1:
                print("  完整参数列表:")
                for i, p in enumerate(ENGINE.get_input_params(), 1):
                    print(f"    {i:2d}. {p}", end="")
                    print(f"（当前: {evidence[p]}）" if p in evidence else "")
                try:
                    pidx = int(input("  输入参数编号: "))
                    node = ENGINE.get_input_params()[pidx - 1]
                except (ValueError, IndexError):
                    print("  ⚠ 编号无效")
                    continue
            elif 1 <= pidx <= len(QUICK_PARAMS):
                node = QUICK_PARAMS[pidx - 1]
            else:
                print("  ⚠ 编号无效")
                continue
            evidence[node] = pick_state(node)
            show_result(evidence)  # 改完自动推理，现场效果连贯
        elif choice == "2":
            show_result(evidence)
        elif choice == "3":
            evidence.clear()
            print("  ✅ 证据已清空（回到先验）")
            show_result(evidence)
        elif choice == "4":
            print("  演示结束")
            break
        else:
            print("  ⚠ 请输入 1-4")


if __name__ == "__main__":
    main()
