"""
演示参数准备脚本
================
一键生成演示参数包（JSON），写入 configs/郑州/demo_params.json，
dashboard 启动时自动加载此文件，替代手工逐项填参数。

支持多场景演示，覆盖不同数据完整度情况。

用法:
  # 完整场景（默认，12 区 16 证据 + 3 张图/区）
  python scripts/prepare_demo_params.py --all

  # 部分区有数据（样本不足演示）
  python scripts/prepare_demo_params.py --scenario partial

  # 仅 1 个区有完整数据（极端稀疏）
  python scripts/prepare_demo_params.py --scenario single

  # 全部区仅数学参数（无图片）
  python scripts/prepare_demo_params.py --scenario params-only

  # 单区操作
  python scripts/prepare_demo_params.py --district 金水区
  python scripts/prepare_demo_params.py --district 金水区 --no-images
  python scripts/prepare_demo_params.py --district 巩义市 --wl 水位图.jpg --road 道路图.jpg

  # 列出当前各区已配置的参数
  python scripts/prepare_demo_params.py --list
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_PATH = ROOT / "configs" / "郑州" / "district_evidence.json"
OUTPUT_PATH = ROOT / "configs" / "郑州" / "demo_params.json"

# 演示图片默认目录
DEFAULT_IMAGE_DIR = ROOT / ".." / "综合演示" / "1-输入图片"
# 备选路径（如果相对路径不可用）
ALT_IMAGE_DIR = Path("H:/实习/综合演示/1-输入图片")

# 演示图片映射：任务名 → 期望文件名
DEMO_IMAGES = {
    "flood": "1-洪水现场.jpg",
    "water_level": "2-河道水位尺.jpg",
    "road": "3-道路路面.jpg",
}

# 郑州 12 区
ALL_DISTRICTS = [
    "中原区", "二七区", "金水区", "管城回族区", "惠济区", "上街区",
    "巩义市", "登封市", "新密市",
    "中牟县", "新郑市", "荥阳市",
]

# 场景定义
# partial 场景的代表区：城区/山区/平原 三种地理类型
PARTIAL_DISTRICTS = ["金水区", "巩义市", "中牟县"]
SINGLE_DISTRICT = "金水区"


def _find_image_dir() -> Path:
    """查找演示图片目录，返回存在的路径"""
    for d in [DEFAULT_IMAGE_DIR, ALT_IMAGE_DIR]:
        if d.exists():
            return d
    return None


def _resolve_image_path(task: str, user_path: str = None) -> str:
    """
    解析图片路径。
    优先级: 用户指定路径 > 默认演示目录
    """
    if user_path:
        p = Path(user_path)
        if p.exists():
            return str(p.resolve())
        # 尝试相对默认目录
        img_dir = _find_image_dir()
        if img_dir:
            p2 = img_dir / user_path
            if p2.exists():
                return str(p2.resolve())
        print(f"  ⚠ 警告: 指定的图片不存在: {user_path}")
        return None

    # 从默认演示目录取
    img_dir = _find_image_dir()
    if img_dir is None:
        print("  ⚠ 警告: 演示图片目录不存在，跳过图片")
        return None

    fname = DEMO_IMAGES.get(task)
    if not fname:
        return None

    img_path = img_dir / fname
    if img_path.exists():
        return str(img_path.resolve())
    else:
        print(f"  ⚠ 警告: 未找到 {task} 图片: {img_path}")
        return None


def load_evidence() -> dict:
    """加载各区默认证据 JSON"""
    if not EVIDENCE_PATH.exists():
        print(f"❌ 错误: 未找到证据文件 {EVIDENCE_PATH}")
        sys.exit(1)
    with open(EVIDENCE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_single_district(district: str, evidence: dict,
                          no_images: bool = False,
                          wl_path: str = None,
                          road_path: str = None,
                          flood_path: str = None) -> dict:
    """构建单区演示参数包"""
    if district not in evidence:
        print(f"❌ 错误: 未知区名 '{district}'，有效区: {list(evidence.keys())}")
        sys.exit(1)

    entry = {
        "evidence": evidence[district],
        "images": {},
    }

    if not no_images:
        # 水位尺图
        img_path = _resolve_image_path("water_level", wl_path)
        if img_path:
            entry["images"]["water_level"] = img_path

        # 道路图
        img_path = _resolve_image_path("road", road_path)
        if img_path:
            entry["images"]["road"] = img_path

        # 洪水图
        img_path = _resolve_image_path("flood", flood_path)
        if img_path:
            entry["images"]["flood"] = img_path

    return entry


def build_empty_district() -> dict:
    """构建空区（无证据、无图片），保持先验"""
    return {
        "evidence": {},
        "images": {},
    }


# ============================================================================
# 场景构建函数
# ============================================================================

def build_scenario_full(evidence: dict) -> dict:
    """完整场景：12 区全部 16 证据 + 3 张图/区"""
    params = {}
    for d in ALL_DISTRICTS:
        params[d] = build_single_district(d, evidence, no_images=False)
    return params


def build_scenario_partial(evidence: dict) -> dict:
    """部分区有数据（样本不足）：3 区有观测，其余 9 区先验"""
    params = {}
    for d in ALL_DISTRICTS:
        if d in PARTIAL_DISTRICTS:
            params[d] = build_single_district(d, evidence, no_images=False)
        else:
            params[d] = build_empty_district()
    return params


def build_scenario_single(evidence: dict) -> dict:
    """仅 1 个区有完整数据（极端稀疏）：金水区有观测，其余 11 区先验"""
    params = {}
    for d in ALL_DISTRICTS:
        if d == SINGLE_DISTRICT:
            params[d] = build_single_district(d, evidence, no_images=False)
        else:
            params[d] = build_empty_district()
    return params


def build_scenario_params_only(evidence: dict) -> dict:
    """纯数学模型：12 区全部 16 证据，无图片"""
    params = {}
    for d in ALL_DISTRICTS:
        params[d] = build_single_district(d, evidence, no_images=True)
    return params


# ============================================================================
# 场景摘要打印
# ============================================================================

SCENARIO_META = {
    "full": {
        "label": "完整场景",
        "desc": "完整数据下的全图推演（12 区全部 16 证据 + 3 张图/区）",
    },
    "partial": {
        "label": "样本不足场景",
        "desc": "仅 3 区有观测（金水区/巩义市/中牟县），其余 9 区先验——展示观测稀疏时推理仍可用",
    },
    "single": {
        "label": "极端稀疏场景",
        "desc": "仅金水区有完整观测，其余 11 区先验——展示单点数据驱动的区域推演",
    },
    "params-only": {
        "label": "纯参数场景",
        "desc": "全部区仅数学参数（无图片），展示纯参数驱动的推理能力",
    },
}


def print_scenario_summary(scenario: str, params: dict):
    """打印场景摘要，方便演示时讲解"""
    meta = SCENARIO_META.get(scenario, {})
    print(f"\n{'='*60}")
    print(f"场景: {meta.get('label', scenario)}")
    print(f"说明: {meta.get('desc', '')}")
    print(f"{'='*60}")

    has_data_count = 0
    no_data_count = 0
    for d, cfg in params.items():
        ev = cfg.get("evidence", {})
        imgs = cfg.get("images", {})
        if ev:
            has_data_count += 1
            img_list = ", ".join(f"{k}" for k in imgs.keys())
            print(f"\n  📊 {d}: {len(ev)} 证据节点, {len(imgs)} 张图片 ({img_list})" if img_list
                  else f"\n  📊 {d}: {len(ev)} 证据节点, 无图片")
        else:
            no_data_count += 1
            print(f"\n  ⬜ {d}: 无证据（保持先验）")

    print(f"\n{'─'*40}")
    print(f"有数据: {has_data_count} 个区 | 先验: {no_data_count} 个区")
    print(f"总证据节点数: {sum(len(cfg.get('evidence', {})) for cfg in params.values())}")
    print(f"总图片数: {sum(len(cfg.get('images', {})) for cfg in params.values())}")
    print()


def cmd_list(params: dict):
    """列出当前各区配置"""
    if not params:
        print("❌ 未找到演示参数配置（demo_params.json 不存在或为空）")
        return

    print(f"\n{'='*60}")
    print(f"当前演示参数配置（共 {len(params)} 个区）")
    print(f"{'='*60}")

    for district, cfg in sorted(params.items()):
        ev = cfg.get("evidence", {})
        imgs = cfg.get("images", {})
        ev_count = len(ev)
        img_count = len(imgs)
        img_list = ", ".join(f"{k}={Path(v).name}" for k, v in imgs.items())
        print(f"\n  {district}:")
        print(f"    证据节点: {ev_count} 个")
        if img_list:
            print(f"    图片: {img_list}")
        else:
            print(f"    图片: 无")


def main():
    parser = argparse.ArgumentParser(
        description="一键生成演示参数包（configs/郑州/demo_params.json）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/prepare_demo_params.py --all
  python scripts/prepare_demo_params.py --scenario partial
  python scripts/prepare_demo_params.py --scenario single
  python scripts/prepare_demo_params.py --scenario params-only
  python scripts/prepare_demo_params.py --district 金水区
  python scripts/prepare_demo_params.py --list
        """,
    )

    # 场景选择（优先级高于 --district/--all）
    parser.add_argument("--scenario", type=str,
                        choices=["full", "partial", "single", "params-only"],
                        default=None,
                        help="演示场景: full(默认完整), partial(样本不足), "
                             "single(极端稀疏), params-only(纯参数)")

    # 目标区（与 --scenario 互斥，不直接互斥——scenario 优先）
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--district", type=str, default=None,
                        help="区名，如 金水区")
    target.add_argument("--all", action="store_true",
                        help="填充全部 12 区")

    # 图片控制（仅在不使用 --scenario 时有效）
    parser.add_argument("--no-images", action="store_true",
                        help="不包含图片路径")
    parser.add_argument("--wl", type=str, default=None,
                        help="水位尺图路径（覆盖默认）")
    parser.add_argument("--road", type=str, default=None,
                        help="道路桥梁图路径（覆盖默认）")
    parser.add_argument("--flood", type=str, default=None,
                        help="洪水现场图路径（覆盖默认）")

    # 查看模式
    parser.add_argument("--list", action="store_true",
                        help="列出当前各区已配置的参数")

    args = parser.parse_args()

    # ── --list 模式 ──
    if args.list:
        if OUTPUT_PATH.exists():
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                params = json.load(f)
        else:
            params = {}
        cmd_list(params)
        return

    # ── 加载证据 ──
    evidence = load_evidence()

    # ── 场景模式（优先于 --district/--all） ──
    if args.scenario:
        builders = {
            "full": build_scenario_full,
            "partial": build_scenario_partial,
            "single": build_scenario_single,
            "params-only": build_scenario_params_only,
        }
        builder = builders.get(args.scenario)
        if not builder:
            print(f"❌ 错误: 未知场景 '{args.scenario}'")
            sys.exit(1)

        params = builder(evidence)

        # 保存场景名到 JSON 中（供 dashboard 识别）
        meta = {"_scenario": args.scenario}

        # 写入 JSON
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        print(f"已写入: {OUTPUT_PATH}")

        # 打印场景摘要
        print_scenario_summary(args.scenario, params)
        print("✅ 完成。重启 dashboard 后自动加载演示参数。")
        return

    # ── 传统模式（--district / --all） ──
    if args.all:
        params = {}
        for d in ALL_DISTRICTS:
            params[d] = build_single_district(
                d, evidence,
                no_images=args.no_images,
                wl_path=args.wl, road_path=args.road, flood_path=args.flood,
            )
        print(f"\n已构建 {len(params)} 个区的演示参数包")
    elif args.district:
        params = {
            args.district: build_single_district(
                args.district, evidence,
                no_images=args.no_images,
                wl_path=args.wl, road_path=args.road, flood_path=args.flood,
            )
        }
        print(f"\n已构建 1 个区的演示参数包: {args.district}")
    else:
        parser.print_help()
        return

    # ── 写入 JSON ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"已写入: {OUTPUT_PATH}")

    # 打印摘要
    for d, cfg in params.items():
        ev = cfg.get("evidence", {})
        imgs = cfg.get("images", {})
        print(f"  {d}: {len(ev)} 证据节点, {len(imgs)} 张图片")
        for task, path in imgs.items():
            print(f"    {task}: {path}")

    print("\n✅ 完成。重启 dashboard 后自动加载演示参数。")


if __name__ == "__main__":
    main()