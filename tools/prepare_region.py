"""
通用区域数据预处理脚本
======================
功能：输入原始数据 + 离散化规则 → 输出离散化样本（CSV）
不绑定特定地区，通过参数指定数据目录和规则文件。

用法：
  python tools/prepare_region.py --data_dir data/北京 --rules rules.json --output data/北京/samples.csv

离散化规则（rules.json）示例：
  {
    "降水强度": {"method": "quantile", "bins": [0.33, 0.66], "labels": ["低", "中", "高"]},
    "降水时长": {"method": "threshold", "value": 6, "labels": ["短", "长"]},
    "海拔": {"method": "quantile", "bins": [0.5], "labels": ["低", "高"]},
    "坡度": {"method": "threshold", "value": 15, "labels": ["缓", "陡"]}
  }

支持方法：
  - quantile: 分位数划分（基于数据分布）
  - threshold: 阈值划分（基于固定阈值）
  - category: 直接映射（字段已是分类值）
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def discretize_column(series: pd.Series, rule: dict) -> pd.Series:
    """
    对单列执行离散化。

    Args:
        series: 原始数据列
        rule: 离散化规则，包含 method, bins/labels 等

    Returns:
        pd.Series: 离散化后的分类值
    """
    method = rule.get("method", "quantile")
    labels = rule.get("labels", [])

    if method == "quantile":
        # 分位数离散化
        bins = rule.get("bins", [0.33, 0.66])
        # 计算实际分位数
        quantiles = series.quantile(bins).tolist()
        # 添加边界
        bin_edges = [-np.inf] + quantiles + [np.inf]
        if len(bin_edges) - 1 != len(labels):
            print(f"  警告: 分位数边界数({len(bin_edges)-1})与标签数({len(labels)})不匹配，自动调整")
            labels = [f"等级{i+1}" for i in range(len(bin_edges) - 1)]
        return pd.cut(series, bins=bin_edges, labels=labels, right=True)

    elif method == "threshold":
        # 阈值离散化
        threshold = rule.get("value", 0)
        if threshold is None:
            raise ValueError("threshold 方法需要指定 value 参数")
        # 支持多阈值
        values = rule.get("values", [threshold])
        if len(values) + 1 != len(labels):
            print(f"  警告: 阈值数({len(values)})与标签数({len(labels)})不匹配")
            labels = [f"等级{i+1}" for i in range(len(values) + 1)]
        bin_edges = [-np.inf] + values + [np.inf]
        return pd.cut(series, bins=bin_edges, labels=labels, right=True)

    elif method == "category":
        # 已是分类值，直接返回
        return series.astype(str)

    else:
        raise ValueError(f"不支持的离散化方法: {method}")


def prepare_region_data(
    data_dir: str,
    rules_file: str,
    output_file: str = None,
    data_file: str = None,
    id_column: str = None,
):
    """
    通用区域数据预处理。

    Args:
        data_dir: 数据目录
        rules_file: 离散化规则 JSON 文件路径
        output_file: 输出 CSV 路径（默认 data_dir/samples.csv）
        data_file: 输入数据文件名（默认自动查找 data_dir 下的 CSV）
        id_column: 区域标识列名（如 "district"），保留在输出中
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)

    # 1. 加载规则
    rules_path = Path(rules_file)
    if not rules_path.exists():
        print(f"错误: 规则文件不存在: {rules_path}")
        sys.exit(1)
    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    print(f"加载离散化规则: {len(rules)} 个字段")

    # 2. 加载数据
    if data_file:
        data_path = data_dir / data_file
    else:
        # 自动查找第一个 CSV
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            print(f"错误: {data_dir} 下未找到 CSV 文件")
            sys.exit(1)
        data_path = csv_files[0]
        print(f"自动选择数据文件: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    print(f"加载数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"列名: {list(df.columns)}")

    # 3. 执行离散化
    result_cols = {}
    if id_column and id_column in df.columns:
        result_cols[id_column] = df[id_column].astype(str)

    discretized_count = 0
    for col_name, rule in rules.items():
        if col_name not in df.columns:
            print(f"  警告: 字段 '{col_name}' 不在数据中，跳过")
            continue
        try:
            result_cols[col_name] = discretize_column(df[col_name], rule)
            discretized_count += 1
            labels = rule.get("labels", [])
            print(f"  ✓ {col_name}: {len(result_cols[col_name].unique())} 个类别 {labels}")
        except Exception as e:
            print(f"  ✗ {col_name}: 离散化失败 - {e}")

    # 4. 输出
    result_df = pd.DataFrame(result_cols)
    if output_file:
        out_path = Path(output_file)
    else:
        out_path = data_dir / "samples.csv"

    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n离散化完成: {discretized_count}/{len(rules)} 个字段")
    print(f"输出: {out_path} ({len(result_df)} 行)")
    print(f"列: {list(result_df.columns)}")

    # 5. 统计分布
    print("\n各字段分布统计:")
    for col in result_df.columns:
        if col == id_column:
            continue
        counts = result_df[col].value_counts()
        total = len(result_df)
        dist_str = "  |  ".join(
            f"{k}: {v} ({v/total*100:.1f}%)"
            for k, v in counts.items()
        )
        print(f"  {col}: {dist_str}")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="通用区域数据预处理脚本")
    parser.add_argument("--data_dir", required=True, help="数据目录路径")
    parser.add_argument("--rules", required=True, help="离散化规则 JSON 文件路径")
    parser.add_argument("--output", default=None, help="输出 CSV 路径（默认 data_dir/samples.csv）")
    parser.add_argument("--data_file", default=None, help="输入数据文件名（默认自动查找）")
    parser.add_argument("--id_column", default=None, help="区域标识列名（如 district）")
    args = parser.parse_args()

    prepare_region_data(
        data_dir=args.data_dir,
        rules_file=args.rules,
        output_file=args.output,
        data_file=args.data_file,
        id_column=args.id_column,
    )


if __name__ == "__main__":
    main()