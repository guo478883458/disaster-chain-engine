"""
文本预处理管线：灾情文本信息抽取
读取 CrisisNLP-C 中文微博 xlsx 文件，抽取灾害类型、地点、时间、灾情关键词
先用规则+词典实现可运行版本，BERT 抽取作为 TODO 注释

数据路径（通过 path_config 统一管理）
"""

import os
import re
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ==================== 路径常量（通过 path_config 统一管理） ====================
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import CRISIS_DIR

CRISIS_FILES = [
    (os.path.join(CRISIS_DIR, "旧分类标准", "广州汕头暴雨.xlsx"), "暴雨"),
    (os.path.join(CRISIS_DIR, "旧分类标准", "台风山竹.xlsx"), "台风"),
    (os.path.join(CRISIS_DIR, "旧分类标准", "台风温比亚.xlsx"), "台风"),
    (os.path.join(CRISIS_DIR, "旧分类标准", "四川凉山森林大火.xls"), "森林火灾"),
    (os.path.join(CRISIS_DIR, "新分类标准", "江苏响水爆炸事故.xlsx"), "爆炸"),
    (os.path.join(CRISIS_DIR, "新分类标准", "四川凉山森林火灾.xls"), "森林火灾"),
]

# 文本列名（实际数据中文本在 weibo_text 列，text_info 为标签列）
TEXT_COLUMN = "weibo_text"

# ==================== 规则词典 ====================

# 灾害类型关键词词典
DISASTER_PATTERNS = {
    "内涝": [
        "内涝", "积水", "淹", "水浸", "水淹", "内涝", "涝", "淹水",
        "水漫", "雨水倒灌", "地下车库进水", "路面成河", "城市看海",
    ],
    "洪水": [
        "洪水", "洪灾", "洪涝", "泄洪", "防汛", "抗洪", "洪峰",
        "洪水泛滥", "河流漫堤", "堤坝", "溃堤", "水库泄洪",
    ],
    "滑坡": [
        "滑坡", "山体滑坡", "塌方", "泥石流", "崩塌", "地陷",
        "地面塌陷", "山体崩塌", "滑塌",
    ],
    "台风": [
        "台风", "飓风", "热带气旋", "热带风暴", "强风", "暴风",
    ],
    "暴雨": [
        "暴雨", "大暴雨", "特大暴雨", "强降雨", "持续降雨", "强降水",
        "大雨", "倾盆大雨", "雷雨", "雷暴",
    ],
    "地震": [
        "地震", "余震", "震感", "震中", "震源", "地动",
    ],
    "森林火灾": [
        "森林火灾", "山火", "林火", "火场", "火势", "大火",
        "火情", "森林大火", "火线", "扑火", "灭火",
    ],
    "爆炸": [
        "爆炸", "爆燃", "化学品泄漏", "化工厂爆炸", "燃爆",
    ],
}

# 中文省份/城市关键词（用于地点抽取）
LOCATION_KEYWORDS = [
    "北京", "上海", "广州", "深圳", "天津", "重庆",
    "广东", "山东", "江苏", "浙江", "福建", "河南", "河北",
    "湖南", "湖北", "四川", "云南", "贵州", "广西", "海南",
    "安徽", "江西", "山西", "陕西", "甘肃", "青海", "宁夏",
    "新疆", "西藏", "内蒙古", "辽宁", "吉林", "黑龙江",
    "寿光", "潍坊", "青州", "汕头", "广州", "深圳", "珠海",
    "佛山", "东莞", "中山", "江门", "湛江", "茂名", "肇庆",
    "惠州", "梅州", "汕尾", "河源", "阳江", "清远", "潮州",
    "揭阳", "云浮", "南京", "苏州", "无锡", "常州", "镇江",
    "扬州", "南通", "盐城", "徐州", "淮安", "连云港",
    "济南", "青岛", "烟台", "威海", "日照", "临沂",
    "成都", "绵阳", "德阳", "宜宾", "泸州", "乐山",
    "武汉", "黄石", "十堰", "宜昌", "襄阳", "荆州",
    "郑州", "洛阳", "开封", "安阳", "新乡", "南阳",
    "长沙", "株洲", "湘潭", "衡阳", "岳阳",
    "福州", "厦门", "泉州", "漳州", "莆田", "宁德",
    "木里", "凉山", "雅砻江", "响水", "陈家港",
]

# 灾情关键词
DISASTER_KEYWORDS = [
    "积水", "滑坡", "被困", "倒塌", "淹没", "受灾", "伤亡",
    "死亡", "受伤", "失踪", "紧急", "转移", "救援", "救灾",
    "抢险", "预警", "应急", "响应", "一级响应", "二级响应",
    "停运", "停课", "停电", "断水", "断路", "通讯中断",
    "损失", "冲毁", "漫堤", "决口", "溃坝", "管涌",
    "疏散", "安置", "安置点", "物资", "捐助", "捐款",
    "防疫", "消毒", "防疫", "消杀",
]

# 时间抽取正则
DATE_PATTERNS = [
    # 2025年7月20日
    r"(\d{4})年(\d{1,2})月(\d{1,2})日",
    # 7月20日
    r"(\d{1,2})月(\d{1,2})日",
    # 2025-07-20
    r"(\d{4})-(\d{1,2})-(\d{1,2})",
    # 2025/07/20
    r"(\d{4})/(\d{1,2})/(\d{1,2})",
    # 今天/昨天/前天
    r"(今天|昨天|前天)",
    # 近日/日前
    r"(近日|日前|日前)",
    # 7月20日前后
    r"(\d{1,2})月(\d{1,2})日前后",
]


def extract_disaster_type(text: str) -> Tuple[str, float]:
    """
    基于规则词典抽取灾害类型
    返回 (灾害类型, 置信度)
    """
    text_lower = text
    scores = {}
    for disaster_type, keywords in DISASTER_PATTERNS.items():
        score = 0
        for kw in keywords:
            count = text_lower.count(kw)
            score += count
        if score > 0:
            scores[disaster_type] = score

    if not scores:
        return ("其他", 0.3)

    # 按得分排序
    sorted_types = sorted(scores.items(), key=lambda x: -x[1])
    best_type, best_score = sorted_types[0]
    total_score = sum(scores.values())
    confidence = min(best_score / max(total_score, 1) * 0.6 + 0.4, 0.95)

    return (best_type, round(confidence, 4))


def extract_location(text: str) -> Tuple[Optional[str], float]:
    """
    基于规则词典抽取地点
    返回 (地点, 置信度)
    """
    found = []
    for loc in LOCATION_KEYWORDS:
        if loc in text:
            found.append(loc)

    if not found:
        return (None, 0.0)

    # 返回最具体的地点（最长匹配优先）
    found.sort(key=len, reverse=True)
    return (found[0], min(0.5 + 0.1 * len(found), 0.9))


def extract_time(text: str) -> Tuple[Optional[str], float]:
    """
    基于规则抽取时间信息
    返回 (时间字符串, 置信度)
    """
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups) == 3 and groups[0].isdigit():
                year, month, day = groups
                return (f"{year}年{month}月{day}日", 0.9)
            elif len(groups) == 2:
                month, day = groups
                return (f"{month}月{day}日", 0.7)
            elif groups[0] in ("今天", "昨天", "前天"):
                return (groups[0], 0.5)
            elif groups[0] in ("近日", "日前"):
                return (groups[0], 0.4)

    # 匹配月份关键词
    month_pattern = r"(\d{1,2})月"
    month_match = re.search(month_pattern, text)
    if month_match:
        return (f"{month_match.group(1)}月", 0.4)

    return (None, 0.0)


def extract_keywords(text: str) -> List[str]:
    """
    抽取灾情关键词
    """
    found = []
    text_lower = text
    for kw in DISASTER_KEYWORDS:
        if kw in text_lower:
            found.append(kw)
    return found


def process_text(text: str) -> Dict:
    """
    单条文本预处理主函数
    返回结构化抽取结果
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "原文": text,
            "灾害类型": "其他",
            "地点": None,
            "时间": None,
            "关键词": [],
            "置信度": 0.0,
        }

    disaster_type, type_conf = extract_disaster_type(text)
    location, loc_conf = extract_location(text)
    time_str, time_conf = extract_time(text)
    keywords = extract_keywords(text)

    # 综合置信度
    confs = [type_conf]
    if location:
        confs.append(loc_conf)
    if time_str:
        confs.append(time_conf)
    overall_conf = round(sum(confs) / len(confs), 4)

    return {
        "原文": text,
        "灾害类型": disaster_type,
        "地点": location,
        "时间": time_str,
        "关键词": keywords,
        "置信度": overall_conf,
    }


def load_crisis_data() -> pd.DataFrame:
    """
    加载 CrisisNLP-C 所有数据文件
    返回合并后的 DataFrame
    """
    all_dfs = []
    for filepath, event_type in CRISIS_FILES:
        try:
            # 先用 openpyxl 尝试（兼容 .xls 实为 .xlsx 的情况）
            try:
                df = pd.read_excel(filepath, engine="openpyxl")
            except Exception:
                # 回退到 xlrd
                df = pd.read_excel(filepath, engine="xlrd")
            df["事件类型"] = event_type
            df["源文件"] = os.path.basename(filepath)
            all_dfs.append(df)
            print(f"  加载 {os.path.basename(filepath)}: {df.shape[0]} 行")
        except Exception as e:
            print(f"  [警告] 加载失败 {os.path.basename(filepath)}: {e}")

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        return merged
    return pd.DataFrame()


def process_dataframe(df: pd.DataFrame, text_column: str = TEXT_COLUMN) -> pd.DataFrame:
    """
    对整个 DataFrame 进行文本预处理
    """
    results = []
    for _, row in df.iterrows():
        text = row.get(text_column, "")
        result = process_text(str(text) if pd.notna(text) else "")
        result["数据标签"] = row.get("text_info", "")
        result["数据分类"] = row.get("text_human", "")
        results.append(result)

    result_df = pd.DataFrame(results)
    return result_df


def batch_process(output_path: Optional[str] = None) -> pd.DataFrame:
    """
    批量处理所有 CrisisNLP-C 数据
    输出 CSV/JSON
    """
    print("加载 CrisisNLP-C 数据...")
    df = load_crisis_data()
    print(f"总计加载 {df.shape[0]} 条微博")

    print("预处理文本...")
    result_df = process_dataframe(df)

    # 统计信息
    print("\n--- 灾害类型分布 ---")
    type_counts = result_df["灾害类型"].value_counts()
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    print(f"\n--- 地点抽取统计 ---")
    loc_count = result_df["地点"].notna().sum()
    print(f"  成功抽取地点: {loc_count}/{len(result_df)}")

    print(f"\n--- 时间抽取统计 ---")
    time_count = result_df["时间"].notna().sum()
    print(f"  成功抽取时间: {time_count}/{len(result_df)}")

    if output_path:
        # 保存 CSV
        csv_path = output_path
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存到: {csv_path}")

        # 保存 JSON
        json_path = os.path.splitext(csv_path)[0] + ".json"
        result_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"结果已保存到: {json_path}")

    return result_df


# ==================== CLI ====================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="灾害链文本预处理管线")
    parser.add_argument("--mode", choices=["single", "batch"], default="single", help="运行模式")
    parser.add_argument("--text", type=str, help="单条文本（mode=single）")
    parser.add_argument("--output", type=str, default=None, help="输出路径（mode=batch）")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.text:
            # 交互式模式
            print("输入文本（输入空行退出）：")
            while True:
                text = input("> ").strip()
                if not text:
                    break
                result = process_text(text)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                print()
        else:
            result = process_text(args.text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "batch":
        batch_process(args.output)


if __name__ == "__main__":
    main()