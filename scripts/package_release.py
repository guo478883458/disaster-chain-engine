"""
演示发布包打包脚本
==================
输出 H:\实习\发布包\灾害链演示_v1.zip（约 200MB，zip64）。

内容:
  v2_engine/         - 灾害链推理引擎代码
  infra_recognition/ - 基础设施灾损识别代码
  data/              - 模型权重 + 必要数据 + 演示图片
  演示材料/           - 演示文档
  README.md / requirements.txt / 启动Dashboard.bat / path_config.py

用法:
  python scripts/package_release.py
"""

import os
import zipfile
import sys
import time
import shutil
from pathlib import Path

# ── 项目根目录 ──
V2_ROOT = Path(__file__).resolve().parent.parent
INFRA_ROOT = Path(r"H:\实习\基础设施灾损识别")
DEMO_DIR = Path(r"H:\实习\综合演示")
DATA_ROOT = Path(r"H:\dev\disaster-data")

# ── 输出 ──
OUTPUT_DIR = Path(r"H:\实习\发布包")
OUTPUT_ZIP = OUTPUT_DIR / "灾害链演示_v1.zip"

# ── 排除模式 ──
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".idea",
    "output", "temp_uploads",
    "logs", "runs",
    "yolo_seg_labels", "yolo_labels",
    "yolo_floodnet", "yolo_rescuenet", "yolo_landslide", "yolo_floodimg",
    "node_modules", ".mypy_cache", ".pytest_cache",
}
EXCLUDE_FILES = {
    "yolov8n-seg.pt", "yolov8n.pt", "yolo26n.pt",
    "yolov8n-seg.engine",
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".log", ".tmp", ".lock", ".gitignore"}

# ── 需要包含的权重文件 ──
WEIGHTS = [
    ("models/flood.pt", "data/models/flood.pt"),
    ("models/flood_rescuenet.pt", "data/models/flood_rescuenet.pt"),
    ("models/floodimg_detect.pt", "data/models/floodimg_detect.pt"),
    ("models/landslide.pt", "data/models/landslide.pt"),
    ("models/infra/water_level/water_level_digit_detector.pt", "data/models/infra/water_level/water_level_digit_detector.pt"),
    ("models/infra/road_bridge/rdd2020_yolov8n.pt", "data/models/infra/road_bridge/rdd2020_yolov8n.pt"),
]

# ── 需要包含的数据文件 ──
DATA_FILES = [
    # 地理数据
    ("zhengzhou_720/zhengzhou_geojson.json", "data/zhengzhou_720/zhengzhou_geojson.json"),
    ("zhengzhou_720/Grid500_AllCity_Edges.csv", "data/zhengzhou_720/Grid500_AllCity_Edges.csv"),
    ("zhengzhou_720/Grid500_AllCity_Nodes.csv", "data/zhengzhou_720/Grid500_AllCity_Nodes.csv"),
    ("zhengzhou_720/Grid500_Selected_FloodArea.csv", "data/zhengzhou_720/Grid500_Selected_FloodArea.csv"),
    # 水位真值
    ("infra_datasets/water_level/extracted/SAM_water_level_Dataset/In-situ water levels/In-situ & simulated water levels.xlsx",
     "data/infra_datasets/water_level/In-situ & simulated water levels.xlsx"),
]

# ── 演示图片 ──
DEMO_IMAGES = [
    "1-洪水现场.jpg",
    "2-河道水位尺.jpg",
    "3-道路路面.jpg",
]

# ── 演示材料 ──
DEMO_MATERIALS = [
    "0-演示讲解文档.md",
    "演示参数清单.md",
    "run_demo.py",
]

# ── V2 代码目录（需要递归包含的顶层目录） ──
V2_CODE_DIRS = ["tools", "scripts", "configs"]
V2_CODE_FILES = [
    "dashboard.py", "bn_engine.py", "region_engine.py",
    "run.py", "demo_infer.py", "demo_interactive.py", "demo_real.py",
    "test_40nodes.py", "test_all.py", "test_validate.py",
]

# ── 基础设施代码目录 ──
INFRA_CODE_DIRS = ["models", "tools"]
INFRA_CODE_FILES = ["path_config.py",]


def should_exclude(rel_path: str, is_dir: bool = False) -> bool:
    """判断是否应排除"""
    parts = Path(rel_path).parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
    if not is_dir:
        fname = Path(rel_path).name
        if fname in EXCLUDE_FILES:
            return True
        ext = Path(rel_path).suffix.lower()
        if ext in EXCLUDE_EXTS:
            return True
    return False


def walk_and_add(zf: zipfile.ZipFile, src_dir: Path, arc_prefix: str,
                 file_filter=None, dir_filter=None):
    """递归扫描目录并添加到 zip"""
    if not src_dir.exists():
        print(f"  ⚠ 目录不存在，跳过: {src_dir}")
        return
    for root, dirs, files in os.walk(src_dir):
        # 过滤目录
        dirs[:] = [d for d in dirs if not should_exclude(d, is_dir=True)
                   and (dir_filter is None or dir_filter(d))]

        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, src_dir)
            if should_exclude(rel):
                continue
            if file_filter and not file_filter(fname):
                continue
            arcname = f"{arc_prefix}/{rel}".replace("\\", "/")
            zf.write(fpath, arcname)


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def main():
    print(f"{'='*60}")
    print(f"演示发布包打包脚本")
    print(f"输出: {OUTPUT_ZIP}")
    print(f"{'='*60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 检查权重文件
    print("\n检查权重文件...")
    missing_weights = []
    for src_rel, _ in WEIGHTS:
        src = DATA_ROOT / src_rel
        if not src.exists():
            missing_weights.append(str(src_rel))
    if missing_weights:
        print(f"  ⚠ 以下权重文件缺失，将跳过:")
        for w in missing_weights:
            print(f"    - {w}")
    else:
        print("  ✅ 全部权重文件存在")

    # 检查演示图片
    print("\n检查演示图片...")
    for img in DEMO_IMAGES:
        p = DEMO_DIR / "1-输入图片" / img
        if not p.exists():
            print(f"  ⚠ 演示图片缺失: {p}")
        else:
            print(f"  ✅ {img}")

    # 检查演示材料
    for f in DEMO_MATERIALS:
        p = DEMO_DIR / f
        if not p.exists():
            print(f"  ⚠ 演示材料缺失: {p}")

    start_time = time.time()
    file_count = 0
    total_bytes = 0

    print(f"\n开始打包...")
    print()

    try:
        with zipfile.ZipFile(
            str(OUTPUT_ZIP),
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as zf:
            last_log = time.time()

            # 1. 根目录文件
            for fname in ["README.md", "requirements.txt", "启动Dashboard.bat", "path_config.py"]:
                fpath = V2_ROOT / fname
                if fpath.exists():
                    zf.write(fpath, fname)
                    file_count += 1
                    total_bytes += fpath.stat().st_size

            # 2. V2 代码
            print("  [v2_engine] 添加代码...")
            for d in V2_CODE_DIRS:
                walk_and_add(zf, V2_ROOT / d, f"v2_engine/{d}")
            for fname in V2_CODE_FILES:
                fpath = V2_ROOT / fname
                if fpath.exists():
                    zf.write(fpath, f"v2_engine/{fname}")
                    file_count += 1
                    total_bytes += fpath.stat().st_size
            # 配置文件
            for f in ["v2-40节点扩展验证报告.md", "数据验证实验报告.md"]:
                fpath = V2_ROOT / f
                if fpath.exists():
                    zf.write(fpath, f"v2_engine/{f}")
                    file_count += 1

            # 3. 基础设施代码
            print("  [infra_recognition] 添加代码...")
            for d in INFRA_CODE_DIRS:
                walk_and_add(zf, INFRA_ROOT / d, f"infra_recognition/{d}")
            for fname in INFRA_CODE_FILES:
                fpath = INFRA_ROOT / fname
                if fpath.exists():
                    zf.write(fpath, f"infra_recognition/{fname}")
                    file_count += 1

            # 4. 模型权重
            print("  [data/models] 添加权重...")
            for src_rel, arcname in WEIGHTS:
                src = DATA_ROOT / src_rel
                if src.exists():
                    zf.write(src, arcname)
                    file_count += 1
                    total_bytes += src.stat().st_size
                    if time.time() - last_log >= 2:
                        print(f"    {arcname} ({format_size(src.stat().st_size)})")
                        last_log = time.time()

            # 5. 数据文件
            print("  [data] 添加数据文件...")
            for src_rel, arcname in DATA_FILES:
                src = DATA_ROOT / src_rel
                if src.exists():
                    zf.write(src, arcname)
                    file_count += 1
                    total_bytes += src.stat().st_size

            # 6. 演示图片
            print("  [data/demo_images] 添加演示图片...")
            for img in DEMO_IMAGES:
                src = DEMO_DIR / "1-输入图片" / img
                if src.exists():
                    zf.write(src, f"data/demo_images/{img}")
                    file_count += 1
                    total_bytes += src.stat().st_size

            # 7. 演示材料
            print("  [演示材料] 添加文档...")
            for fname in DEMO_MATERIALS:
                src = DEMO_DIR / fname
                if src.exists():
                    zf.write(src, f"演示材料/{fname}")
                    file_count += 1
                    total_bytes += src.stat().st_size

    except Exception as e:
        print(f"\n❌ 打包失败: {e}")
        # 清理不完整的 zip
        if OUTPUT_ZIP.exists():
            OUTPUT_ZIP.unlink()
        sys.exit(1)

    elapsed = time.time() - start_time
    zip_size = OUTPUT_ZIP.stat().st_size if OUTPUT_ZIP.exists() else 0

    print(f"\n{'='*60}")
    print(f"✅ 打包完成!")
    print(f"  文件数: {file_count}")
    print(f"  输出大小: {format_size(zip_size)}")
    print(f"  原始大小: {format_size(total_bytes)}")
    print(f"  压缩率: {zip_size/total_bytes*100:.1f}%" if total_bytes > 0 else "")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  输出: {OUTPUT_ZIP}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()