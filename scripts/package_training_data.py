"""
训练数据分卷打包脚本
====================
将训练数据集按文件遍历打包为多个独立 zip 分卷（每卷约 2.9GB），
每卷可独立解压，接收方把所有卷解压到同一根目录即还原完整结构。

输出: H:\实习\发布包\训练数据_001.zip, 训练数据_002.zip, ...

保留内容：
  - image_datasets/ 全部原始数据（训练用）
  - infra_datasets/ 全部原始数据（训练用）
  - models/ 全部最终权重

排除内容（转换产物/中间文件，可重新生成）：
  - image_datasets/rescuenet/yolo_seg_labels（15G）
  - models/yolo_floodnet（8.3G）、models/yolo_rescuenet（5.4G）
  - models/yolo_landslide（1.4G）、models/yolo_floodimg
  - models/*_train/、runs/ 目录

用法:
  python scripts/package_training_data.py
  python scripts/package_training_data.py --vol-size-gb 3.5   # 自定义卷大小
"""

import os
import zipfile
import sys
import time
import argparse
from pathlib import Path

# ── 数据根目录 ──
DATA_ROOT = Path(r"H:\dev\disaster-data")

# ── 输出路径 ──
OUTPUT_DIR = Path(r"H:\实习\发布包")

# ── 默认每卷大小（GB） ──
DEFAULT_VOL_SIZE_GB = 2.9

# ── 排除模式（目录名匹配） ──
EXCLUDE_DIRS = {
    # 转换产物（可重新生成）
    "yolo_seg_labels",
    "yolo_floodnet",
    "yolo_rescuenet",
    "yolo_landslide",
    "yolo_floodimg",
    "yolo_labels",       # 基础设施转换产物
    # 训练中间文件
    "runs",
    "__pycache__",
    ".git",
}

# ── 排除后缀 ──
EXCLUDE_EXTS = {
    ".pyc", ".pyo",
    ".log", ".tmp",
    ".lock",
}

# ── 包含的顶层目录 ──
INCLUDE_TOPS = [
    "image_datasets",
    "infra_datasets",
    "models",
]


def should_exclude(rel_path: str) -> bool:
    """判断是否应排除该文件/目录"""
    parts = Path(rel_path).parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
        # 排除 *_train 目录（如 rescuenet_train、floodnet_train 等转换产物）
        if part.endswith("_train"):
            return True
    ext = Path(rel_path).suffix.lower()
    if ext in EXCLUDE_EXTS:
        return True
    return False


def collect_files():
    """收集所有需要打包的文件，返回 [(fpath, rel_path, size), ...]"""
    file_list = []
    total_size = 0

    for top in INCLUDE_TOPS:
        top_dir = DATA_ROOT / top
        if not top_dir.exists():
            print(f"  ⚠ 目录不存在，跳过: {top_dir}")
            continue

        print(f"正在扫描 {top}/...")
        for root, dirs, files in os.walk(top_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fname in files:
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, DATA_ROOT)
                if should_exclude(rel_path):
                    continue
                fsize = os.path.getsize(fpath)
                file_list.append((fpath, rel_path, fsize))
                total_size += fsize

    return file_list, total_size


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def build_volume_plan(file_list, vol_size_bytes):
    """
    将文件列表分配到各卷。
    返回 [(vol_index, vol_files, vol_size), ...]
    vol_files: [(fpath, rel_path, size), ...]
    文件按大小降序排列以保证大文件优先分配，避免单文件超卷。
    """
    # 按大小降序排列
    sorted_files = sorted(file_list, key=lambda x: -x[2])

    # 用 first-fit 贪心算法分配到各卷
    volumes = []  # [(vol_idx, files, total_size)]

    for fpath, rel_path, fsize in sorted_files:
        placed = False
        for vol_idx, vol_files, vol_size in volumes:
            if vol_size + fsize <= vol_size_bytes:
                vol_files.append((fpath, rel_path, fsize))
                volumes[vol_idx] = (vol_idx, vol_files, vol_size + fsize)
                placed = True
                break
        if not placed:
            # 新开一卷
            volumes.append((len(volumes), [(fpath, rel_path, fsize)], fsize))

    # 转成输出格式，按原顺序重排每卷内文件（保持可读性）
    # 先把原顺序的索引建好
    orig_order = {rel: (fpath, size) for fpath, rel, size in file_list}

    result = []
    for vol_idx, vol_files, vol_size in volumes:
        # 按原顺序排列
        vol_files_sorted = sorted(vol_files, key=lambda x: [
            next(i for i, (_, r, _) in enumerate(file_list) if r == x[1])
        ])
        result.append((vol_idx + 1, vol_files_sorted, vol_size))

    return result


def main():
    parser = argparse.ArgumentParser(description="训练数据分卷打包脚本")
    parser.add_argument("--vol-size-gb", type=float, default=DEFAULT_VOL_SIZE_GB,
                        help=f"每卷目标大小（GB），默认 {DEFAULT_VOL_SIZE_GB}")
    args = parser.parse_args()

    vol_size_bytes = int(args.vol_size_gb * 1024 * 1024 * 1024)

    print(f"{'='*60}")
    print(f"训练数据分卷打包脚本")
    print(f"数据根目录: {DATA_ROOT}")
    print(f"输出目录:   {OUTPUT_DIR}")
    print(f"每卷目标:   {args.vol_size_gb} GB")
    print(f"{'='*60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 扫描文件 ──
    print("\n正在扫描文件...")
    scan_start = time.time()
    file_list, total_size = collect_files()
    scan_time = time.time() - scan_start
    print(f"\n扫描完成: {len(file_list)} 个文件, 总计 {format_size(total_size)}")
    print(f"扫描耗时: {scan_time:.1f} 秒")

    if not file_list:
        print("\n❌ 没有找到需要打包的文件")
        sys.exit(1)

    # ── 分卷规划 ──
    print(f"\n正在规划分卷（每卷 ≤ {format_size(vol_size_bytes)}）...")
    plan_start = time.time()
    volumes = build_volume_plan(file_list, vol_size_bytes)
    plan_time = time.time() - plan_start
    print(f"分卷规划完成: {len(volumes)} 卷, 耗时 {plan_time:.1f} 秒")

    for vol_idx, vol_files, vol_size in volumes:
        print(f"  第 {vol_idx:02d} 卷: {len(vol_files)} 个文件, {format_size(vol_size)}")

    # ── 确认 ──
    print(f"\n{'─'*60}")
    print(f"即将开始打包 {len(volumes)} 个分卷，预计总耗时 15~30 分钟")
    print(f"按 Ctrl+C 取消，或等待 3 秒后自动开始...")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(0)

    # ── 执行打包 ──
    total_start = time.time()
    manifest = []  # 分卷清单
    accumulated_bytes = 0  # 累计压缩前大小

    for vol_idx, vol_files, vol_size in volumes:
        vol_name = f"训练数据_{vol_idx:03d}.zip"
        vol_path = OUTPUT_DIR / vol_name
        vol_start = time.time()
        last_log = time.time()
        written = 0
        vol_uncompressed = 0

        # 删除旧卷（如果存在）
        if vol_path.exists():
            vol_path.unlink()

        print(f"\n--- 第 {vol_idx}/{len(volumes)} 卷: {vol_name} ---")

        try:
            with zipfile.ZipFile(
                str(vol_path),
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as zf:
                for fpath, rel_path, fsize in vol_files:
                    zf.write(fpath, arcname=str(rel_path))
                    written += 1
                    vol_uncompressed += fsize

                    now = time.time()
                    if now - last_log >= 5.0:
                        pct = written / len(vol_files) * 100
                        print(f"  [{written}/{len(vol_files)}] {pct:.0f}% "
                              f"({format_size(vol_uncompressed)})", end="\r")
                        last_log = now

        except Exception as e:
            print(f"\n❌ 第 {vol_idx} 卷打包失败: {e}")
            if vol_path.exists():
                vol_path.unlink()
            sys.exit(1)

        vol_elapsed = time.time() - vol_start
        vol_compressed = vol_path.stat().st_size
        accumulated_bytes += vol_uncompressed

        print(f"\n  ✅ 第 {vol_idx} 卷完成: {format_size(vol_compressed)} "
              f"(压缩后) / {format_size(vol_uncompressed)} (原始), "
              f"耗时 {vol_elapsed:.1f}s")

        manifest.append({
            "序号": vol_idx,
            "文件名": vol_name,
            "文件数": written,
            "压缩后大小": vol_compressed,
            "原始大小": vol_uncompressed,
            "耗时_秒": round(vol_elapsed, 1),
        })

    total_time = time.time() - total_start

    # ── 生成分卷清单 ──
    manifest_path = OUTPUT_DIR / "分卷清单.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("训练数据分卷清单\n")
        f.write(f"{'='*60}\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据根目录: {DATA_ROOT}\n")
        f.write(f"每卷目标: {args.vol_size_gb} GB\n")
        f.write(f"总文件数: {len(file_list)}\n")
        f.write(f"总原始大小: {format_size(total_size)}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"{'卷号':<6} {'文件名':<30} {'文件数':<8} {'压缩后大小':<12} {'原始大小':<12} {'耗时':<8}\n")
        f.write("-" * 76 + "\n")
        for m in manifest:
            f.write(f"{m['序号']:<6} {m['文件名']:<30} {m['文件数']:<8} "
                    f"{format_size(m['压缩后大小']):<12} {format_size(m['原始大小']):<12} "
                    f"{m['耗时_秒']}s\n")
        f.write("\n")
        f.write(f"总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)\n")

    print(f"\n{'='*60}")
    print(f"✅ 全部打包完成!")
    print(f"  总卷数: {len(volumes)}")
    print(f"  总文件数: {len(file_list)}")
    print(f"  总原始大小: {format_size(total_size)}")
    print(f"  总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  分卷清单: {manifest_path}")
    print(f"{'='*60}")

    # 打印分卷总览
    print(f"\n分卷总览:")
    for m in manifest:
        print(f"  {m['文件名']:30s}  {format_size(m['压缩后大小']):>10s}  "
              f"{m['文件数']:>6d} 个文件")
    print(f"  {'分卷清单.txt':30s}  (已生成)")


if __name__ == "__main__":
    main()