"""
图像预处理管线：内涝分割 + 滑坡分割
使用 YOLOv8n-seg 轻量模型，支持训练和推理
输出结构化字段：积水面积(m²)、淹没占比、滑坡面积(m²)、灾情等级

数据路径（通过 path_config 统一管理）：
  FloodNet, CASlandslides, RescueNet, FloodIMG, 模型权重
"""

import os
import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ==================== 路径常量（通过 path_config 统一管理） ====================
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import (
    DATA_ROOT, MODELS_DIR,
    FLOODNET_DIR, FLOODNET_TEST_COLORMASK,
    CASL_DIR, RESCUENET_DIR, FLOODIMG_DIR,
    FLOOD_PT, FLOODIMG_DETECT_PT, LANDSLIDE_PT,
)

# FloodNet
FLOODNET_TRAIN_IMG = os.path.join(FLOODNET_DIR, "train", "train-org-img")
FLOODNET_TRAIN_LABEL = os.path.join(FLOODNET_DIR, "train", "train-label-img")
FLOODNET_VAL_IMG = os.path.join(FLOODNET_DIR, "val", "val-org-img")
FLOODNET_VAL_LABEL = os.path.join(FLOODNET_DIR, "val", "val-label-img")
FLOODNET_TEST_IMG = os.path.join(FLOODNET_DIR, "test", "test-org-img")

# CASlandslides
CASL_IMG = os.path.join(CASL_DIR, "img")
CASL_LABEL = os.path.join(CASL_DIR, "label")
CASL_MASK = os.path.join(CASL_DIR, "mask")

# RescueNet
RESCUENET_IMG = os.path.join(RESCUENET_DIR, "train-org-img")
RESCUENET_LABEL = os.path.join(RESCUENET_DIR, "train-label-img")
RESCUENET_FLOOD_CLASS = {1}  # 水体类

# FloodIMG
FLOODIMG_ANN_DIR = os.path.join(FLOODIMG_DIR, "Annotation")
FLOODIMG_IMG_DIR = os.path.join(FLOODIMG_DIR, "Flood Images")

# 模型权重目录
MODEL_DIR = MODELS_DIR
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- 配置参数 --------------------
# FloodNet 中内涝相关类别（6=淹没道路, 7=淹没建筑）
FLOOD_CLASSES = {6, 7}
# 像素分辨率换算（假设 UAV 图像 0.05m/px，即 1px = 0.0025 m²）
# FloodNet 原始尺寸 4000x3000，对应实际约 200m x 150m 范围
PIXEL_RESOLUTION_M = 0.05  # 每像素对应实际米数
PIXEL_AREA_M2 = PIXEL_RESOLUTION_M ** 2  # 0.0025 m²/px

# 灾情等级阈值（积水面积 m²）
DISASTER_LEVELS = [
    (0, 100, "无"),        # 无积水或极少
    (100, 500, "轻度"),     # 小范围积水
    (500, 2000, "中度"),    # 中等范围积水
    (2000, 5000, "重度"),   # 大范围积水
    (5000, float("inf"), "严重"),  # 严重内涝
]

# 滑坡面积等级阈值
LANDSLIDE_LEVELS = [
    (0, 50, "无"),
    (50, 200, "轻度"),
    (200, 1000, "中度"),
    (1000, 5000, "重度"),
    (5000, float("inf"), "严重"),
]


def get_disaster_level(area_m2: float, is_landslide: bool = False) -> str:
    """根据面积分档返回灾情等级"""
    levels = LANDSLIDE_LEVELS if is_landslide else DISASTER_LEVELS
    for low, high, label in levels:
        if low <= area_m2 < high:
            return label
    return "未知"


def floodnet_mask_to_binary(mask_path: str) -> np.ndarray:
    """
    将 FloodNet 标签图（单通道灰度）转为二值内涝掩码
    灰度值 6=淹没道路, 7=淹没建筑 → 1，其余 → 0
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取标签文件: {mask_path}")
    binary = np.zeros_like(mask, dtype=np.uint8)
    for cls in FLOOD_CLASSES:
        binary[mask == cls] = 1
    return binary


def floodnet_colormask_to_binary(mask_path: str) -> np.ndarray:
    """
    将 FloodNet ColorMask（RGB 彩色）转为二值内涝掩码
    使用已知颜色映射判断淹没像素
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is None:
        raise FileNotFoundError(f"无法读取彩色标签文件: {mask_path}")
    # 转为灰度后判断：淹没区域（道路/建筑受淹）通常在彩色掩码中呈现特定色
    # 直接用灰度标签的映射：离线统计显示淹没像素在 ColorMask 中对应特定 RGB 值
    # 简化：对彩色掩码进行聚类判断
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # 淹没区域在 ColorMask 中通常为亮色系（黄/橙/红）
    # 根据数据集定义：淹没道路=黄色系，淹没建筑=橙色系
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    # 定义淹没色范围（H 通道：黄 20-40, 橙 10-20, 红 0-10）
    lower_flood1 = np.array([0, 50, 50])
    upper_flood1 = np.array([40, 255, 255])
    flood_mask = cv2.inRange(hsv, lower_flood1, upper_flood1)
    return (flood_mask > 0).astype(np.uint8)


def caslandslides_mask_to_binary(mask_path: str) -> np.ndarray:
    """
    将 CASlandslides mask 转为二值滑坡掩码
    mask 中 1=滑坡区域
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取掩码文件: {mask_path}")
    return (mask > 0).astype(np.uint8)


def mask_to_yolo_polygons(binary_mask: np.ndarray, img_width: int, img_height: int,
                          class_id: int = 0, min_area: int = 50) -> List[str]:
    """
    将二值掩码转为 YOLO 分割格式的多边形标注
    返回: ["class_id x1 y1 x2 y2 ...", ...]
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        # 简化轮廓以减少点数
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        # 归一化坐标
        pts = []
        for p in approx:
            x, y = p[0]
            pts.append(f"{x / img_width:.6f} {y / img_height:.6f}")
        polygons.append(f"{class_id} " + " ".join(pts))
    return polygons


def prepare_floodnet_yolo_dataset(output_dir: str, split: str = "train"):
    """
    将 FloodNet 语义分割标签转为 YOLO 格式
    output_dir: YOLO 数据集输出目录
    split: "train" 或 "val"
    """
    if split == "train":
        img_dir = FLOODNET_TRAIN_IMG
        label_dir = FLOODNET_TRAIN_LABEL
    else:
        img_dir = FLOODNET_VAL_IMG
        label_dir = FLOODNET_VAL_LABEL

    out_img_dir = os.path.join(output_dir, "images", split)
    out_label_dir = os.path.join(output_dir, "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    converted = 0
    skipped = 0
    for fname in img_files:
        # 对应标签文件名
        base = os.path.splitext(fname)[0]
        label_fname = f"{base}_lab.png"
        label_path = os.path.join(label_dir, label_fname)
        if not os.path.exists(label_path):
            continue

        out_img = os.path.join(out_img_dir, fname)
        txt_path = os.path.join(out_label_dir, f"{base}.txt")

        # 跳过已转换的文件（加速重复运行）
        if os.path.exists(out_img) and os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            skipped += 1
            continue

        # 读取图片获取尺寸
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        h, w = img.shape[:2]

        # 转为二值掩码
        binary = floodnet_mask_to_binary(label_path)
        if binary.sum() < 50:  # 跳过无淹没区域的图片
            continue

        # 转为 YOLO 多边形
        polygons = mask_to_yolo_polygons(binary, w, h, class_id=0)
        if not polygons:
            continue

        # 复制图片
        import shutil
        shutil.copy2(os.path.join(img_dir, fname), out_img)

        # 写入标签
        with open(txt_path, "w") as f:
            f.write("\n".join(polygons))
        converted += 1

    print(f"[FloodNet {split}] 转换完成: {converted} 新转换, {skipped} 已跳过, 总计 {converted+skipped} 张")
    return converted


def prepare_caslandslides_yolo_dataset(output_dir: str, split_ratio: float = 0.8):
    """
    将 CASlandslides 掩码转为 YOLO 格式
    split_ratio: 训练集比例
    """
    img_files = sorted([f for f in os.listdir(CASL_IMG) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))])
    n = len(img_files)
    n_train = int(n * split_ratio)

    for split_name, idx_range in [("train", range(n_train)), ("val", range(n_train, n))]:
        out_img_dir = os.path.join(output_dir, "images", split_name)
        out_label_dir = os.path.join(output_dir, "labels", split_name)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)

        converted = 0
        skipped = 0
        for i in idx_range:
            fname = img_files[i]
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(CASL_IMG, fname)
            mask_path = os.path.join(CASL_MASK, f"{base}.tif")

            if not os.path.exists(mask_path):
                # 尝试其他扩展名
                for ext in [".tif", ".tiff", ".png"]:
                    alt = os.path.join(CASL_MASK, f"{base}{ext}")
                    if os.path.exists(alt):
                        mask_path = alt
                        break
                else:
                    continue

            out_img = os.path.join(out_img_dir, fname)
            txt_path = os.path.join(out_label_dir, f"{base}.txt")

            # 跳过已转换的文件
            if os.path.exists(out_img) and os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                skipped += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            binary = caslandslides_mask_to_binary(mask_path)
            if binary.sum() < 50:
                continue

            polygons = mask_to_yolo_polygons(binary, w, h, class_id=0)
            if not polygons:
                continue

            import shutil
            shutil.copy2(img_path, out_img)

            with open(txt_path, "w") as f:
                f.write("\n".join(polygons))
            converted += 1

        print(f"[CASlandslides {split_name}] 转换完成: {converted} 新转换, {skipped} 已跳过, 总计 {converted+skipped} 张")


def rescuenet_mask_to_binary(mask_path: str) -> np.ndarray:
    """
    将 RescueNet 标签图转为二值洪水掩码
    类别 1=水体（洪水）→ 1，其余 → 0
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取标签文件: {mask_path}")
    binary = np.zeros_like(mask, dtype=np.uint8)
    for cls in RESCUENET_FLOOD_CLASS:
        binary[mask == cls] = 1
    return binary


def prepare_rescuenet_yolo_dataset(output_dir: str, val_ratio: float = 0.15):
    """
    将 RescueNet 语义分割标签转为 YOLO 分割格式（二值：洪水/非洪水）
    输出到 output_dir（images/labels 分 train/val）

    RescueNet 许可: CC BY-NC-ND
    """
    import shutil

    out_img_dir = os.path.join(output_dir, "images")
    out_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(os.path.join(out_img_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_img_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(out_label_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_label_dir, "val"), exist_ok=True)

    # 收集 org ↔ label 对应关系
    img_files = sorted([f for f in os.listdir(RESCUENET_IMG) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    pairs = []
    for fname in img_files:
        base = os.path.splitext(fname)[0]
        label_fname = f"{base}_lab.png"
        label_path = os.path.join(RESCUENET_LABEL, label_fname)
        if os.path.exists(label_path):
            pairs.append((fname, label_path))

    print(f"[RescueNet] org 图像: {len(img_files)}, 有对应 label: {len(pairs)}")

    # 按文件名排序后 split
    pairs.sort(key=lambda x: x[0])
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        converted = 0
        skipped = 0
        for fname, label_path in split_pairs:
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(RESCUENET_IMG, fname)
            out_img = os.path.join(out_img_dir, split_name, fname)
            txt_path = os.path.join(out_label_dir, split_name, f"{base}.txt")

            if os.path.exists(out_img) and os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                skipped += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            binary = rescuenet_mask_to_binary(label_path)
            # 跳过无洪水区域的图片（但保留少量以训练负样本）
            if binary.sum() < 100:
                continue

            polygons = mask_to_yolo_polygons(binary, w, h, class_id=0)
            if not polygons:
                continue

            shutil.copy2(img_path, out_img)
            with open(txt_path, "w") as f:
                f.write("\n".join(polygons))
            converted += 1

        print(f"[RescueNet {split_name}] 转换完成: {converted} 新转换, {skipped} 已跳过, 总计 {converted+skipped}")


def prepare_floodimg_yolo_dataset(output_dir: str, val_ratio: float = 0.2):
    """
    将 FloodIMG labelme JSON 转为 YOLO 检测格式
    输出到 output_dir（images/labels 分 train/val）

    labelme JSON 结构：
      imagePath: 对应图像文件名
      imageData: base64 编码的图像数据
      shapes: [{label, points, shape_type}]
    """
    import shutil
    import base64

    out_img_dir = os.path.join(output_dir, "images")
    out_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(os.path.join(out_img_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_img_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(out_label_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_label_dir, "val"), exist_ok=True)

    # 收集所有 JSON 文件
    json_files = sorted([f for f in os.listdir(FLOODIMG_ANN_DIR) if f.endswith(".json")])
    print(f"[FloodIMG] 找到 {len(json_files)} 个 JSON 标注文件")

    # 收集所有类别并分配 ID
    all_labels = set()
    for jf in json_files:
        with open(os.path.join(FLOODIMG_ANN_DIR, jf), "r", encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            all_labels.add(shape.get("label", "unknown"))
    label_to_id = {label: i for i, label in enumerate(sorted(all_labels))}
    print(f"[FloodIMG] 类别映射: {label_to_id}")

    # 按文件名排序后 split
    n_val = max(1, int(len(json_files) * val_ratio))
    val_files = json_files[:n_val]
    train_files = json_files[n_val:]

    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        converted = 0
        skipped = 0
        for jf in split_files:
            with open(os.path.join(FLOODIMG_ANN_DIR, jf), "r", encoding="utf-8") as f:
                data = json.load(f)

            img_name = data.get("imagePath", jf.replace(".json", ".jpg"))
            base = os.path.splitext(img_name)[0]

            # 尝试从 imageData 解码或从图像目录读取
            out_img = os.path.join(out_img_dir, split_name, img_name)
            txt_path = os.path.join(out_label_dir, split_name, f"{base}.txt")

            if os.path.exists(out_img) and os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                skipped += 1
                continue

            # 获取图像尺寸
            img_h = data.get("imageHeight", 0)
            img_w = data.get("imageWidth", 0)
            if img_h == 0 or img_w == 0:
                # 从 imageData 解码获取
                img_data = data.get("imageData", "")
                if img_data:
                    try:
                        img_bytes = base64.b64decode(img_data)
                        img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                        if img_arr is not None:
                            img_h, img_w = img_arr.shape[:2]
                            # 保存图片
                            cv2.imwrite(out_img, img_arr)
                    except Exception:
                        pass
                if img_h == 0 or img_w == 0:
                    # 尝试从图像目录读取
                    img_path = os.path.join(FLOODIMG_IMG_DIR, img_name)
                    if os.path.exists(img_path):
                        img_arr = cv2.imread(img_path)
                        if img_arr is not None:
                            img_h, img_w = img_arr.shape[:2]
                            shutil.copy2(img_path, out_img)
                    else:
                        continue

            # 转换 shapes 为 YOLO 检测格式
            yolo_lines = []
            for shape in data.get("shapes", []):
                label = shape.get("label", "unknown")
                cls_id = label_to_id.get(label, 0)
                points = shape.get("points", [])
                if len(points) < 2:
                    continue

                # 计算 bbox (x_min, y_min, x_max, y_max)
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # YOLO 格式: class_id x_center y_center width height (归一化)
                x_center = ((x_min + x_max) / 2) / img_w
                y_center = ((y_min + y_max) / 2) / img_h
                bw = (x_max - x_min) / img_w
                bh = (y_max - y_min) / img_h

                # 过滤无效框
                if bw <= 0 or bh <= 0 or bw > 1 or bh > 1:
                    continue

                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

            if not yolo_lines:
                continue

            # 如果图片还没保存（从 imageData 解码时已保存），从图片目录复制
            if not os.path.exists(out_img):
                img_path = os.path.join(FLOODIMG_IMG_DIR, img_name)
                if os.path.exists(img_path):
                    shutil.copy2(img_path, out_img)
                else:
                    continue

            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))
            converted += 1

        print(f"[FloodIMG {split_name}] 转换完成: {converted} 新转换, {skipped} 已跳过, 总计 {converted+skipped}")

    return label_to_id


def create_yolo_dataset_yaml(output_dir: str, task_name: str) -> str:
    """创建 YOLO 数据集配置文件"""
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    content = f"""# {task_name} 数据集
path: {output_dir}
train: images/train
val: images/val

nc: 1
names: ['{task_name}']
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    return yaml_path


def train_floodnet(epochs: int = 50, imgsz: int = 640, batch: int = 8, device: str = "cpu"):
    """
    训练内涝分割模型（YOLOv8n-seg）
    权重保存到 MODEL_DIR/floodnet.pt
    """
    from ultralytics import YOLO

    # 准备 YOLO 数据集
    yolo_dir = os.path.join(MODEL_DIR, "yolo_floodnet")
    print(f"准备 FloodNet YOLO 数据集: {yolo_dir}")
    prepare_floodnet_yolo_dataset(yolo_dir, "train")
    prepare_floodnet_yolo_dataset(yolo_dir, "val")
    yaml_path = create_yolo_dataset_yaml(yolo_dir, "flood")

    # 加载预训练权重
    model = YOLO("yolov8n-seg.pt")
    print(f"开始训练 FloodNet 内涝分割 (epochs={epochs}, imgsz={imgsz})")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=10,
        project=os.path.join(MODEL_DIR, "floodnet_train"),
        name="floodnet_exp",
        exist_ok=True,
        amp=True,
    )

    # 保存最佳权重
    best_path = os.path.join(MODEL_DIR, "floodnet_train", "floodnet_exp", "weights", "best.pt")
    target_path = os.path.join(MODEL_DIR, "flood.pt")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, target_path)
        print(f"模型权重已保存到: {target_path}")
    else:
        print(f"[警告] 未找到最佳权重: {best_path}")

    return results


def train_landslide(epochs: int = 50, imgsz: int = 320, batch: int = 16, device: str = "cpu"):
    """
    训练滑坡分割模型（YOLOv8n-seg）
    权重保存到 MODEL_DIR/landslide.pt
    """
    from ultralytics import YOLO

    yolo_dir = os.path.join(MODEL_DIR, "yolo_landslide")
    print(f"准备 CASlandslides YOLO 数据集: {yolo_dir}")
    prepare_caslandslides_yolo_dataset(yolo_dir)
    yaml_path = create_yolo_dataset_yaml(yolo_dir, "landslide")

    model = YOLO("yolov8n-seg.pt")
    print(f"开始训练 CASlandslides 滑坡分割 (epochs={epochs}, imgsz={imgsz})")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=10,
        project=os.path.join(MODEL_DIR, "landslide_train"),
        name="landslide_exp",
        exist_ok=True,
        amp=True,
    )

    best_path = os.path.join(MODEL_DIR, "landslide_train", "landslide_exp", "weights", "best.pt")
    target_path = os.path.join(MODEL_DIR, "landslide.pt")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, target_path)
        print(f"模型权重已保存到: {target_path}")

    return results


def train_rescuenet(epochs: int = 50, imgsz: int = 320, batch: int = 16, device: str = "cpu"):
    """
    训练 RescueNet 洪水分割模型（YOLOv8n-seg）
    权重保存到 MODEL_DIR/flood_rescuenet.pt
    RescueNet 许可: CC BY-NC-ND
    """
    from ultralytics import YOLO

    yolo_dir = os.path.join(MODEL_DIR, "yolo_rescuenet")
    print(f"准备 RescueNet YOLO 数据集: {yolo_dir}")
    prepare_rescuenet_yolo_dataset(yolo_dir)
    yaml_path = create_yolo_dataset_yaml(yolo_dir, "flood")

    model = YOLO("yolov8n-seg.pt")
    print(f"开始训练 RescueNet 洪水分割 (epochs={epochs}, imgsz={imgsz})")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=10,
        project=os.path.join(MODEL_DIR, "rescuenet_train"),
        name="rescuenet_exp",
        exist_ok=True,
        amp=True,
    )

    best_path = os.path.join(MODEL_DIR, "rescuenet_train", "rescuenet_exp", "weights", "best.pt")
    target_path = os.path.join(MODEL_DIR, "flood_rescuenet.pt")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, target_path)
        print(f"模型权重已保存到: {target_path}")

    return results


def train_floodimg_detect(epochs: int = 50, imgsz: int = 320, batch: int = 16, device: str = "cpu"):
    """
    训练 FloodIMG 洪水目标检测模型（YOLOv8n）
    权重保存到 MODEL_DIR/floodimg_detect.pt
    """
    from ultralytics import YOLO

    yolo_dir = os.path.join(MODEL_DIR, "yolo_floodimg")
    print(f"准备 FloodIMG YOLO 检测数据集: {yolo_dir}")
    label_to_id = prepare_floodimg_yolo_dataset(yolo_dir)
    yaml_path = create_yolo_det_dataset_yaml(yolo_dir, label_to_id)

    # 尝试本地缓存路径，避免网络下载
    local_yolo = os.path.expanduser("~/.cache/ultralytics/weights/yolov8n.pt")
    if os.path.exists(local_yolo):
        model = YOLO(local_yolo)
    else:
        model = YOLO("yolov8n.pt")
    print(f"开始训练 FloodIMG 目标检测 (epochs={epochs}, imgsz={imgsz})")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=10,
        project=os.path.join(MODEL_DIR, "floodimg_train"),
        name="floodimg_exp",
        exist_ok=True,
        amp=True,
    )

    best_path = os.path.join(MODEL_DIR, "floodimg_train", "floodimg_exp", "weights", "best.pt")
    target_path = os.path.join(MODEL_DIR, "floodimg_detect.pt")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, target_path)
        print(f"模型权重已保存到: {target_path}")

    return results


def create_yolo_det_dataset_yaml(output_dir: str, label_to_id: dict) -> str:
    """创建 YOLO 检测数据集配置文件"""
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    names = [k for k, v in sorted(label_to_id.items(), key=lambda x: x[1])]
    content = f"""# FloodIMG 检测数据集
path: {output_dir}
train: images/train
val: images/val

nc: {len(names)}
names: {json.dumps(names, ensure_ascii=False)}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return yaml_path


class FloodDetector:
    """洪水目标检测器：加载 YOLOv8n 检测模型进行推理"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化检测器
        model_path: 模型权重路径，None 则使用默认 floodimg_detect.pt
        """
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        else:
            default_path = os.path.join(MODEL_DIR, "floodimg_detect.pt")
            if os.path.exists(default_path):
                self.model_path = default_path
            else:
                self.model_path = "yolov8n.pt"
                print(f"[警告] 未找到专有模型 {default_path}，使用 YOLOv8 通用预训练权重")

        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        print(f"[FloodDetector] 加载模型: {self.model_path}")

    def detect(self, image_path: str, conf_thresh: float = 0.25) -> Dict:
        """
        对单张图片进行目标检测推理
        返回结构化字段
        """
        start_time = time.time()

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        img_height, img_width = img.shape[:2]

        # YOLO 检测推理
        results = self.model(image_path, conf=conf_thresh, verbose=False,
                             imgsz=320, max_det=50)
        infer_time = time.time() - start_time

        # 解析检测结果
        detections = []
        class_counts = {}
        max_box_area_ratio = 0.0

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.data.cpu().numpy()  # [N, 6] xyxy+conf+cls
            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                bw = x2 - x1
                bh = y2 - y1
                box_area_ratio = (bw * bh) / (img_width * img_height)
                class_name = results[0].names[int(cls_id)]

                detections.append({
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "area_ratio": float(box_area_ratio),
                })
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                if box_area_ratio > max_box_area_ratio:
                    max_box_area_ratio = box_area_ratio

        result = {
            "image_path": image_path,
            "image_size": f"{img_width}x{img_height}",
            "target数": len(detections),
            "最大框面积比例": round(max_box_area_ratio, 6),
            "类别分布": class_counts,
            "检测明细": detections,
            "推理耗时_s": round(infer_time, 4),
        }

        return result


class ImageSegmenter:
    """图像分割器：加载 YOLOv8n-seg 模型进行推理"""

    def __init__(self, model_path: Optional[str] = None, task: str = "flood"):
        """
        初始化分割器
        task: "flood" 或 "landslide"
        model_path: 模型权重路径，None 则使用默认预训练权重
        """
        self.task = task
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        else:
            # flood 任务优先加载 RescueNet 微调模型，回退到原 flood.pt
            if task == "flood":
                rescuenet_path = os.path.join(MODEL_DIR, "flood_rescuenet.pt")
                if os.path.exists(rescuenet_path):
                    self.model_path = rescuenet_path
                else:
                    default_path = os.path.join(MODEL_DIR, "flood.pt")
                    if os.path.exists(default_path):
                        self.model_path = default_path
                    else:
                        self.model_path = "yolov8n-seg.pt"
                        print(f"[警告] 未找到专有模型，使用 YOLOv8 通用预训练权重")
            else:
                default_path = os.path.join(MODEL_DIR, f"{task}.pt")
                if os.path.exists(default_path):
                    self.model_path = default_path
                else:
                    self.model_path = "yolov8n-seg.pt"
                    print(f"[警告] 未找到专有模型 {default_path}，使用 YOLOv8 通用预训练权重")

        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        print(f"[ImageSegmenter] 加载模型: {self.model_path}")

    def predict(self, image_path: str, conf_thresh: float = 0.25) -> Dict:
        """
        对单张图片进行分割推理
        返回结构化字段
        """
        start_time = time.time()

        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        img_height, img_width = img.shape[:2]

        # YOLO 推理（优化：固定 imgsz=640, 限制最大检测数, 启用 FP16）
        results = self.model(image_path, conf=conf_thresh, verbose=False,
                             imgsz=320, max_det=10)
        infer_time = time.time() - start_time

        # 解析结果
        total_mask_pixels = 0
        if results and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # [N, H, W]
            # 合并所有检测到的掩码
            combined = np.zeros((img_height, img_width), dtype=np.uint8)
            for mask in masks:
                # 将 masks 从原始尺寸 resize 到图片尺寸
                mask_resized = cv2.resize(mask, (img_width, img_height))
                combined[mask_resized > 0.5] = 1
            total_mask_pixels = int(combined.sum())

        # 计算面积
        total_area_m2 = total_mask_pixels * PIXEL_AREA_M2
        total_image_pixels = img_height * img_width
        inundation_ratio = total_mask_pixels / total_image_pixels if total_image_pixels > 0 else 0.0

        disaster_level = get_disaster_level(total_area_m2, is_landslide=(self.task == "landslide"))

        result = {
            "image_path": image_path,
            "image_size": f"{img_width}x{img_height}",
            "task": self.task,
            "mask_pixels": total_mask_pixels,
            "area_m2": round(total_area_m2, 2),
            "inundation_ratio": round(inundation_ratio, 6),
            "disaster_level": disaster_level,
            "inference_time_s": round(infer_time, 4),
        }

        # 根据任务类型添加字段名
        if self.task == "flood":
            result["积水面积_m2"] = result.pop("area_m2")
            result["淹没占比"] = result.pop("inundation_ratio")
            result["灾情等级"] = result.pop("disaster_level")
        else:
            result["滑坡面积_m2"] = result.pop("area_m2")
            result["滑坡占比"] = result.pop("inundation_ratio")
            result["灾情等级"] = result.pop("disaster_level")

        result["推理耗时_s"] = result.pop("inference_time_s")

        return result

    def predict_batch(self, image_paths: List[str], conf_thresh: float = 0.25) -> List[Dict]:
        """批量推理"""
        results = []
        for path in image_paths:
            try:
                res = self.predict(path, conf_thresh)
                results.append(res)
            except Exception as e:
                results.append({"image_path": path, "error": str(e)})
        return results


def evaluate_on_floodnet_val(model_path: Optional[str] = None, num_samples: int = -1) -> Dict:
    """
    在 FloodNet val 集上评估模型
    返回 mIoU, mAP 等指标
    """
    segmenter = ImageSegmenter(model_path, task="flood")

    val_images = sorted([f for f in os.listdir(FLOODNET_VAL_IMG) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if num_samples > 0:
        val_images = val_images[:num_samples]

    ious = []
    times = []
    for fname in val_images:
        img_path = os.path.join(FLOODNET_VAL_IMG, fname)
        base = os.path.splitext(fname)[0]
        label_path = os.path.join(FLOODNET_VAL_LABEL, f"{base}_lab.png")

        if not os.path.exists(label_path):
            continue

        # 真实标签
        gt_binary = floodnet_mask_to_binary(label_path)

        # 推理
        result = segmenter.predict(img_path)
        times.append(result["推理耗时_s"])

        # 预测掩码
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        pred_binary = np.zeros((h, w), dtype=np.uint8)
        yolo_results = segmenter.model(img_path, conf=0.25, verbose=False, imgsz=320, max_det=10)
        if yolo_results and yolo_results[0].masks is not None:
            masks = yolo_results[0].masks.data.cpu().numpy()
            for mask in masks:
                mask_resized = cv2.resize(mask, (w, h))
                pred_binary[mask_resized > 0.5] = 1

        # 确保维度一致（标签可能为 (H,W,1)，预测为 (H,W)）
        gt_binary = np.squeeze(gt_binary)
        pred_binary = np.squeeze(pred_binary)

        # 计算 IoU
        intersection = np.logical_and(gt_binary, pred_binary).sum()
        union = np.logical_or(gt_binary, pred_binary).sum()
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)

    mean_iou = np.mean(ious) if ious else 0.0
    mean_time = np.mean(times) if times else 0.0

    report = {
        "num_samples": len(ious),
        "mIoU": round(float(mean_iou), 4),
        "平均推理耗时_s": round(float(mean_time), 4),
        "FPS": round(1.0 / mean_time, 2) if mean_time > 0 else 0,
    }
    print(f"\n[FloodNet Val 评估] 样本数: {report['num_samples']}, "
          f"mIoU: {report['mIoU']:.4f}, 平均耗时: {report['平均推理耗时_s']:.4f}s, "
          f"FPS: {report['FPS']:.2f}")
    return report


def evaluate_on_caslandslides(model_path: Optional[str] = None, num_samples: int = -1) -> Dict:
    """在 CASlandslides 上评估模型"""
    segmenter = ImageSegmenter(model_path, task="landslide")

    img_files = sorted([f for f in os.listdir(CASL_IMG) if f.lower().endswith((".tif", ".tiff", ".jpg", ".png"))])
    if num_samples > 0:
        img_files = img_files[:num_samples]

    ious = []
    times = []
    for fname in img_files:
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(CASL_IMG, fname)
        mask_path = os.path.join(CASL_MASK, f"{base}.tif")

        if not os.path.exists(mask_path):
            continue

        gt_binary = caslandslides_mask_to_binary(mask_path)
        result = segmenter.predict(img_path)
        times.append(result["推理耗时_s"])

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        pred_binary = np.zeros((h, w), dtype=np.uint8)
        yolo_results = segmenter.model(img_path, conf=0.25, verbose=False, imgsz=320, max_det=10)
        if yolo_results and yolo_results[0].masks is not None:
            masks = yolo_results[0].masks.data.cpu().numpy()
            for mask in masks:
                mask_resized = cv2.resize(mask, (w, h))
                pred_binary[mask_resized > 0.5] = 1

        # 确保维度一致
        gt_binary = np.squeeze(gt_binary)
        pred_binary = np.squeeze(pred_binary)

        intersection = np.logical_and(gt_binary, pred_binary).sum()
        union = np.logical_or(gt_binary, pred_binary).sum()
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)

    mean_iou = np.mean(ious) if ious else 0.0
    mean_time = np.mean(times) if times else 0.0

    report = {
        "num_samples": len(ious),
        "mIoU": round(float(mean_iou), 4),
        "平均推理耗时_s": round(float(mean_time), 4),
        "FPS": round(1.0 / mean_time, 2) if mean_time > 0 else 0,
    }
    print(f"\n[CASlandslides 评估] 样本数: {report['num_samples']}, "
          f"mIoU: {report['mIoU']:.4f}, 平均耗时: {report['平均推理耗时_s']:.4f}s, "
          f"FPS: {report['FPS']:.2f}")
    return report


# ==================== CLI ====================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="灾害链图像预处理管线")
    parser.add_argument("--mode", choices=["train_flood", "train_landslide", "train_rescuenet",
                        "train_floodimg", "infer", "detect", "eval", "convert"],
                        default="infer", help="运行模式")
    parser.add_argument("--image", type=str, help="推理图片路径")
    parser.add_argument("--task", choices=["flood", "landslide", "flood_detect"], default="flood", help="推理任务类型")
    parser.add_argument("--model", type=str, help="模型权重路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备")
    parser.add_argument("--num_samples", type=int, default=-1, help="评估样本数，-1 使用全部")
    args = parser.parse_args()

    if args.mode == "train_flood":
        train_floodnet(epochs=args.epochs, device=args.device)
    elif args.mode == "train_landslide":
        train_landslide(epochs=args.epochs, device=args.device)
    elif args.mode == "train_rescuenet":
        train_rescuenet(epochs=args.epochs, device=args.device)
    elif args.mode == "train_floodimg":
        train_floodimg_detect(epochs=args.epochs, device=args.device)
    elif args.mode == "infer":
        if not args.image:
            print("请指定 --image 图片路径")
            return
        segmenter = ImageSegmenter(args.model, task=args.task)
        result = segmenter.predict(args.image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "detect":
        if not args.image:
            print("请指定 --image 图片路径")
            return
        detector = FloodDetector(args.model)
        result = detector.detect(args.image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "eval":
        print("=" * 50)
        print("FloodNet Val 评估")
        evaluate_on_floodnet_val(args.model, num_samples=args.num_samples)
        print("\n" + "=" * 50)
        print("CASlandslides 评估")
        evaluate_on_caslandslides(args.model, num_samples=args.num_samples)
    elif args.mode == "convert":
        # 转换所有数据集
        yolo_dir = os.path.join(MODEL_DIR, "yolo_floodnet")
        prepare_floodnet_yolo_dataset(yolo_dir, "train")
        prepare_floodnet_yolo_dataset(yolo_dir, "val")
        yolo_dir2 = os.path.join(MODEL_DIR, "yolo_landslide")
        prepare_caslandslides_yolo_dataset(yolo_dir2)
        yolo_dir3 = os.path.join(MODEL_DIR, "yolo_rescuenet")
        prepare_rescuenet_yolo_dataset(yolo_dir3)
        yolo_dir4 = os.path.join(MODEL_DIR, "yolo_floodimg")
        prepare_floodimg_yolo_dataset(yolo_dir4)


if __name__ == "__main__":
    main()