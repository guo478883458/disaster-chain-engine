"""
路径配置模块（三级回退设计）
============================
统一管理所有数据/权重路径，实现路径可移植化。

优先级（三级回退）：
  1. 环境变量 DISASTER_DATA_DIR（显式覆盖）
  2. 原 H 盘路径（本机兼容，零影响）
  3. 包内 data/ 目录（新机器）

使用方式：
    from path_config import (
        DATA_ROOT, MODELS_DIR, GEOJSON_PATH,
        FLOOD_PT, FLOOD_RESCUENET_PT, ...
    )
"""

import os
import sys

# ── 发布包根目录 = 本文件所在目录的上级 ──
# 若 path_config.py 放在包根目录，则 PACKAGE_ROOT = 本文件目录
# 若放在 v2_engine/ 内，则 PACKAGE_ROOT = 上级目录
_PATH_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# 判断是否在 v2_engine/ 子目录中（发布包结构）
if os.path.basename(_PATH_CONFIG_DIR) in ("v2_engine", "灾害链推理引擎-v2"):
    PACKAGE_ROOT = os.path.dirname(_PATH_CONFIG_DIR) if os.path.basename(_PATH_CONFIG_DIR) == "v2_engine" else _PATH_CONFIG_DIR
else:
    PACKAGE_ROOT = _PATH_CONFIG_DIR

# ── 数据根目录：三级回退 ──
_ORIG_H = r"H:\dev\disaster-data"
_env_data = os.environ.get("DISASTER_DATA_DIR")
if _env_data:
    DATA_ROOT = _env_data
elif os.path.exists(_ORIG_H):
    DATA_ROOT = _ORIG_H
else:
    DATA_ROOT = os.path.join(PACKAGE_ROOT, "data")

# ==================== 模型权重路径 ====================
MODELS_DIR = os.path.join(DATA_ROOT, "models")

# 基础设施模型权重子目录
INFRA_WATER_LEVEL_DIR = os.path.join(MODELS_DIR, "infra", "water_level")
INFRA_ROAD_BRIDGE_DIR = os.path.join(MODELS_DIR, "infra", "road_bridge")
INFRA_FLOOD_DIR = os.path.join(MODELS_DIR, "infra", "flood_detection")

# 各权重文件（存在性检查在调用时进行）
FLOOD_PT = os.path.join(MODELS_DIR, "flood.pt")
FLOOD_RESCUENET_PT = os.path.join(MODELS_DIR, "flood_rescuenet.pt")
FLOODIMG_DETECT_PT = os.path.join(MODELS_DIR, "floodimg_detect.pt")
LANDSLIDE_PT = os.path.join(MODELS_DIR, "landslide.pt")
WATER_LEVEL_DIGIT_PT = os.path.join(INFRA_WATER_LEVEL_DIR, "water_level_digit_detector.pt")
ROAD_BRIDGE_PT = os.path.join(INFRA_ROAD_BRIDGE_DIR, "rdd2020_yolov8n.pt")

# ==================== 地理数据 ====================
GEOJSON_PATH = os.path.join(DATA_ROOT, "zhengzhou_720", "zhengzhou_geojson.json")
ZHENGZHOU_DIR = os.path.join(DATA_ROOT, "zhengzhou_720")

# ==================== 基础设施数据目录 ====================
WATER_LEVEL_DATA_DIR = os.path.join(
    DATA_ROOT, "infra_datasets", "water_level", "extracted", "SAM_water_level_Dataset"
)
WATER_LEVEL_IMAGES_DIR = os.path.join(WATER_LEVEL_DATA_DIR, "Staff gauge images")
WATER_LEVEL_XLSX = os.path.join(
    WATER_LEVEL_DATA_DIR, "In-situ water levels", "In-situ & simulated water levels.xlsx"
)
RDD2020_DIR = os.path.join(DATA_ROOT, "infra_datasets", "rdd2020")

# ==================== 训练数据集（只读引用） ====================
FLOODNET_DIR = os.path.join(
    DATA_ROOT, "image_datasets", "floodnet", "extracted", "FloodNet-Supervised_v1.0"
)
FLOODNET_TEST_COLORMASK = os.path.join(
    DATA_ROOT, "image_datasets", "floodnet", "extracted", "ColorMasks-FloodNetv1.0", "ColorMasks-TestSet"
)
CASL_DIR = os.path.join(DATA_ROOT, "image_datasets", "caslandslides")
RESCUENET_DIR = os.path.join(DATA_ROOT, "image_datasets", "rescuenet", "segmentation-trainset")
FLOODIMG_DIR = os.path.join(DATA_ROOT, "image_datasets", "floodimg")
CRISIS_DIR = os.path.join(DATA_ROOT, "image_datasets", "crisisnlp_c", "CrisisNLP-C-master")

# ==================== 基础设施项目路径（发布包内） ====================
# 在发布包中，基础设施代码放在 infra_recognition/
INFRA_PROJECT_DIR = os.path.join(PACKAGE_ROOT, "infra_recognition")
if not os.path.exists(INFRA_PROJECT_DIR):
    # 开发环境：基础设施代码在独立目录
    INFRA_PROJECT_DIR = os.path.join(os.path.dirname(PACKAGE_ROOT), "基础设施灾损识别")