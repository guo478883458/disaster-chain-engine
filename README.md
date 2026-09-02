# 灾害链推理引擎 v2

40 节点贝叶斯网络（暴雨→内涝→地质灾害链式推理）+ 郑州真实数据可视化 + 实时监测 + 视觉识别联动的城市灾害态势感知系统。

---

## 功能特性

### 6 个标签页

| 标签页 | 功能 |
|--------|------|
| **🗺️ 地图总览** | 郑州 12 区风险态势热力图，置信度显示，预测时段（现在/未来）切换 |
| **📋 区详情** | 单区参数入口（降水/土壤/水位等），BN 节点置信度展示 |
| **⏱️ 链式预测** | 暴雨→内涝→地质灾害六阶段递进推理，现在→未来预测切换 |
| **📡 实时监测** | 回放模式：ERA5 144 小时逐小时回放 + 趋势预测（AR 降水规则 / LSTM 土壤湿度 / 水位回落检测）；数据流模式：FastAPI 持续接入 + 图片识别联动 |
| **⚙️ 参数** | 系统参数配置与查看 |
| **ℹ️ 帮助** | 使用说明与文档链接 |

### 视觉识别接入

系统支持 4 类图片识别任务，识别结果直接映射为贝叶斯网络证据，驱动风险概率更新：

| 任务类型 | 模型 | 输出 | BN 证据 |
|----------|------|------|---------|
| **水位尺** (`water_level`) | YOLOv8n 数字检测 + EasyOCR 读数 | 当前水位高度（cm） | 河道水位、内涝深度 |
| **道路损毁** (`road`) | YOLOv8n（RDD2020 训练） | 裂缝/龟裂/坑洼数量 | 道路积水历史频率 |
| **洪水分割** (`flood`) | YOLOv8n-seg（RescueNet 训练） | 积水面积（m²）、淹没占比、灾情等级 | 内涝深度 |
| **滑坡分割** (`landslide`) | ImageSegmenter（CASlandslides 训练） | 滑坡面积（m²）、灾情等级 | 滑坡历史密度 |

**联动流程**：图片推流/上传 → 模型识别 → 结构化字段 → `map_to_bn_states` 映射为 BN 证据 → 贝叶斯网络推理 → 各节点风险概率更新 → 地图总览/区详情实时反映。

---

## 目录结构

```
灾害链推理引擎-v2/
├── README.md                         # 本文件
├── 项目进展与改动记录.md              # 完整开发历史
├── requirements.txt                  # Python 依赖
├── path_config.py                    # 路径配置（三级回退）
├── 启动Dashboard.bat                  # 一键启动脚本
├── dashboard.py                      # 可视化主界面（Streamlit）
├── bn_engine.py                      # 贝叶斯网络推理引擎
├── region_engine.py                  # 区域推理引擎
├── demo_infer.py                     # 单图推理演示
├── demo_interactive.py               # 交互式推理演示
├── demo_real.py                      # 真实数据演示
├── run.py                            # 批量推理入口
├── test_40nodes.py / test_all.py     # 测试与验证
├── configs/
│   ├── config_40nodes.yaml           # 40 节点 BN 配置（主网络）
│   ├── config_20nodes.yaml           # 20 节点 BN 配置
│   ├── config_8nodes.yaml            # 8 节点精简配置
│   ├── 郑州/                         # 郑州区域配置、demo_params、区证据
│   ├── 北京/                         # 北京区域配置
│   └── 模板/                         # 配置模板
├── scripts/
│   ├── ingest_server.py              # 数据流接入服务器（FastAPI, port 8502）
│   ├── simulate_data_stream.py       # 数据流推流模拟（ERA5/图片/随机）
│   ├── forecast_engine.py            # 趋势预测引擎（AR/LSTM/水位回落）
│   ├── build_era5_timeseries.py      # ERA5 时序构建
│   ├── build_zz_evidence.py          # 郑州证据构建
│   └── package_release.py            # 发布包生成
├── tools/
│   ├── fuse_infer.py                 # 统一推理入口（CLI + 证据映射）
│   ├── preprocess_api.py             # 视觉预处理 API（含 ImageSegmenter）
│   ├── image_preprocess.py           # 图片预处理工具
│   ├── text_preprocess.py            # 文本预处理工具
│   ├── validate_bn.py                # BN 网络验证
│   ├── visualize.py                  # BN 结构可视化
│   └── test_e2e.py                   # 端到端测试
└── data/
    ├── models/                       # 模型权重（~80MB，自动下载）
    ├── zhengzhou_720/                # 郑州地理数据（GeoJSON/网格/洪水区）
    └── era5land/                     # ERA5 再分析数据
```

---

## 快速开始（3 步）

### 1. 安装依赖

```bash
conda create -n disasterlex python=3.10
conda activate disasterlex
pip install -r requirements.txt
```

> **PyTorch 安装**（GPU 可选）：
> - 有 GPU（RTX 4060 已验证）：安装 CUDA 版 torch
> - 无 GPU：安装 CPU 版 torch
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

> **注意**：`tifffile` 必须装入 conda 环境而非用户目录（`PYTHONNOUSERSITE=1` 下用户级包不可见，会导致后台线程 easyocr 导入失败）。

### 2. 启动 Dashboard

双击 `启动Dashboard.bat`，或在终端中运行：

```bash
streamlit run dashboard.py
```

### 3. 打开浏览器

访问 **http://localhost:8501**

- 在地图总览页选择"演示参数"推演数据来源
- 点击"🚀 使用演示参数推演全部 12 个区"
- 查看各区风险评估结果

### 数据路径说明

系统通过 `DISASTER_DATA_DIR` 环境变量三级回退定位数据目录：

1. **环境变量** `DISASTER_DATA_DIR` 指定的路径
2. **H 盘原目录** `H:\dev\disaster-data`
3. **包内 data/** 目录（回退选项）

---

## 环境要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| 操作系统 | Windows 10/11 | Windows 11 |
| Python | 3.10+ | 3.10 |
| 磁盘空间 | ≥ 2GB | ≥ 10GB（含数据） |
| 内存 | ≥ 8GB | ≥ 16GB |
| GPU | 可选（CPU 可运行，推理变慢） | RTX 4060（已验证） |
| 网络 | 首次运行需下载模型权重（自动） | — |

---

## 常见问题

| 问题 | 解决 |
|------|------|
| 端口 8501 被占用 | 运行 `streamlit run dashboard.py --server.port 8502` |
| 权重文件缺失 | 检查 `data/models/` 目录是否存在且包含 `.pt` 文件 |
| 首次加载慢 | 首次启动 30~60 秒属正常（模型加载 + 缓存构建） |
| 地图显示空白 | 检查 `data/zhengzhou_720/zhengzhou_geojson.json` 是否存在 |
| 图片识别无记录 | 确认推流已启动（`python scripts/simulate_data_stream.py --mode image`）；interval 建议 ≥10s（滑坡推理约 8s） |
| easyocr 导入失败 | 确认 `tifffile` 已装入 conda 环境而非用户目录；在 `PYTHONNOUSERSITE=1` 下测试 `python -c "import easyocr"` |
| 滑坡推理报错 | 确认 `landslide.pt` 存在于模型目录 |

---

## 更新记录

详见 [项目进展与改动记录.md](./项目进展与改动记录.md)，涵盖从 v1 演示系统到 40 节点 v2、郑州真实数据重构、实时监测、图片识别、滑坡接入、演示优化等完整开发历史。

---

## 许可证

- 本系统代码仅供学习研究使用
- RescueNet 数据集：CC BY-NC-ND（非商业用途）
- RDD2020 数据集：学术研究用途
- CASlandslides 数据集：学术研究用途