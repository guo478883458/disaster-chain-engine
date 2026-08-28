# 灾害链推理演示系统 v1

基于贝叶斯网络的郑州城市洪涝/地质灾害链推理与可视化系统。

---

## 快速开始（3 步）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或使用 conda：

```bash
conda create -n disasterlex python=3.10
conda activate disasterlex
pip install -r requirements.txt
```

> **PyTorch 安装**（CPU 版）：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 2. 启动 Dashboard

双击 `启动Dashboard.bat`，或在终端中运行：

```bash
python -m streamlit run dashboard.py
```

### 3. 打开浏览器

访问 **http://localhost:8501**

- 在地图总览页选择"演示参数"推演数据来源
- 点击"🚀 使用演示参数推演全部 12 个区"
- 查看各区风险评估结果

---

## 环境要求

| 项目 | 最低要求 |
|------|---------|
| 操作系统 | Windows 10/11 |
| Python | 3.10+ |
| 磁盘空间 | ≥ 2GB |
| 内存 | ≥ 8GB |
| 网络 | 首次运行需下载模型权重（自动） |

## 目录结构

```
灾害链演示_v1/
├── README.md                         # 本文件
├── requirements.txt                  # Python 依赖
├── path_config.py                    # 路径配置（三级回退）
├── 启动Dashboard.bat                  # 一键启动脚本
├── v2_engine/                        # 灾害链推理引擎
│   ├── dashboard.py                  # 可视化主界面
│   ├── bn_engine.py                  # 贝叶斯网络推理引擎
│   ├── region_engine.py              # 区域推理引擎
│   ├── tools/                        # 推理工具
│   ├── scripts/                      # 辅助脚本
│   └── configs/                      # 模型配置
├── infra_recognition/                # 基础设施灾损识别
│   ├── models/                       # 水位/道路/洪水模型
│   └── tools/                        # 推理接口
├── data/
│   ├── models/                       # 模型权重（~80MB）
│   ├── zhengzhou_720/                # 郑州地理数据
│   ├── infra_datasets/               # 基础设施数据集
│   └── demo_images/                  # 演示图片
└── 演示材料/
    └── 演示参数清单.md
```

## 常见问题

| 问题 | 解决 |
|------|------|
| 端口 8501 被占用 | 运行 `streamlit run dashboard.py --server.port 8502` |
| 权重文件缺失 | 检查 `data/models/` 目录是否存在且包含 `.pt` 文件 |
| 首次加载慢 | 首次启动 30~60 秒属正常（模型加载 + 缓存构建） |
| 地图显示空白 | 检查 `data/zhengzhou_720/zhengzhou_geojson.json` 是否存在 |
| 图片识别失败 | 检查 `data/demo_images/` 目录下的演示图片 |

## 许可证

- 本系统代码仅供学习研究使用
- RescueNet 数据集：CC BY-NC-ND（非商业用途）
- RDD2020 数据集：学术研究用途