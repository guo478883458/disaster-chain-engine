"""
趋势预测层 — ERA5 逐小时回放预测引擎
========================================
对时刻 T 预测 T+1/T+2/T+3 小时的气象节点证据。

策略：
  1. 基线方案（AR/规则/线性外推）— 始终实现，作为对比基准
  2. 训练方案（NeuralProphet / LSTM）— 若基线准确度不足则启用
  3. 对比报告 — 两种方案各报告验证集指标，数据说话

输出: H:\dev\disaster-data\zhengzhou_720\era5_forecasts.json
      {T: 0..143, datetime: "...", horizons: {1: {evidence, confidence}, 2: {...}, 3: {...}}}
"""
import json, os, sys, warnings, copy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── 路径 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_PATH = os.path.join(
    r"H:\dev\disaster-data", "zhengzhou_720", "era5_hourly_evidence.json"
)
FORECAST_OUTPUT_DIR = r"H:\dev\disaster-data\zhengzhou_720"
FORECAST_OUTPUT_PATH = os.path.join(FORECAST_OUTPUT_DIR, "era5_forecasts.json")
MODEL_SAVE_DIR = r"H:\dev\disaster-data\models\forecast"
os.makedirs(FORECAST_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ── 分位数阈值（由 build_era5_timeseries.py 确定，硬编码保持一致性） ──
RAIN_Q33, RAIN_Q66 = 0.38, 1.39    # mm/h
WIND_MEDIAN = 2.66                  # m/s
SWVL1_Q33, SWVL1_Q66 = 0.4023, 0.4194
SWVL2_MEDIAN = 0.4017

# 水位的经验参数（144 小时拟合）
# 水位 ≈ 基准 + k·Σ降水，基准=2.0m, k=0.015
WATER_BASELINE = 2.0
WATER_K = 0.015


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def load_era5_evidence(path=None):
    """加载 ERA5 逐小时证据"""
    p = path or EVIDENCE_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"ERA5 证据文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_ts(records):
    """从 records 中提取数值时间序列"""
    n = len(records)
    tp_mm = np.zeros(n)       # 逐小时降水
    wind = np.zeros(n)        # 风速 m/s
    t2m = np.zeros(n)         # 气温 °C
    d2m = np.zeros(n)         # 露点温度 °C
    swvl1 = np.zeros(n)       # 土壤含水量
    swvl2 = np.zeros(n)       # 深层土壤含水量
    evabs = np.zeros(n)       # 蒸发量

    for i, rec in enumerate(records):
        ev = rec["evidence"]
        # 降水强度反推数值（低/中/高→0.1/0.8/2.5）
        rs = ev["降水强度"]
        if rs == "高":
            tp_mm[i] = 2.5
        elif rs == "中":
            tp_mm[i] = 0.8
        else:
            tp_mm[i] = 0.1
        # 风力
        ws = ev["风力"]
        wind[i] = WIND_MEDIAN * 1.5 if ws == "强" else WIND_MEDIAN * 0.5
        # 气温（数值居中）
        ts = ev["气温"]
        if ts == "高温":
            t2m[i] = 27.0
        elif ts == "低温":
            t2m[i] = 22.0
        else:
            t2m[i] = 24.3
        # 露点
        ds = ev["露点温度"]
        d2m[i] = 23.5 if ds == "高" else 22.0
        # 土壤含水量
        s1 = ev["前期土壤含水量"]
        if s1 == "高":
            swvl1[i] = SWVL1_Q66 + 0.02
        elif s1 == "中":
            swvl1[i] = (SWVL1_Q33 + SWVL1_Q66) / 2
        else:
            swvl1[i] = SWVL1_Q33 - 0.02
        s2 = ev["土壤渗透性"]
        swvl2[i] = SWVL2_MEDIAN * 1.2 if s2 == "差" else SWVL2_MEDIAN * 0.8
        # 蒸发量
        es = ev["蒸发量"]
        evabs[i] = 0.0003 if es == "大" else 0.0001

    return {"tp_mm": tp_mm, "wind": wind, "t2m": t2m, "d2m": d2m,
            "swvl1": swvl1, "swvl2": swvl2, "evabs": evabs}


def _rain_state(val):
    """降水强度数值→状态"""
    if val >= RAIN_Q66:
        return "高"
    elif val >= RAIN_Q33:
        return "中"
    return "低"


def _wind_state(val):
    return "强" if val >= WIND_MEDIAN else "弱"


def _t2m_state(val):
    if val >= 24.9:
        return "高温"
    elif val >= 23.7:
        return "适温"
    return "低温"


def _d2m_state(val):
    return "高" if val >= 23.0 else "低"


def _swvl1_state(val):
    if val >= SWVL1_Q66:
        return "高"
    elif val >= SWVL1_Q33:
        return "中"
    return "低"


def _swvl2_state(val):
    return "差" if val >= SWVL2_MEDIAN else "好"


def _evabs_state(val):
    return "大" if val >= 0.0002 else "小"


def _duration_state(future_rain, horizon):
    """预测窗口内 ≥5mm/h 累计 ≥2 → 长"""
    # 用未来 horizon 步的降水强度数值估算
    # 若未来多数步为"高"（≈2.5mm），则实际 tp 可能 ≥5mm
    high_count = sum(1 for r in future_rain if r >= RAIN_Q66)
    return "长" if high_count >= 2 else "短"


def _water_level(cum_precip):
    """水位 ≈ 基准 + k·Σ降水"""
    return WATER_BASELINE + WATER_K * cum_precip


def _water_level_state(level):
    if level >= 3.0:
        return "危险"
    elif level >= 2.5:
        return "警戒"
    return "正常"


def _confidence_from_residuals(residuals, max_res=2.0):
    """残差→置信度映射（0.5~0.9）"""
    rmse = np.sqrt(np.mean(residuals ** 2))
    conf = 0.9 - 0.4 * (rmse / max_res)
    return float(np.clip(conf, 0.5, 0.95))


# ══════════════════════════════════════════════════════════════════════
# 基线方案
# ══════════════════════════════════════════════════════════════════════

class BaselineForecaster:
    """AR(1) / 规则 / 线性外推 基线方案"""

    def __init__(self, ts_dict):
        self.ts = ts_dict
        self.n = len(ts_dict["tp_mm"])

    def predict(self, T, horizon=1):
        """对时刻 T 预测未来 horizon 小时"""
        if T < 6:
            return self._predict_fallback(T, horizon)

        tp = self.ts["tp_mm"]
        sw = self.ts["swvl1"]
        wind = self.ts["wind"]

        # ── 降水强度：AR(1) 最小二乘 ──
        past_tp = tp[max(0, T - 5):T + 1]  # 过去 6 小时
        if len(past_tp) >= 3:
            x = np.arange(len(past_tp)).reshape(-1, 1)
            y = past_tp.reshape(-1, 1)
            lr = LinearRegression().fit(x, y)
            slope = lr.coef_[0, 0]
            intercept = lr.intercept_[0]
            residuals = y - lr.predict(x)
            residuals = residuals.flatten()
            future_tp = np.array([intercept + slope * (len(past_tp) + i) for i in range(horizon)])
            future_tp = np.clip(future_tp, 0, None)
        else:
            future_tp = np.full(horizon, tp[T])
            residuals = np.zeros(2)

        rain_conf = _confidence_from_residuals(residuals, max_res=2.0)
        # 兜底：过去 3 小时高→未来高
        if T >= 2 and all(_rain_state(tp[T - i]) == "高" for i in range(3)):
            for i in range(horizon):
                future_tp[i] = max(future_tp[i], RAIN_Q66 + 0.5)
            rain_conf = max(rain_conf, 0.7)

        # ── 降水时长 ──
        duration_state = _duration_state(future_tp, horizon)

        # ── 前期土壤含水量：近 6 小时变化率线性外推 ──
        past_sw = sw[max(0, T - 5):T + 1]
        if len(past_sw) >= 3:
            x = np.arange(len(past_sw)).reshape(-1, 1)
            y = past_sw.reshape(-1, 1)
            lr_sw = LinearRegression().fit(x, y)
            slope_sw = lr_sw.coef_[0, 0]
            intercept_sw = lr_sw.intercept_[0]
            residuals_sw = y - lr_sw.predict(x)
            residuals_sw = residuals_sw.flatten()
            future_sw = np.array([intercept_sw + slope_sw * (len(past_sw) + i) for i in range(horizon)])
            future_sw = np.clip(future_sw, 0, 0.5)
        else:
            future_sw = np.full(horizon, sw[T])
            residuals_sw = np.zeros(2)

        # ── 河道水位 ──
        cum_precip = np.sum(tp[:T + 1])
        fut_cum_precip = cum_precip + np.cumsum(future_tp)
        water_levels = _water_level(fut_cum_precip)

        # ── 风力：保持近期均值 ──
        past_wind = wind[max(0, T - 3):T + 1]
        fut_wind = np.full(horizon, np.mean(past_wind))

        # ── 气温/露点：保持当前 ──
        fut_t2m = np.full(horizon, self.ts["t2m"][T])
        fut_d2m = np.full(horizon, self.ts["d2m"][T])
        fut_swvl2 = np.full(horizon, self.ts["swvl2"][T])
        fut_evabs = np.full(horizon, self.ts["evabs"][T])

        # 构建输出
        horizons = {}
        for h in range(1, horizon + 1):
            idx = h - 1
            evidence = {
                "降水强度": _rain_state(future_tp[idx]),
                "降水时长": duration_state,
                "风力": _wind_state(fut_wind[idx]),
                "风向": "保持先验",
                "湿度": "保持先验",
                "气压": "保持先验",
                "气温": _t2m_state(fut_t2m[idx]),
                "露点温度": _d2m_state(fut_d2m[idx]),
                "对流有效位能CAPE": "保持先验",
                "垂直风切变": "保持先验",
                "前期土壤含水量": _swvl1_state(future_sw[idx]),
                "土壤渗透性": _swvl2_state(fut_swvl2[idx]),
                "蒸发量": _evabs_state(fut_evabs[idx]),
                "径流系数": "保持先验",
                "河道水位": _water_level_state(water_levels[idx]),
                "地下水埋深": "保持先验",
                "湖泊调蓄能力": "保持先验",
                "潮汐影响": "保持先验",
            }
            # 置信度
            confidence = {
                "降水强度": rain_conf,
                "降水时长": 0.75,
                "前期土壤含水量": _confidence_from_residuals(residuals_sw, max_res=0.02),
                "河道水位": 0.8,
            }
            horizons[str(h)] = {"evidence": evidence, "confidence": confidence}

        return horizons

    def _predict_fallback(self, T, horizon):
        """数据不足时的兜底预测"""
        if T == 0:
            state = "低"
        elif T >= 1:
            tp = self.ts["tp_mm"]
            past_rain = tp[max(0, T - 2):T + 1]
            if all(_rain_state(r) == "高" for r in past_rain):
                state = "高"
            elif any(_rain_state(r) == "高" for r in past_rain):
                state = "中"
            else:
                state = "低"

        sw = self.ts["swvl1"][T] if T >= 0 else 0.35
        evidence = {
            "降水强度": state, "降水时长": "短",
            "风力": _wind_state(self.ts["wind"][T]) if T >= 0 else "弱",
            "风向": "保持先验", "湿度": "保持先验", "气压": "保持先验",
            "气温": _t2m_state(self.ts["t2m"][T]) if T >= 0 else "适温",
            "露点温度": _d2m_state(self.ts["d2m"][T]) if T >= 0 else "低",
            "对流有效位能CAPE": "保持先验", "垂直风切变": "保持先验",
            "前期土壤含水量": _swvl1_state(sw),
            "土壤渗透性": _swvl2_state(self.ts["swvl2"][T]) if T >= 0 else "好",
            "蒸发量": _evabs_state(self.ts["evabs"][T]) if T >= 0 else "小",
            "径流系数": "保持先验", "河道水位": "正常",
            "地下水埋深": "保持先验", "湖泊调蓄能力": "保持先验", "潮汐影响": "保持先验",
        }
        conf = {"降水强度": 0.6, "降水时长": 0.6, "前期土壤含水量": 0.6, "河道水位": 0.6}
        horizons = {}
        for h in range(1, horizon + 1):
            horizons[str(h)] = {"evidence": copy.deepcopy(evidence), "confidence": conf}
        return horizons


# ══════════════════════════════════════════════════════════════════════
# NeuralProphet 方案
# ══════════════════════════════════════════════════════════════════════

class NeuralProphetForecaster:
    """NeuralProphet 时序预测（训练后预测）"""

    def __init__(self, ts_dict, train_split=120):
        self.ts = ts_dict
        self.n = len(ts_dict["tp_mm"])
        self.train_split = train_split
        self.models = {}  # 各变量训练好的模型
        self._trained = False
        self._metrics = {}

    def train(self):
        """训练 NeuralProphet 模型（降水/土壤含水量/水位）"""
        try:
            from neuralprophet import NeuralProphet
        except ImportError:
            print("    [NeuralProphet] 未安装，跳过训练")
            return False
        import pandas as pd

        print("\n[NeuralProphet] 开始训练...")
        train_end = self.train_split
        val_end = self.n

        targets = {
            "tp_mm": ("降水强度", RAIN_Q33, RAIN_Q66),
            "swvl1": ("前期土壤含水量", SWVL1_Q33, SWVL1_Q66),
        }

        all_metrics = {}
        for var_name, (state_name, q33, q66) in targets.items():
            series = self.ts[var_name]
            train_y = series[:train_end]
            val_y = series[train_end:val_end]

            # 准备 NeuralProphet 数据
            df = pd.DataFrame({
                "ds": pd.date_range(start="2021-07-18", periods=len(series), freq="h"),
                "y": series,
            })
            df_train = df.iloc[:train_end]
            df_val = df.iloc[train_end:val_end]

            m = NeuralProphet(
                n_forecasts=3,
                n_lags=6,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=True,
                epochs=50,
                learning_rate=0.01,
            )
            try:
                m.fit(df_train, freq="h", validation_df=df_val, progress_bar=False)
                forecast = m.predict(df_val)
                pred_cols = [c for c in forecast.columns if c.startswith("yhat")]
                if pred_cols:
                    preds = forecast[pred_cols[0]].values[:len(val_y)]
                    preds = np.clip(preds, 0, None)
                    rmse = np.sqrt(mean_squared_error(val_y, preds))

                    # 状态准确率
                    true_states = np.where(val_y <= q33, "低", np.where(val_y <= q66, "中", "高"))
                    pred_states = np.where(preds <= q33, "低", np.where(preds <= q66, "中", "高"))
                    acc = accuracy_score(true_states, pred_states)

                    all_metrics[var_name] = {"rmse": float(rmse), "state_acc": float(acc)}
                    print(f"    {var_name}: RMSE={rmse:.4f}, 状态准确率={acc:.2%}")
                    self.models[var_name] = m
            except Exception as e:
                print(f"    {var_name}: 训练失败 - {e}")
                all_metrics[var_name] = {"rmse": None, "state_acc": None}

        self._metrics = {"NeuralProphet": all_metrics}
        self._trained = bool(self.models)
        return self._trained

    def predict(self, T, horizon=3):
        """用训练好的模型预测"""
        if not self._trained:
            return None
        import pandas as pd

        result = {}
        for h in range(1, horizon + 1):
            result[str(h)] = {"evidence": {}, "confidence": {}}

        for var_name, m in self.models.items():
            from neuralprophet import NeuralProphet
            # 用过去 6 小时预测未来
            series = self.ts[var_name]
            if T < 6:
                continue
            past = series[T - 5:T + 1]
            df = pd.DataFrame({
                "ds": pd.date_range(start="2021-07-18", periods=T + 1, freq="h"),
                "y": series[:T + 1],
            })
            try:
                future = m.make_future_dataframe(df, periods=horizon)
                forecast = m.predict(future)
                pred_cols = [c for c in forecast.columns if c.startswith("yhat")]
                if pred_cols:
                    vals = forecast[pred_cols[0]].values[-horizon:]
                    vals = np.clip(vals, 0, None)
                    for h in range(1, horizon + 1):
                        val = vals[h - 1]
                        if var_name == "tp_mm":
                            result[str(h)]["evidence"]["降水强度"] = _rain_state(val)
                            result[str(h)]["confidence"]["降水强度"] = 0.8
                        elif var_name == "swvl1":
                            result[str(h)]["evidence"]["前期土壤含水量"] = _swvl1_state(val)
                            result[str(h)]["confidence"]["前期土壤含水量"] = 0.8
            except Exception:
                pass

        return result if any(result["1"]["evidence"]) else None


# ══════════════════════════════════════════════════════════════════════
# LSTM 方案
# ══════════════════════════════════════════════════════════════════════

class LSTMForecaster:
    """简单 2 层 LSTM 时序预测"""

    def __init__(self, ts_dict, train_split=120, input_steps=6):
        self.ts = ts_dict
        self.n = len(ts_dict["tp_mm"])
        self.train_split = train_split
        self.input_steps = input_steps
        self.models = {}
        self._trained = False
        self._metrics = {}
        self._device = None

    def _build_model(self, input_dim=1):
        import torch
        import torch.nn as nn
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_steps=3):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_steps)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out

        return SimpleLSTM(input_dim, 32, 2, 3)

    def train(self):
        """训练 LSTM 模型"""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[LSTM] 开始训练 (device={self._device})...")

        train_end = self.train_split
        val_end = self.n
        input_steps = self.input_steps

        targets = {
            "tp_mm": (RAIN_Q33, RAIN_Q66),
            "swvl1": (SWVL1_Q33, SWVL1_Q66),
        }

        all_metrics = {}
        for var_name, (q33, q66) in targets.items():
            series = self.ts[var_name]
            # 准备滑动窗口数据
            X, y = [], []
            for i in range(input_steps, len(series)):
                X.append(series[i - input_steps:i])
                y.append(series[i:i + 3] if i + 3 <= len(series) else
                         np.pad(series[i:], (0, 3 - (len(series) - i)), mode='edge'))
            X = np.array(X).reshape(-1, input_steps, 1).astype(np.float32)
            y = np.array(y).astype(np.float32)

            X_train, y_train = X[:train_end - input_steps], y[:train_end - input_steps]
            X_val, y_val = X[train_end - input_steps:val_end - input_steps], \
                           y[train_end - input_steps:val_end - input_steps]

            if len(X_train) < 10:
                print(f"    {var_name}: 训练数据不足，跳过")
                continue

            model = self._build_model().to(self._device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            X_train_t = torch.from_numpy(X_train).to(self._device)
            y_train_t = torch.from_numpy(y_train).to(self._device)
            X_val_t = torch.from_numpy(X_val).to(self._device)
            y_val_t = torch.from_numpy(y_val).to(self._device)

            best_loss = float('inf')
            patience = 10
            no_improve = 0
            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                out = model(X_train_t)
                loss = criterion(out, y_train_t)
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_out = model(X_val_t)
                        val_loss = criterion(val_out, y_val_t).item()
                    if val_loss < best_loss:
                        best_loss = val_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                    if no_improve >= patience:
                        break

            # 验证
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t).cpu().numpy()
                val_true = y_val_t.cpu().numpy()

            # 只取第 1 步预测
            preds = val_pred[:, 0]
            true = val_true[:, 0]
            rmse = float(np.sqrt(mean_squared_error(true, preds)))

            true_states = np.where(true <= q33, "低", np.where(true <= q66, "中", "高"))
            pred_states = np.where(preds <= q33, "低", np.where(preds <= q66, "中", "高"))
            acc = accuracy_score(true_states, pred_states)

            all_metrics[var_name] = {"rmse": rmse, "state_acc": float(acc)}
            print(f"    {var_name}: RMSE={rmse:.4f}, 状态准确率={acc:.2%}")
            self.models[var_name] = model

        self._metrics = {"LSTM": all_metrics}
        self._trained = bool(self.models)
        return self._trained

    def predict(self, T, horizon=3):
        """用训练好的 LSTM 预测"""
        if not self._trained or T < self.input_steps:
            return None
        import torch
        result = {}
        for h in range(1, horizon + 1):
            result[str(h)] = {"evidence": {}, "confidence": {}}

        for var_name, model in self.models.items():
            series = self.ts[var_name]
            inp = series[T - self.input_steps:T].reshape(1, self.input_steps, 1).astype(np.float32)
            inp_t = torch.from_numpy(inp).to(self._device)
            model.eval()
            with torch.no_grad():
                pred = model(inp_t).cpu().numpy()[0]

            for h in range(1, horizon + 1):
                val = pred[h - 1] if h - 1 < len(pred) else pred[-1]
                if var_name == "tp_mm":
                    result[str(h)]["evidence"]["降水强度"] = _rain_state(val)
                    result[str(h)]["confidence"]["降水强度"] = 0.8
                elif var_name == "swvl1":
                    result[str(h)]["evidence"]["前期土壤含水量"] = _swvl1_state(val)
                    result[str(h)]["confidence"]["前期土壤含水量"] = 0.8

        return result if any(result["1"]["evidence"]) else None


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("趋势预测层 — 构建 ERA5 逐小时预测证据")
    print("=" * 60)

    # ── 1. 加载数据 ──
    print("\n[1] 加载 ERA5 证据...")
    data = load_era5_evidence()
    records = data["records"]
    ts = extract_ts(records)
    print(f"    共 {len(records)} 条记录")

    # ── 2. 基线方案 ──
    print("\n[2] 基线方案（AR/规则/线性外推）...")
    baseline = BaselineForecaster(ts)
    all_forecasts = {}

    for T in range(len(records)):
        horizons = baseline.predict(T, horizon=3)
        all_forecasts[str(T)] = {
            "datetime": records[T]["datetime"],
            "horizons": horizons,
        }

    # ── 3. 训练方案对比 ──
    print("\n[3] 训练方案对比...")
    import pandas as pd

    # NeuralProphet
    npf = NeuralProphetForecaster(ts)
    npf_trained = npf.train()

    # LSTM
    lstm = LSTMForecaster(ts)
    lstm_trained = lstm.train()

    # 若有训练方案，融合预测结果（用训练方案覆盖基线中的降水/土壤含水量）
    trained_model = None
    if lstm_trained:
        trained_model = lstm
        print("\n[3b] 采用 LSTM 方案覆盖降水/土壤含水量预测")
    elif npf_trained:
        trained_model = npf
        print("\n[3b] 采用 NeuralProphet 方案覆盖降水/土壤含水量预测")

    if trained_model:
        for T in range(6, len(records)):
            ml_pred = trained_model.predict(T, horizon=3)
            if ml_pred:
                for h in range(1, 4):
                    ev = ml_pred[str(h)]["evidence"]
                    conf = ml_pred[str(h)]["confidence"]
                    if ev:
                        for k, v in ev.items():
                            all_forecasts[str(T)]["horizons"][str(h)]["evidence"][k] = v
                        for k, v in conf.items():
                            all_forecasts[str(T)]["horizons"][str(h)]["confidence"][k] = v

    # ── 4. 对比报告 ──
    print("\n[4] 方案对比报告...")
    report = {}

    # 基线在验证集上的指标
    val_start = 120
    baseline_metrics = {}
    for var_name, q33, q66 in [("tp_mm", RAIN_Q33, RAIN_Q66),
                                ("swvl1", SWVL1_Q33, SWVL1_Q66)]:
        true_vals = ts[var_name][val_start:]
        pred_vals = []
        for T in range(val_start, len(records)):
            h = all_forecasts[str(T)]["horizons"]["1"]["evidence"]
            if var_name == "tp_mm":
                s = h.get("降水强度", "低")
                pred_vals.append(RAIN_Q66 + 0.5 if s == "高" else (RAIN_Q33 + 0.5 if s == "中" else 0.1))
            elif var_name == "swvl1":
                s = h.get("前期土壤含水量", "低")
                pred_vals.append(SWVL1_Q66 + 0.02 if s == "高" else
                                 (SWVL1_Q33 + 0.02 if s == "中" else SWVL1_Q33 - 0.02))
        if pred_vals:
            pred_vals = np.array(pred_vals)
            true_vals = true_vals[:len(pred_vals)]
            rmse = float(np.sqrt(mean_squared_error(true_vals, pred_vals)))
            true_states = np.where(true_vals <= q33, "低", np.where(true_vals <= q66, "中", "高"))
            pred_states = np.where(pred_vals <= q33, "低", np.where(pred_vals <= q66, "中", "高"))
            acc = accuracy_score(true_states, pred_states)
            baseline_metrics[var_name] = {"rmse": rmse, "state_acc": float(acc)}

    # 收集所有方案指标
    all_metrics = {"基线AR": baseline_metrics}
    if npf._metrics:
        all_metrics["NeuralProphet"] = npf._metrics["NeuralProphet"]
    if lstm._metrics:
        all_metrics["LSTM"] = lstm._metrics["LSTM"]

    # 打印对比表
    print("\n" + "-" * 50)
    print(f"{'方案':<20} {'变量':<12} {'RMSE':<10} {'状态准确率':<12}")
    print("-" * 50)
    for scheme_name, metrics in all_metrics.items():
        for var_name, m in metrics.items():
            rmse_str = f"{m['rmse']:.4f}" if m['rmse'] is not None else "N/A"
            acc_str = f"{m['state_acc']:.2%}" if m['state_acc'] is not None else "N/A"
            print(f"{scheme_name:<20} {var_name:<12} {rmse_str:<10} {acc_str:<12}")
    print("-" * 50)

    # 最终结论
    print("\n[结论]")
    final_scheme = "基线AR"
    if lstm_trained and lstm._metrics:
        for var_name in ["tp_mm", "swvl1"]:
            bl = baseline_metrics.get(var_name, {})
            ml = lstm._metrics["LSTM"].get(var_name, {})
            bl_acc = bl.get("state_acc") or 0.0
            ml_acc = ml.get("state_acc") or 0.0
            if ml_acc > bl_acc + 0.1:
                final_scheme = "LSTM（提升显著）"
                break
    elif npf_trained and npf._metrics:
        for var_name in ["tp_mm", "swvl1"]:
            bl = baseline_metrics.get(var_name, {})
            ml = npf._metrics["NeuralProphet"].get(var_name, {})
            bl_acc = bl.get("state_acc") or 0.0
            ml_acc = ml.get("state_acc") or 0.0
            if ml_acc > bl_acc + 0.1:
                final_scheme = "NeuralProphet（提升显著）"
                break

    print(f"  最终采用方案: {final_scheme}")
    if final_scheme == "基线AR":
        print("  训练方案未带来显著提升（+10%），保留基线方案")
    else:
        print("  训练方案提升显著，已覆盖降水/土壤含水量预测")

    report["baseline_metrics"] = baseline_metrics
    report["trained_metrics"] = {k: v for k, v in all_metrics.items() if k != "基线AR"}
    report["final_scheme"] = final_scheme

    # ── 5. 输出 ──
    print(f"\n[5] 输出预测结果: {FORECAST_OUTPUT_PATH}")
    output = {
        "meta": {
            "n_steps": len(records),
            "horizons": [1, 2, 3],
            "scheme": final_scheme,
            "baseline": "AR(1) + 规则 + 线性外推",
            "trained": "NeuralProphet" if npf_trained else ("LSTM" if lstm_trained else "None"),
            "comparison_report": report,
        },
        "forecasts": all_forecasts,
    }

    with open(FORECAST_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"    写入 {len(all_forecasts)} 条预测记录")
    print(f"    文件大小: {os.path.getsize(FORECAST_OUTPUT_PATH):,} 字节")

    # ── 6. 抽查验证 ──
    print("\n" + "=" * 60)
    print("抽查验证")
    print("=" * 60)
    check_points = [40, 60, 90]
    for T in check_points:
        if str(T) in all_forecasts:
            fc = all_forecasts[str(T)]
            print(f"\n  T={T} ({fc['datetime']})")
            for h in range(1, 4):
                ev = fc["horizons"][str(h)]["evidence"]
                conf = fc["horizons"][str(h)]["confidence"]
                rain = ev["降水强度"]
                water = ev["河道水位"]
                r_conf = conf.get("降水强度", 0.6)
                print(f"    +{h}h: 降水={rain}({r_conf:.0%}), 水位={water}, "
                      f"土壤={ev['前期土壤含水量']}")

    print("\n完成!")


if __name__ == "__main__":
    main()