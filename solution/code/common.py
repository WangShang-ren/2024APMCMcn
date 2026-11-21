"""公共工具：路径、数据读取、指标、绘图风格、日志。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# 原始附件只保留 B 题目录下的一份；solution/data 若存在则优先使用（本地覆盖）
DATA = ROOT / "data"
if not DATA.is_dir():
    DATA = (ROOT.parent / "B题 洪水灾害的数据分析与预测"
            / "附件 B题 洪水灾害的数据分析与预测")
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
LOGS = RESULTS / "logs"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"

for _d in (RESULTS, TABLES, LOGS, FIGURES, PAPER):
    _d.mkdir(parents=True, exist_ok=True)

# 附件列顺序（id + 20 个指标 + 目标）
FEATURES = [
    "季风强度", "地形排水", "河流管理", "森林砍伐", "城市化",
    "气候变化", "大坝质量", "淤积", "农业实践", "侵蚀",
    "无效防灾", "排水系统", "海岸脆弱性", "滑坡", "流域",
    "基础设施恶化", "人口得分", "湿地损失", "规划不足", "政策因素",
]
TARGET = "洪水概率"


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA / "train.csv", encoding="gbk")


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA / "test.csv", encoding="gbk")


def load_submit() -> pd.DataFrame:
    return pd.read_csv(DATA / "submit.csv", encoding="gbk")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def setup_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["font.size"] = 10


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        print(f"[start] {self.name}", flush=True)
        return self

    def __exit__(self, *exc):
        self.dt = time.time() - self.t0
        print(f"[done ] {self.name}: {self.dt:.1f}s", flush=True)


def log_line(msg: str, path: Path | None = None) -> None:
    print(msg, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
