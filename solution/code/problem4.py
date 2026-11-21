"""问题 4：预测 test.csv 的洪水概率、写回 submit.csv、分析预测分布。

产出：
- results/submit.csv（保持原文件 GBK/CRLF/列结构）
- results/problem4.json
- figures/fig_p4_hist.png
- figures/fig_p4_qq.png
- figures/fig_p4_line.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    FIGURES, RESULTS, DATA, FEATURES, TARGET, Timer, load_train, load_test,
    save_json, setup_plot, rmse, r2_score,
)

RNG = np.random.default_rng(42)


def write_submit(pred: np.ndarray, test_ids: np.ndarray, out_path: Path) -> dict:
    """按行替换 submit.csv 的概率列，保留 GBK、CRLF、末尾空列。"""
    raw = DATA / "submit.csv"
    lines = raw.read_text(encoding="gbk", newline="").splitlines(keepends=True)
    header = lines[0]
    body = lines[1:]
    pred_map = {int(i): float(v) for i, v in zip(test_ids, pred)}

    seen = 0
    new_lines = [header]
    for line in body:
        # 保留原始换行符
        if line.endswith("\r\n"):
            core, nl = line[:-2], "\r\n"
        elif line.endswith("\n"):
            core, nl = line[:-1], "\n"
        else:
            core, nl = line, ""
        parts = core.split(",")
        rid = int(parts[0])
        parts[1] = f"{pred_map[rid]:.6f}"
        new_lines.append(",".join(parts) + nl)
        seen += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(new_lines), encoding="gbk", newline="")
    return {
        "rows_written": seen,
        "rows_expected": len(body),
        "ids_matched": seen == len(pred_map) == len(body),
        "header": header.strip(),
    }


def hist_modes(v: np.ndarray, bins: int = 300, smooth: float = 3.0) -> list[float]:
    """对直方图做高斯平滑后找局部极大值，用于描述多峰性。"""
    from scipy.ndimage import gaussian_filter1d
    h, e = np.histogram(v, bins=bins, density=True)
    sm = gaussian_filter1d(h, smooth)
    centers = (e[:-1] + e[1:]) / 2
    return [float(centers[i]) for i in range(1, len(sm) - 1)
            if sm[i] > sm[i - 1] and sm[i] > sm[i + 1]]


def hist_modes(v: np.ndarray, bins: int = 300, smooth: float = 3.0) -> list[float]:
    """对直方图做高斯平滑后找局部极大值，用于描述多峰性。"""
    from scipy.ndimage import gaussian_filter1d
    h, e = np.histogram(v, bins=bins, density=True)
    sm = gaussian_filter1d(h, smooth)
    centers = (e[:-1] + e[1:]) / 2
    return [float(centers[i]) for i in range(1, len(sm) - 1)
            if sm[i] > sm[i - 1] and sm[i] > sm[i + 1]]


def main() -> None:
    setup_plot()
    import matplotlib.pyplot as plt

    with Timer("读取数据与模型"):
        tr = load_train()
        te = load_test()
        model = np.load(RESULTS / "final_model.npz", allow_pickle=True)
        coef = model["coef"]
        intercept = float(model["intercept"])

    Xte = te[FEATURES].to_numpy(dtype=float)
    test_ids = te["id"].to_numpy()

    with Timer("预测"):
        pred = Xte @ coef + intercept

    # ---------- 写回 submit.csv ----------
    info = write_submit(pred, test_ids, RESULTS / "submit.csv")
    # 同时放一份到 solution 根目录，便于直接提交
    write_submit(pred, test_ids, RESULTS.parent / "submit.csv")

    # ---------- 分布统计 ----------
    pred_stats = {
        "n": int(len(pred)),
        "mean": float(pred.mean()),
        "std": float(pred.std(ddof=1)),
        "min": float(pred.min()),
        "max": float(pred.max()),
        "median": float(np.median(pred)),
        "q01": float(np.quantile(pred, 0.01)),
        "q99": float(np.quantile(pred, 0.99)),
        "nunique_rounded_6": int(len(np.unique(np.round(pred, 6)))),
    }

    # ---------- 正态性检验 ----------
    mu, sd = pred.mean(), pred.std(ddof=1)
    ks = stats.kstest(pred, stats.norm(mu, sd).cdf)
    ad = stats.anderson(pred, dist="norm")
    jb = stats.jarque_bera(pred)
    sample = RNG.choice(pred, size=5000, replace=False)
    sh = stats.shapiro(sample)
    ks_vs_train = stats.ks_2samp(pred, tr[TARGET].to_numpy(dtype=float))
    y_train = tr[TARGET].to_numpy(dtype=float)
    ks_y = stats.kstest(y_train, stats.norm(y_train.mean(), y_train.std(ddof=1)).cdf)
    normality = {
        "skew": float(stats.skew(pred)),
        "kurtosis_excess": float(stats.kurtosis(pred)),
        "ks_D_vs_fitted_normal": float(ks.statistic),
        "ks_p_vs_fitted_normal": float(ks.pvalue),
        "anderson_stat": float(ad.statistic),
        "anderson_critical_5pct": float(ad.critical_values[2]),
        "jarque_bera_stat": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
        "shapiro_W_5000": float(sh.statistic),
        "shapiro_p_5000": float(sh.pvalue),
        "ks_D_vs_train_target": float(ks_vs_train.statistic),
        "ks_p_vs_train_target": float(ks_vs_train.pvalue),
        "pred_modes": hist_modes(pred),
        "train_target": {
            "skew": float(stats.skew(y_train)),
            "kurtosis_excess": float(stats.kurtosis(y_train)),
            "ks_D_vs_fitted_normal": float(ks_y.statistic),
            "ks_p_vs_fitted_normal": float(ks_y.pvalue),
            "modes": hist_modes(y_train),
        },
    }

    # ---------- 图 ----------
    # 直方图 + 核密度 + 正态拟合（预测 vs 观测目标）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)
    for ax, v, title in [
        (axes[0], pred, "预测概率（信号）"),
        (axes[1], y_train, "训练集观测概率（信号+噪声）"),
    ]:
        ax.hist(v, bins=80, density=True, color="#4C78A8", alpha=0.55,
                label="直方图")
        kde_sample = RNG.choice(v, size=min(50_000, len(v)), replace=False)
        kde = stats.gaussian_kde(kde_sample)
        xs = np.linspace(v.min(), v.max(), 500)
        ax.plot(xs, kde(xs), color="#54A24B", lw=1.6, label="核密度估计")
        ax.plot(xs, stats.norm.pdf(xs, v.mean(), v.std(ddof=1)), "r--", lw=1.5,
                label=f"正态拟合 μ={v.mean():.4f}, σ={v.std(ddof=1):.4f}")
        ax.set_xlabel("洪水概率")
        ax.set_ylabel("密度")
        ax.set_title(title)
        ax.legend(fontsize=7.5)
    axes[0].text(0.02, 0.95, f"峰度={stats.kurtosis(pred):.2f}\n"
                 f"KS D={ks.statistic:.4f}",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    axes[1].text(0.02, 0.95, f"峰度={stats.kurtosis(y_train):.2f}\n"
                 f"KS D={ks_y.statistic:.4f}",
                 transform=axes[1].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.suptitle("预测分布是多峰的；观测目标因噪声叠加才接近正态", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p4_hist.png")
    plt.close(fig)

    # Q-Q
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    stats.probplot(pred, dist="norm", plot=ax)
    ax.set_title("预测概率的 Q-Q 图")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p4_qq.png")
    plt.close(fig)

    # 折线图：按 id 排序分箱
    order = np.argsort(test_ids)
    nbin = 500
    edges = np.linspace(0, len(order), nbin + 1).astype(int)
    xb, mb, lo, hi = [], [], [], []
    for b in range(nbin):
        seg = pred[order[edges[b]:edges[b + 1]]]
        xb.append(np.median(test_ids[order[edges[b]:edges[b + 1]]]))
        mb.append(seg.mean())
        lo.append(np.quantile(seg, 0.05))
        hi.append(np.quantile(seg, 0.95))
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.plot(xb, mb, color="#E45756", lw=1.2, label="分箱均值")
    ax.fill_between(xb, lo, hi, color="#E45756", alpha=0.18,
                    label="5%–95% 分位带")
    ax.set_xlabel("事件 id（按 id 排序后分 500 箱）")
    ax.set_ylabel("预测洪水概率")
    ax.set_title("预测概率随 id 的折线图（分箱统计）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p4_line.png")
    plt.close(fig)

    out = {
        "prediction_stats": pred_stats,
        "normality": normality,
        "submit": info,
        "conclusion": (
            "预测概率均值约 0.504、标准差约 0.047。预测分布并不严格服从正态："
            "峰度约 0.84、KS D≈0.073，直方图/平滑后可见多个峰，其根源是 20 个指标"
            "的“行总和”本身是多峰分布。训练集观测概率因为叠加了标准差约 0.020 的"
            "噪声，峰度降到约 −0.04、KS D≈0.028，才在实用意义上接近正态。"
            "因此对“是否服从正态分布”的回答应是：观测值近似正态，预测信号本身非正态。"
        ),
    }
    save_json(out, RESULTS / "problem4.json")

    print("预测统计:", {k: round(v, 6) if isinstance(v, float) else v
                          for k, v in pred_stats.items()})
    print("正态性:", {k: round(v, 6) if isinstance(v, float) else v
                      for k, v in normality.items()})
    print("提交文件:", info)


if __name__ == "__main__":
    main()
