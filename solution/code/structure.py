"""数据结构与生成机制专项分析。

核心发现：
1. 20 个指标近似可交换（均值/方差几乎相同）；
2. 指标两两弱负相关，且“在给定行总和 N 的窄区间内”平均相关≈-1/19，
   与“把 N 个计数单元多项分配到 20 个指标”的机制一致；
3. 行总和 N 本身呈多峰分布（GMM 支持 2–3 个成分）；
4. 目标 y 几乎只由 N 决定：y≈aN+b+噪声，R²≈0.845，噪声 std≈0.020；
5. 预测信号（aN+b）是多峰的，观测 y 因噪声叠加才接近正态。

产出：
- results/structure.json
- figures/fig_structure_rowsum.png
- figures/fig_structure_corr.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

from common import (
    FIGURES, RESULTS, FEATURES, TARGET, Timer, load_train,
    save_json, rmse, r2_score, setup_plot,
)

RNG = np.random.default_rng(42)


def main() -> None:
    setup_plot()
    import matplotlib.pyplot as plt

    with Timer("读取数据"):
        tr = load_train()
    X = tr[FEATURES].to_numpy(dtype=float)
    y = tr[TARGET].to_numpy(dtype=float)
    N = X.sum(axis=1)

    # ---------- 1. 可交换性 ----------
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    exchange = {
        "feature_mean_min": float(means.min()),
        "feature_mean_max": float(means.max()),
        "feature_mean_range_over_mean": float((means.max() - means.min()) / means.mean()),
        "feature_std_min": float(stds.min()),
        "feature_std_max": float(stds.max()),
        "feature_std_range_over_mean": float((stds.max() - stds.min()) / stds.mean()),
    }

    # ---------- 2. 相关结构 ----------
    C = np.corrcoef(X, rowvar=False)
    off = C[~np.eye(len(FEATURES), dtype=bool)]
    corr_struct = {
        "mean_pairwise_r": float(off.mean()),
        "min_pairwise_r": float(off.min()),
        "max_pairwise_r": float(off.max()),
        "all_negative": bool(np.all(off < 0)),
    }

    # 给定 N 窄区间内的平均两两相关（与多项分配的 -1/19 比较）
    bin_rows = []
    edges = np.arange(80, 130, 5)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (N >= lo) & (N < hi)
        if m.sum() < 5000:
            continue
        Z = X[m]
        Cz = np.corrcoef(Z, rowvar=False)
        offz = Cz[~np.eye(len(FEATURES), dtype=bool)]
        bin_rows.append({
            "N_low": int(lo), "N_high": int(hi),
            "n": int(m.sum()),
            "mean_N": float(N[m].mean()),
            "mean_pairwise_r": float(offz.mean()),
            "mean_feature_var": float(Z.var(axis=0, ddof=1).mean()),
            "expected_var_if_multinomial": float(N[m].mean() / 20 * 19 / 20),
        })
    bin_tab = pd.DataFrame(bin_rows)
    structure_bins = {
        "bins": bin_rows,
        "multinomial_reference_corr": -1.0 / 19.0,
    }

    # ---------- 3. 行总和分布 / GMM ----------
    with Timer("GMM on row sum"):
        s = 200_000
        idx = RNG.choice(len(N), size=s, replace=False)
        Ns = N[idx].reshape(-1, 1)
        gmm = {}
        for k in (1, 2, 3, 4):
            g = GaussianMixture(k, n_init=3, random_state=42).fit(Ns)
            gmm[k] = {
                "bic": float(g.bic(Ns)),
                "means": sorted(g.means_.ravel().tolist()),
                "weights": sorted(g.weights_.tolist(), reverse=True),
                "stds": sorted(np.sqrt(g.covariances_.ravel()).tolist(), reverse=True),
            }
    best_k = min(gmm, key=lambda k: gmm[k]["bic"])
    rowsum_struct = {
        "mean": float(N.mean()),
        "std": float(N.std(ddof=1)),
        "min": float(N.min()),
        "max": float(N.max()),
        "median": float(np.median(N)),
        "kurtosis": float(stats.kurtosis(N)),
        "skew": float(stats.skew(N)),
        "gmm": gmm,
        "best_k_by_bic": int(best_k),
    }

    # ---------- 4. 目标 = f(N) + 噪声 ----------
    A = np.vstack([N, np.ones_like(N)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    signal = a * N + b
    noise = y - signal
    target_struct = {
        "signal_slope": float(a),
        "signal_intercept": float(b),
        "r2": float(r2_score(y, signal)),
        "rmse": float(rmse(y, signal)),
        "signal_std": float(signal.std(ddof=1)),
        "noise_std": float(noise.std(ddof=1)),
        "y_std": float(y.std(ddof=1)),
        "noise_share_of_variance": float(noise.var(ddof=1) / y.var(ddof=1)),
        "noise_corr_with_N": float(np.corrcoef(noise, N)[0, 1]),
    }

    # ---------- 5. 预测信号的多峰性 ----------
    pred_struct = {
        "pred_skew": float(stats.skew(signal)),
        "pred_kurtosis": float(stats.kurtosis(signal)),
        "y_skew": float(stats.skew(y)),
        "y_kurtosis": float(stats.kurtosis(y)),
        "pred_normal_ks_D": float(stats.kstest(
            signal, stats.norm(signal.mean(), signal.std(ddof=1)).cdf
        ).statistic),
        "y_normal_ks_D": float(stats.kstest(
            y, stats.norm(y.mean(), y.std(ddof=1)).cdf
        ).statistic),
    }

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].hist(N, bins=np.arange(N.min(), N.max() + 1, 1),
                 color="#4C78A8", alpha=0.75, density=True)
    xs = np.linspace(N.min(), N.max(), 600)
    for k in (2, 3):
        g = GaussianMixture(k, n_init=3, random_state=42).fit(Ns)
        comp = np.zeros_like(xs)
        for w, mu, var in zip(g.weights_, g.means_.ravel(), g.covariances_.ravel()):
            comp += w * stats.norm.pdf(xs, mu, np.sqrt(var))
        axes[0].plot(xs, comp, lw=1.2, label=f"GMM k={k}, BIC={gmm[k]['bic']:.2e}")
    axes[0].set_xlim(70, 130)
    axes[0].set_xlabel("行总和 N = Σ 20 个指标")
    axes[0].set_ylabel("密度")
    axes[0].set_title("行总和本身是多峰分布")
    axes[0].legend(fontsize=8)

    axes[1].scatter(N[::200], y[::200], s=1, alpha=0.15, color="#4C78A8")
    axes[1].plot(xs, a * xs + b, "r--", lw=1.5,
                 label=f"y={a:.5f}N{b:+.4f}, R²={r2_score(y, signal):.4f}")
    axes[1].set_xlabel("行总和 N")
    axes[1].set_ylabel("洪水概率")
    axes[1].set_title("目标几乎只由行总和决定，散点宽度=噪声")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_structure_rowsum.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(bin_tab["mean_N"], bin_tab["mean_pairwise_r"], "o-",
            color="#E45756", label="给定 N 窄区间内的平均两两相关")
    ax.axhline(-1 / 19, color="k", ls="--", lw=1,
               label="多项分配理论值 −1/19≈−0.0526")
    ax.set_xlabel("行总和 N 的区间中心")
    ax.set_ylabel("平均两两相关系数")
    ax.set_title("控制 N 后指标呈负相关：与“总量-多项分配”机制一致")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_structure_corr.png")
    plt.close(fig)

    out = {
        "exchangeability": exchange,
        "correlation_structure": corr_struct,
        "within_N_bins": structure_bins,
        "row_sum": rowsum_struct,
        "target_decomposition": target_struct,
        "multimodality": pred_struct,
        "conclusion": (
            "20 个指标近似可交换；控制行总和 N 后两两相关≈−1/19，"
            "符合“把 N 个计数单元多项分配到 20 个指标”的生成机制；"
            "N 本身多峰，目标 y≈0.00564676N−0.05335+噪声，R²≈0.845。"
            "因此指标之间只存在“总量守恒”的负相关，不存在真正的风险因子结构。"
        ),
    }
    save_json(out, RESULTS / "structure.json")

    print("可交换性:", exchange)
    print("相关结构:", corr_struct)
    print("GMM best k:", best_k, "means:", gmm[best_k]["means"])
    print("目标分解:", {k: round(v, 6) for k, v in target_struct.items()})
    print("多峰性:", {k: round(v, 5) for k, v in pred_struct.items()})


if __name__ == "__main__":
    main()
