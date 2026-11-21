"""问题 0：数据总览与探索性分析（EDA）。

产出：
- results/eda_summary.json
- results/tables/eda_features.csv
- results/tables/eda_correlations.csv
- figures/fig_target_hist.png
- figures/fig_corr_heatmap.png
- figures/fig_feature_hist.png
- figures/fig_residual.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    FIGURES, TABLES, RESULTS, FEATURES, TARGET, Timer, load_train, load_test,
    save_json, rmse, r2_score, setup_plot,
)

RNG = np.random.default_rng(42)


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """相关系数的 Fisher z 置信区间。"""
    if n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
    return float(lo), float(hi)


def main() -> None:
    setup_plot()
    import matplotlib.pyplot as plt

    with Timer("读取数据"):
        tr = load_train()
        te = load_test()

    X = tr[FEATURES].to_numpy(dtype=float)
    y = tr[TARGET].to_numpy(dtype=float)
    Xte = te[FEATURES].to_numpy(dtype=float)
    n = len(tr)

    summary: dict = {
        "train_shape": list(tr.shape),
        "test_shape": list(te.shape),
        "n_features": len(FEATURES),
        "missing_train": int(tr.isna().sum().sum()),
        "missing_test_features": int(te[FEATURES].isna().sum().sum()),
        "test_target_missing_by_design": int(te[TARGET].isna().sum()),
        "target_stats": {
            k: float(v) for k, v in pd.Series(y).describe().to_dict().items()
        },
        "target_nunique": int(pd.Series(y).nunique()),
        "target_skew": float(stats.skew(y)),
        "target_kurtosis": float(stats.kurtosis(y)),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "feature_value_min": float(X.min()),
        "feature_value_max": float(X.max()),
        "id_train_range": [int(tr["id"].min()), int(tr["id"].max())],
        "id_test_range": [int(te["id"].min()), int(te["id"].max())],
        "id_overlap": int(len(set(tr["id"]) & set(te["id"]))),
    }

    # ---------- 1. 指标描述统计 ----------
    rows = []
    for j, c in enumerate(FEATURES):
        s = X[:, j]
        rows.append({
            "指标": c,
            "min": float(s.min()), "max": float(s.max()),
            "mean": float(s.mean()), "std": float(s.std(ddof=1)),
            "median": float(np.median(s)),
            "unique": int(len(np.unique(s))),
            "skew": float(stats.skew(s)),
            "kurtosis": float(stats.kurtosis(s)),
        })
    feat_tab = pd.DataFrame(rows)
    feat_tab.to_csv(TABLES / "eda_features.csv", index=False, encoding="utf-8-sig")
    summary["feature_mean_min"] = float(feat_tab["mean"].min())
    summary["feature_mean_max"] = float(feat_tab["mean"].max())
    summary["feature_std_min"] = float(feat_tab["std"].min())
    summary["feature_std_max"] = float(feat_tab["std"].max())

    # ---------- 2. 指标间相关 ----------
    C = np.corrcoef(X, rowvar=False)
    off = C[~np.eye(len(FEATURES), dtype=bool)]
    summary["feature_corr_max_abs"] = float(np.max(np.abs(off)))
    summary["feature_corr_mean_abs"] = float(np.mean(np.abs(off)))

    # ---------- 3. 单变量相关（Pearson / Spearman，含 95% CI） ----------
    corr_rows = []
    for j, c in enumerate(FEATURES):
        pear = float(np.corrcoef(X[:, j], y)[0, 1])
        spear = float(stats.spearmanr(X[:, j], y).statistic)
        lo, hi = fisher_ci(pear, n)
        corr_rows.append({
            "指标": c, "pearson_r": pear, "ci_low": lo, "ci_high": hi,
            "r2": pear ** 2, "spearman_rho": spear,
        })
    corr_tab = pd.DataFrame(corr_rows).sort_values("pearson_r", ascending=False)
    corr_tab.to_csv(TABLES / "eda_correlations.csv", index=False, encoding="utf-8-sig")
    summary["univariate_r_min"] = float(corr_tab["pearson_r"].min())
    summary["univariate_r_max"] = float(corr_tab["pearson_r"].max())
    summary["univariate_r2_min"] = float(corr_tab["r2"].min())
    summary["univariate_r2_max"] = float(corr_tab["r2"].max())

    # ---------- 4. 线性可解释性 / 等权结构 ----------
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    intercept = float(y.mean() - X.mean(axis=0) @ coef)
    pred = X @ coef + intercept
    resid = y - pred
    summary["ols"] = {
        "intercept": intercept,
        "coef_mean": float(coef.mean()),
        "coef_std": float(coef.std()),
        "coef_min": float(coef.min()),
        "coef_max": float(coef.max()),
        "r2_train": float(r2_score(y, pred)),
        "rmse_train": float(rmse(y, pred)),
        "resid_std": float(resid.std(ddof=1)),
        "resid_min": float(resid.min()),
        "resid_max": float(resid.max()),
        "resid_within_0p035": float(np.mean(np.abs(resid) <= 0.035)),
        "resid_corr_max_abs": float(np.max(np.abs([
            np.corrcoef(resid, X[:, j])[0, 1] for j in range(len(FEATURES))
        ]))),
    }

    # 等权平均模型：y ~ a * mean(X) + b
    xbar = X.mean(axis=1)
    A = np.vstack([xbar, np.ones_like(xbar)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred_eq = a * xbar + b
    summary["equal_weight"] = {
        "slope": float(a), "intercept": float(b),
        "r2": float(r2_score(y, pred_eq)), "rmse": float(rmse(y, pred_eq)),
        "corr_with_target": float(np.corrcoef(pred_eq, y)[0, 1]),
    }

    # 行总和结构：目标是否只依赖 20 个指标的总量
    row_sum = X.sum(axis=1)
    (a_rs, b_rs), *_ = np.linalg.lstsq(
        np.vstack([row_sum, np.ones_like(row_sum)]).T, y, rcond=None
    )
    pred_rs = a_rs * row_sum + b_rs
    summary["row_sum_structure"] = {
        "mean": float(row_sum.mean()),
        "std": float(row_sum.std(ddof=1)),
        "min": float(row_sum.min()),
        "max": float(row_sum.max()),
        "median": float(np.median(row_sum)),
        "corr_with_target": float(np.corrcoef(row_sum, y)[0, 1]),
        "r2_with_target": float(r2_score(y, pred_rs)),
        "rmse_with_target": float(rmse(y, pred_rs)),
        "slope": float(a_rs), "intercept": float(b_rs),
        "mean_pairwise_corr": float(np.mean(off)),
        "var_sum_over_sum_var": float(row_sum.var(ddof=1) / X.var(axis=0, ddof=1).sum()),
    }

    # ---------- 5. 自助法：相关系数排名的稳定性 ----------
    B = 200
    m = 50_000
    boot_r = np.empty((B, len(FEATURES)))
    for b_i in range(B):
        idx = RNG.choice(n, size=m, replace=False)
        Xb, yb = X[idx], y[idx]
        Xb_c = Xb - Xb.mean(axis=0)
        yb_c = yb - yb.mean()
        boot_r[b_i] = (Xb_c * yb_c[:, None]).sum(axis=0) / np.sqrt(
            (Xb_c ** 2).sum(axis=0) * (yb_c ** 2).sum()
        )
    ranks = np.argsort(np.argsort(-boot_r, axis=1), axis=1) + 1  # 1 = 最强
    rank_tab = pd.DataFrame({
        "指标": FEATURES,
        "rank_mean": ranks.mean(axis=0),
        "rank_p2.5": np.percentile(ranks, 2.5, axis=0),
        "rank_p97.5": np.percentile(ranks, 97.5, axis=0),
        "r_boot_mean": boot_r.mean(axis=0),
        "r_boot_std": boot_r.std(axis=0),
    }).sort_values("rank_mean")
    rank_tab.to_csv(TABLES / "eda_rank_stability.csv", index=False, encoding="utf-8-sig")
    summary["rank_stability"] = {
        "best_mean_rank": float(rank_tab["rank_mean"].min()),
        "worst_mean_rank": float(rank_tab["rank_mean"].max()),
        "rank_interval_width_mean": float(
            (rank_tab["rank_p97.5"] - rank_tab["rank_p2.5"]).mean()
        ),
    }

    # ---------- 6. 训练/测试分布一致性 ----------
    ks = []
    for j, c in enumerate(FEATURES):
        res = stats.ks_2samp(X[:, j], Xte[:, j])
        ks.append({"指标": c, "ks_stat": float(res.statistic), "p_value": float(res.pvalue)})
    ks_tab = pd.DataFrame(ks)
    ks_tab.to_csv(TABLES / "eda_train_test_ks.csv", index=False, encoding="utf-8-sig")
    summary["train_test_ks_max"] = float(ks_tab["ks_stat"].max())
    summary["train_test_mean_diff_max"] = float(np.max(np.abs(X.mean(0) - Xte.mean(0))))

    # ---------- 7. 图 ----------
    # 目标分布
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(y, bins=120, color="#4C78A8", alpha=0.85, density=True)
    mu, sd = y.mean(), y.std()
    xs = np.linspace(y.min(), y.max(), 400)
    axes[0].plot(xs, stats.norm.pdf(xs, mu, sd), "r--", lw=1.5,
                 label=f"正态拟合 μ={mu:.4f}, σ={sd:.4f}")
    axes[0].set_title("洪水概率分布")
    axes[0].set_xlabel("洪水概率")
    axes[0].set_ylabel("密度")
    axes[0].legend(fontsize=8)
    stats.probplot(y, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q 图（正态）")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_target_hist.png")
    plt.close(fig)

    # 指标相关热图
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=90, fontsize=8)
    ax.set_yticklabels(FEATURES, fontsize=8)
    ax.set_title("20 个指标之间的相关系数（最大 |r|≈0.017）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_corr_heatmap.png")
    plt.close(fig)

    # 代表指标分布
    show = ["季风强度", "地形排水", "基础设施恶化", "政策因素", "大坝质量", "侵蚀"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, c in zip(axes.ravel(), show):
        ax.hist(X[:, FEATURES.index(c)], bins=np.arange(-0.5, 19.5, 1),
                color="#59A14F", alpha=0.8, rwidth=0.85)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("指标得分")
    fig.suptitle("代表指标的取值分布（0–18 的离散计数型）", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_feature_hist.png")
    plt.close(fig)

    # 残差诊断
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(resid, bins=200, color="#E45756", alpha=0.8, density=True)
    axes[0].set_title(f"线性模型残差（std={resid.std():.4f}）")
    axes[0].set_xlabel("残差")
    axes[0].set_ylabel("密度")
    axes[1].scatter(pred[::200], resid[::200], s=1, alpha=0.2, color="#4C78A8")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("残差 vs 拟合值")
    axes[1].set_xlabel("拟合值")
    axes[1].set_ylabel("残差")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_residual.png")
    plt.close(fig)

    # 行总和 vs 目标（核心结构）
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    hb = axes[0].hexbin(row_sum, y, gridsize=55, cmap="viridis", mincnt=1)
    xs = np.linspace(row_sum.min(), row_sum.max(), 200)
    axes[0].plot(xs, a_rs * xs + b_rs, "r--", lw=1.4,
                 label=f"y={a_rs:.5f}·Σx{b_rs:+.4f}，R²={r2_score(y, pred_rs):.4f}")
    axes[0].set_xlabel("20 个指标之和 Σx")
    axes[0].set_ylabel("洪水概率")
    axes[0].set_title("目标几乎只由指标总和决定")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.colorbar(hb, ax=axes[0], fraction=0.046, pad=0.03)

    axes[1].hist(off, bins=70, color="#F58518", alpha=0.85)
    axes[1].axvline(0, color="k", lw=1)
    axes[1].set_xlabel("指标两两相关系数 r")
    axes[1].set_ylabel("频数")
    axes[1].set_title(f"两两相关均为弱负值（均值 {off.mean():.4f}）")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_rowsum.png")
    plt.close(fig)

    summary["figures"] = [
        "fig_target_hist.png", "fig_corr_heatmap.png",
        "fig_feature_hist.png", "fig_residual.png", "fig_rowsum.png",
    ]
    save_json(summary, RESULTS / "eda_summary.json")
    print("EDA 完成，关键结论：")
    print(f"  R²(线性) = {summary['ols']['r2_train']:.4f}, "
          f"噪声 std = {summary['ols']['resid_std']:.4f}")
    print(f"  单变量 r ∈ [{summary['univariate_r_min']:.4f}, "
          f"{summary['univariate_r_max']:.4f}]")
    print(f"  等权模型 R² = {summary['equal_weight']['r2']:.4f}")


if __name__ == "__main__":
    main()
