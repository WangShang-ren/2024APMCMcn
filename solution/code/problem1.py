"""问题 1：20 个指标与洪水发生概率的关联分析。

方法：
1. Pearson / Spearman 相关（含 Fisher 95% CI）
2. 互信息（离散特征 -> 连续目标，抽样估计）
3. 线性模型置换重要性（RMSE 增量）
4. 留一指标 R² 下降（drop-one）
5. 自助法排名稳定性（来自 EDA）

产出：
- results/problem1.json
- results/tables/p1_importance.csv
- figures/fig_p1_corr_ci.png
- figures/fig_p1_rank_interval.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from common import (
    FIGURES, TABLES, RESULTS, FEATURES, TARGET, Timer, load_train,
    save_json, rmse, r2_score, setup_plot,
)

RNG = np.random.default_rng(42)


def ols_fit(X: np.ndarray, y: np.ndarray):
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    intercept = float(y.mean() - X.mean(axis=0) @ coef)
    pred = X @ coef + intercept
    return coef, intercept, pred


def main() -> None:
    setup_plot()
    import matplotlib.pyplot as plt

    with Timer("读取数据"):
        tr = load_train()
    X = tr[FEATURES].to_numpy(dtype=float)
    y = tr[TARGET].to_numpy(dtype=float)
    n, p = X.shape

    # ---------- 基线线性模型 ----------
    coef, intercept, pred = ols_fit(X, y)
    r2_full = r2_score(y, pred)
    rmse_full = rmse(y, pred)

    # ---------- 1. 相关 ----------
    corr = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(p)])
    spear = np.array([stats.spearmanr(X[:, j], y).statistic for j in range(p)])

    # ---------- 2. 互信息（抽样） ----------
    with Timer("互信息"):
        m = 60_000
        idx = RNG.choice(n, size=m, replace=False)
        mi = mutual_info_regression(
            X[idx], y[idx], discrete_features=True,
            n_neighbors=3, random_state=42,
        )

    # ---------- 3. 置换重要性（抽样 + 重复） ----------
    with Timer("置换重要性"):
        v = 200_000
        vidx = RNG.choice(n, size=v, replace=False)
        Xv, yv = X[vidx], y[vidx]
        base = rmse(yv, Xv @ coef + intercept)
        reps = 10
        perm = np.zeros((reps, p))
        for r in range(reps):
            for j in range(p):
                Xp = Xv.copy()
                RNG.shuffle(Xp[:, j])
                perm[r, j] = rmse(yv, Xp @ coef + intercept) - base
        perm_mean = perm.mean(axis=0)
        perm_std = perm.std(axis=0, ddof=1)

    # ---------- 4. drop-one R² ----------
    with Timer("drop-one"):
        drop_r2 = np.empty(p)
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        # 利用分块 Gram 矩阵加速：只需求 20 次 19x19
        G = Xc.T @ Xc
        b = Xc.T @ yc
        for j in range(p):
            keep = [k for k in range(p) if k != j]
            cj, *_ = np.linalg.lstsq(G[np.ix_(keep, keep)], b[keep], rcond=None)
            # R² = b^T c / (y^T y)
            r2_j = float(b[keep] @ cj / (yc @ yc))
            drop_r2[j] = r2_full - r2_j

    # ---------- 汇总表 ----------
    tab = pd.DataFrame({
        "指标": FEATURES,
        "pearson_r": corr,
        "spearman_rho": spear,
        "mutual_info": mi,
        "perm_rmse_increase": perm_mean,
        "perm_rmse_std": perm_std,
        "drop_one_r2": drop_r2,
        "ols_coef": coef,
    })
    tab["r2_univariate"] = tab["pearson_r"] ** 2
    tab["rank_pearson"] = tab["pearson_r"].rank(ascending=False).astype(int)
    tab = tab.sort_values("pearson_r", ascending=False).reset_index(drop=True)
    tab.to_csv(TABLES / "p1_importance.csv", index=False, encoding="utf-8-sig")

    # ---------- 不同重要性度量之间的一致性 ----------
    from itertools import combinations
    metric_cols = ["pearson_r", "mutual_info", "perm_rmse_increase", "drop_one_r2"]
    metric_corr = {}
    for a, b in combinations(metric_cols, 2):
        metric_corr[f"{a} vs {b}"] = float(
            stats.spearmanr(tab[a], tab[b]).statistic
        )

    # ---------- 图 1：相关系数 + CI ----------
    order = np.argsort(corr)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ci = 1.96 / np.sqrt(n - 3)
    z = np.arctanh(corr)
    lo = np.tanh(z - ci)
    hi = np.tanh(z + ci)
    ax.barh(np.arange(p), corr[order], color="#4C78A8", alpha=0.85)
    ax.errorbar(corr[order], np.arange(p),
                xerr=[corr[order] - lo[order], hi[order] - corr[order]],
                fmt="none", ecolor="k", elinewidth=0.8, capsize=2)
    ax.set_yticks(np.arange(p))
    ax.set_yticklabels([FEATURES[k] for k in order], fontsize=9)
    ax.set_xlabel("Pearson 相关系数 r（误差线为 95% 置信区间）")
    ax.set_title("20 个指标与洪水概率的相关性几乎相同（r≈0.18）")
    ax.set_xlim(0, 0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p1_corr_ci.png")
    plt.close(fig)

    # ---------- 图 2：自助法排名区间 ----------
    rank_tab = pd.read_csv(TABLES / "eda_rank_stability.csv", encoding="utf-8-sig")
    rank_tab = rank_tab.sort_values("rank_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ypos = np.arange(len(rank_tab))
    ax.hlines(ypos, rank_tab["rank_p2.5"], rank_tab["rank_p97.5"],
              color="#B0B0B0", lw=5, alpha=0.8)
    ax.scatter(rank_tab["rank_mean"], ypos, color="#E45756", zorder=3, s=28)
    ax.set_yticks(ypos)
    ax.set_yticklabels(rank_tab["指标"], fontsize=9)
    ax.set_xlabel("相关系数排名（1=最强，20=最弱）")
    ax.set_title("自助法 95% 排名区间几乎覆盖 1–20：排名不可区分")
    ax.set_xlim(0.5, 20.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p1_rank_interval.png")
    plt.close(fig)

    out = {
        "n": n,
        "ols_r2": r2_full,
        "ols_rmse": rmse_full,
        "pearson_r_range": [float(corr.min()), float(corr.max())],
        "spearman_range": [float(spear.min()), float(spear.max())],
        "mi_range": [float(mi.min()), float(mi.max())],
        "perm_rmse_increase_range": [float(perm_mean.min()), float(perm_mean.max())],
        "drop_one_r2_range": [float(drop_r2.min()), float(drop_r2.max())],
        "rank_interval_width_mean": float(
            (rank_tab["rank_p97.5"] - rank_tab["rank_p2.5"]).mean()
        ),
        "metric_rank_corr": metric_corr,
        "conclusion": (
            "20 个指标与洪水概率的关联强度几乎相同（r≈0.18，单变量 R²≈3.2%–3.6%）；"
            "自助法排名区间几乎覆盖 1–20，说明指标间的重要性排序不可区分。"
            "更关键的是，边际相关、互信息、置换重要性、留一 R² 四种度量给出的排序"
            "互不一致（Pearson 与置换重要性的 Spearman 甚至为负），"
            "因此任何“关键指标”结论都依赖度量选择，不可靠。"
        ),
    }
    save_json(out, RESULTS / "problem1.json")
    print(tab[["指标", "pearson_r", "mutual_info", "perm_rmse_increase", "drop_one_r2"]]
          .round(5).to_string(index=False))


if __name__ == "__main__":
    main()
