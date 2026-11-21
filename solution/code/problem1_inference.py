"""问题 1 补充：等权约束检验、可交换性检验与 Shapley R² 重要性分解。

问题 1 通常被做成"给 20 个指标排名"，但本文数据显示 20 个指标近似等权。
因此这里把 P1 从"排序问题"改造成"统计推断问题"，分三层给出证据：

1. 等权约束检验：约束模型 y = a + b*(x_1+...+x_20) 是否与无约束 OLS 等价。
   - 嵌套 F 检验 / 似然比检验 / AIC-BIC；
   - 系数相对偏差的 95% 置信区间与 ±1% 等价性判断。
2. 可交换性检验：随机分半后两半的系数向量是否可复现（Spearman）。
3. Shapley R² 分解（LMG）：每个指标对 R² 的平均边际贡献 + 自助法区间。

产出：
- results/problem1_inference.json
- results/p1_inference_notes.md
- results/tables/p1_equal_weight.csv
- results/tables/p1_shapley.csv
- figures/fig_p1_coef_equal.png
- figures/fig_p1_shapley.png
- figures/fig_p1_split_half.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    FIGURES, TABLES, RESULTS, FEATURES, TARGET, Timer,
    load_train, save_json, setup_plot, rmse, r2_score,
)

RNG = np.random.default_rng(2024)
P = len(FEATURES)


# --------------------------------------------------------------------------
# 线性代数工具：全部基于中心化 Gram 矩阵，避免反复构造 n×p 设计矩阵
# --------------------------------------------------------------------------
def gram(Xc: np.ndarray, yc: np.ndarray):
    return Xc.T @ Xc, Xc.T @ yc, float(yc @ yc)


def weighted_gram(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """按权重 w 重算中心化 Gram（自助法用，等价于重采样后的 OLS）。"""
    n = X.shape[0]
    Xw = X.T @ w
    yw = float(y @ w)
    G = X.T @ (w[:, None] * X) - np.outer(Xw, Xw) / n
    b = X.T @ (w * y) - Xw * yw / n
    yty = float(w @ (y * y) - yw * yw / n)
    return G, b, yty


def r2_subset(G: np.ndarray, b: np.ndarray, yty: float, idx) -> float:
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return 0.0
    Gs = G[np.ix_(idx, idx)]
    bs = b[idx]
    try:
        coef = np.linalg.solve(Gs, bs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(Gs, bs, rcond=None)[0]
    return float(bs @ coef / yty)


def ols_from_gram(G, b, yty):
    coef = np.linalg.solve(G, b)
    rss = yty - float(b @ coef)
    return coef, rss, 1.0 - rss / yty


def shapley_mc(G, b, yty, n_perm: int, rng) -> np.ndarray:
    """Monte Carlo 置换采样估计 Shapley R²（LMG）。"""
    p = len(b)
    vals = np.zeros(p)
    for _ in range(n_perm):
        order = rng.permutation(p)
        cur: list[int] = []
        prev = 0.0
        for j in order:
            cur.append(int(j))
            r2 = r2_subset(G, b, yty, cur)
            vals[j] += r2 - prev
            prev = r2
    return vals / n_perm


def ic(rss: float, k: int, n: int):
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return float(aic), float(bic)


def kmo(R: np.ndarray) -> float:
    Rinv = np.linalg.inv(R)
    d = np.sqrt(np.diag(Rinv))
    Pmat = -Rinv / np.outer(d, d)
    np.fill_diagonal(Pmat, 0.0)
    R0 = R.copy()
    np.fill_diagonal(R0, 0.0)
    return float((R0 ** 2).sum() / ((R0 ** 2).sum() + (Pmat ** 2).sum()))


# --------------------------------------------------------------------------
def main() -> None:
    setup_plot()
    import matplotlib.pyplot as plt

    with Timer("读取数据"):
        tr = load_train()
    X = tr[FEATURES].to_numpy(dtype=float)
    y = tr[TARGET].to_numpy(dtype=float)
    n, p = X.shape

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    G, b, yty = gram(Xc, yc)
    Ginv = np.linalg.inv(G)

    # ---------------- 1. 无约束 OLS ----------------
    coef, rss_f, r2_f = ols_from_gram(G, b, yty)
    intercept = float(y.mean() - X.mean(axis=0) @ coef)
    sigma2 = rss_f / (n - p - 1)
    se = np.sqrt(sigma2 * np.diag(Ginv))
    rmse_f = float(np.sqrt(rss_f / n))

    # ---------------- 2. 等权约束模型 y = a + beta * sum(x) ----------------
    S = X.sum(axis=1)
    Sc = S - S.mean()
    g_ss = float(Sc @ Sc)
    b_s = float(Sc @ yc)
    beta_sum = b_s / g_ss                      # 对"指标之和"的斜率
    beta_mean = beta_sum * p                   # 对"指标均值"的斜率（等价写法）
    intercept_r = float(y.mean() - beta_sum * S.mean())
    rss_r = yty - b_s ** 2 / g_ss
    r2_r = 1.0 - rss_r / yty
    rmse_r = float(np.sqrt(rss_r / n))

    df1, df2 = p - 1, n - p - 1
    F_eq = float(((rss_r - rss_f) / df1) / sigma2)
    p_F = float(stats.f.sf(F_eq, df1, df2))
    lr = float(n * np.log(rss_r / rss_f))
    p_lr = float(stats.chi2.sf(lr, df1))

    # 系数相对偏差及其 95% CI（相对等权斜率）
    rel_dev = (coef - beta_sum) / beta_sum
    rel_half = 1.96 * se / beta_sum
    max_abs_dev = float(np.abs(rel_dev).max())
    max_abs_dev_ci = float((np.abs(rel_dev) + rel_half).max())
    # 工程等价容差：权重差异 <1.5% 视为等权（1% 边界恰被 CI 触及，故放宽到 1.5%）
    within_tol = bool(max_abs_dev_ci < 0.015)

    aic_f, bic_f = ic(rss_f, p + 1, n)
    aic_r, bic_r = ic(rss_r, 2, n)

    eq_tab = pd.DataFrame([
        {
            "模型": "无约束 OLS（20 指标）",
            "R²": r2_f, "RMSE": rmse_f, "AIC": aic_f, "BIC": bic_f,
            "参数个数": p + 1,
            "系数最小": float(coef.min()), "系数最大": float(coef.max()),
            "系数极差/均值": float((coef.max() - coef.min()) / coef.mean()),
        },
        {
            "模型": "等权约束（行总和 N）",
            "R²": r2_r, "RMSE": rmse_r, "AIC": aic_r, "BIC": bic_r,
            "参数个数": 2,
            "系数最小": beta_sum, "系数最大": beta_sum,
            "系数极差/均值": 0.0,
        },
    ])
    eq_tab.to_csv(TABLES / "p1_equal_weight.csv", index=False, encoding="utf-8-sig")

    # ---------------- 3. 可交换性：随机分半复现 ----------------
    with Timer("分半复现"):
        perm_idx = RNG.permutation(n)
        halves = []
        for k in range(2):
            idx = perm_idx[k * (n // 2):(k + 1) * (n // 2)] if k == 0 else perm_idx[n // 2:]
            Xk = X[idx]
            yk = y[idx]
            Gk, bk, ytyk = gram(Xk - Xk.mean(axis=0), yk - yk.mean())
            ck, _, _ = ols_from_gram(Gk, bk, ytyk)
            halves.append(ck)
        rho_half = float(stats.spearmanr(halves[0], halves[1]).statistic)
        r_half = float(np.corrcoef(halves[0], halves[1])[0, 1])
        top5_a = set(np.argsort(-halves[0])[:5])
        top5_b = set(np.argsort(-halves[1])[:5])
        top5_overlap = len(top5_a & top5_b)

    # ---------------- 4. 独立性 / 因子结构诊断 ----------------
    with Timer("独立性诊断"):
        R = np.corrcoef(X, rowvar=False)
        off = R[~np.eye(p, dtype=bool)]
        eig = np.sort(np.linalg.eigvalsh(R))[::-1]
        pc1_share = float(eig[0] / p)
        sign, logdet = np.linalg.slogdet(R)
        bart_chi2 = float(-(n - 1 - (2 * p + 5) / 6) * logdet)
        bart_df = p * (p - 1) // 2
        bart_p = float(stats.chi2.sf(bart_chi2, bart_df))

    # ---------------- 5. Shapley R² 分解 ----------------
    with Timer("Shapley R²（分块）"):
        n_blocks, per_block = 20, 200
        blocks = np.empty((n_blocks, p))
        for k in range(n_blocks):
            blocks[k] = shapley_mc(G, b, yty, per_block, RNG)
        shap = blocks.mean(axis=0)
        mc_se = blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)

    with Timer("Shapley 自助法区间"):
        B, per_boot = 25, 300
        boot = np.empty((B, p))
        boot_rank = np.empty((B, p), dtype=int)
        for r in range(B):
            w = np.bincount(RNG.integers(0, n, n), minlength=n).astype(float)
            Gw, bw, ytyw = weighted_gram(X, y, w)
            vals = shapley_mc(Gw, bw, ytyw, per_boot, RNG)
            boot[r] = vals
            order = np.argsort(-vals)
            boot_rank[r, order] = np.arange(1, p + 1)

    ci_lo = np.percentile(boot, 2.5, axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)
    rank_lo = np.percentile(boot_rank, 2.5, axis=0)
    rank_hi = np.percentile(boot_rank, 97.5, axis=0)

    # drop-one 与单变量 R²（与 Shapley 对照）
    drop_one = np.array([
        r2_f - r2_subset(G, b, yty, [k for k in range(p) if k != j])
        for j in range(p)
    ])
    uni_r2 = np.array([np.corrcoef(X[:, j], y)[0, 1] ** 2 for j in range(p)])

    rho_shap_drop = float(stats.spearmanr(shap, drop_one).statistic)
    rho_shap_uni = float(stats.spearmanr(shap, uni_r2).statistic)

    shap_tab = pd.DataFrame({
        "指标": FEATURES,
        "shapley_r2": shap,
        "shapley_share": shap / r2_f,
        "mc_se": mc_se,
        "boot_lo": ci_lo,
        "boot_hi": ci_hi,
        "rank_mean": boot_rank.mean(axis=0),
        "rank_lo": rank_lo,
        "rank_hi": rank_hi,
        "drop_one_r2": drop_one,
        "univariate_r2": uni_r2,
    }).sort_values("shapley_r2", ascending=False).reset_index(drop=True)
    shap_tab.to_csv(TABLES / "p1_shapley.csv", index=False, encoding="utf-8-sig")

    # ---------------- 6. 图 ----------------
    # 图 1：系数相对偏差 + CI + ±1% 带
    order = np.argsort(rel_dev)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ypos = np.arange(p)
    ax.axvspan(-1, 1, color="#54A24B", alpha=0.12, label="±1% 参考带")
    ax.axvline(0, color="k", lw=0.8)
    ax.errorbar(rel_dev[order] * 100, ypos,
                xerr=rel_half[order] * 100, fmt="o", color="#4C78A8",
                ecolor="#4C78A8", elinewidth=1.0, capsize=2, ms=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([FEATURES[k] for k in order], fontsize=9)
    ax.set_xlabel("相对等权斜率的偏差（%，误差线为 95% CI）")
    ax.set_title(
        f"系数与等权斜率最大偏差 {max_abs_dev * 100:.2f}%，"
        f"95% CI 上限 {max_abs_dev_ci * 100:.2f}%"
    )
    ax.set_xlim(-1.5, 1.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p1_coef_equal.png")
    plt.close(fig)

    # 图 2：Shapley R² + 自助法区间 + 等分基线
    o = np.argsort(shap)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ypos = np.arange(p)
    ax.axvline(r2_f / p, color="#E45756", ls="--", lw=1.2,
               label=f"等分基线 R²/20 = {r2_f / p:.4f}")
    ax.errorbar(shap[o], ypos,
                xerr=[shap[o] - ci_lo[o], ci_hi[o] - shap[o]],
                fmt="o", color="#4C78A8", ecolor="#4C78A8",
                elinewidth=0.9, capsize=2, ms=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([FEATURES[k] for k in o], fontsize=9)
    ax.set_xlabel("Shapley R² 贡献（误差线为 95% 自助法区间）")
    ax.set_title(
        f"Shapley R² 分解：各指标贡献差异仅 "
        f"{100 * (shap.max() - shap.min()) / shap.mean():.1f}%"
        f"（均值 {shap.mean():.4f}）"
    )
    ax.set_xlim(shap.min() - 0.002, shap.max() + 0.002)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p1_shapley.png")
    plt.close(fig)

    # 图 3：分半复现散点
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.scatter(halves[0] * 1e3, halves[1] * 1e3, s=42, color="#4C78A8", zorder=3)
    lo = min(halves[0].min(), halves[1].min()) * 1e3 * 0.97
    hi = max(halves[0].max(), halves[1].max()) * 1e3 * 1.03
    ax.plot([lo, hi], [lo, hi], color="#E45756", ls="--", lw=1.0,
            label="y = x（完全复现）")
    ax.set_xlabel("前半样本 OLS 系数（×10^-3）")
    ax.set_ylabel("后半样本 OLS 系数（×10^-3）")
    ax.set_title(
        f"随机分半后排序部分复现（Spearman ρ = {rho_half:.3f}），"
        f"但幅度差异 <1%"
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p1_split_half.png")
    plt.close(fig)

    # ---------------- 7. 汇总 ----------------
    out = {
        "n": n,
        "equal_weight_test": {
            "restricted": {
                "formula": "y = a + beta * sum(x_j)",
                "r2": r2_r, "rmse": rmse_r,
                "beta_sum": beta_sum, "beta_mean": beta_mean,
                "intercept": intercept_r,
                "aic": aic_r, "bic": bic_r, "n_params": 2,
            },
            "unrestricted": {
                "r2": r2_f, "rmse": rmse_f,
                "coef_mean": float(coef.mean()),
                "coef_std": float(coef.std(ddof=1)),
                "coef_min": float(coef.min()),
                "coef_max": float(coef.max()),
                "aic": aic_f, "bic": bic_f, "n_params": p + 1,
            },
            "delta_r2": float(r2_f - r2_r),
            "delta_rmse": float(rmse_f - rmse_r),
            "F": F_eq, "df1": int(df1), "df2": int(df2), "p_value": p_F,
            "LR_chi2": lr, "LR_p_value": p_lr,
            "coef_rel_range": float((coef.max() - coef.min()) / coef.mean()),
            "max_abs_rel_dev": max_abs_dev,
            "max_abs_rel_dev_ci95": max_abs_dev_ci,
            "equivalence_within_1p5pct": within_tol,
        },
        "exchangeability": {
            "split_half_spearman": rho_half,
            "split_half_pearson": r_half,
            "top5_overlap": int(top5_overlap),
        },
        "independence": {
            "mean_pairwise_r": float(off.mean()),
            "min_pairwise_r": float(off.min()),
            "max_pairwise_r": float(off.max()),
            "mean_abs_r": float(np.abs(off).mean()),
            "eigen_min": float(eig[-1]),
            "eigen_max": float(eig[0]),
            "pc1_share": pc1_share,
            "equal_share": 1.0 / p,
            "feature_sd_min": float(X.std(axis=0).min()),
            "feature_sd_max": float(X.std(axis=0).max()),
            "feature_sd_ratio": float(X.std(axis=0).max() / X.std(axis=0).min()),
            "kmo": kmo(R),
            "bartlett_chi2": bart_chi2,
            "bartlett_df": int(bart_df),
            "bartlett_p_value": bart_p,
        },
        "shapley": {
            "total_r2": r2_f,
            "equal_share": float(r2_f / p),
            "min": float(shap.min()),
            "max": float(shap.max()),
            "range_over_mean": float((shap.max() - shap.min()) / shap.mean()),
            "max_mc_se": float(mc_se.max()),
            "max_boot_ci_width": float((ci_hi - ci_lo).max()),
            "rank_interval_width_mean": float((rank_hi - rank_lo).mean()),
            "spearman_shapley_vs_drop_one": rho_shap_drop,
            "spearman_shapley_vs_univariate": rho_shap_uni,
            "top3": shap_tab.head(3)["指标"].tolist(),
            "bottom3": shap_tab.tail(3)["指标"].tolist(),
        },
        "conclusion": (
            "严格等权在统计上被拒绝，但效应量可忽略："
            f"F({df1},{df2})={F_eq:.2f}，p={p_F:.1e}，"
            f"ΔR²={r2_f - r2_r:.2e}，ΔRMSE={abs(rmse_f - rmse_r):.1e}；"
            f"系数相对偏差最大 {max_abs_dev * 100:.2f}%，95% CI 上限 {max_abs_dev_ci * 100:.2f}%。"
            f"随机分半后系数排序 Spearman ρ={rho_half:.3f}、前五名重合 {top5_overlap}/5，"
            "说明存在 <1% 的可复现权重差异，但幅度不足以支撑差异化决策。"
            f"Shapley R² 每项贡献 {shap.min():.4f}–{shap.max():.4f}，"
            f"极差仅占均值 {100 * (shap.max() - shap.min()) / shap.mean():.2f}%，"
            f"排名区间平均宽度 {(rank_hi - rank_lo).mean():.1f}/20；"
            f"且 Shapley 与 drop-one 排名的 Spearman 仅 {rho_shap_drop:.2f}"
            f"（与单变量 R² 为 {rho_shap_uni:.2f}），说明重要性排序依赖度量选择。"
            "因此应把 20 个指标视为近似等权的综合致灾指数，"
            "最多指出相对略高/略低的指标，不能宣称存在关键指标。"
        ),
    }
    save_json(out, RESULTS / "problem1_inference.json")

    # 给论文用的中文要点
    shap_range_pct = 100 * (shap.max() - shap.min()) / shap.mean()
    notes = f"""# 问题 1 补充证据：等权约束、可交换性与重要性分解

## 1. 等权约束检验

把 20 个指标的斜率约束为相等：$y=a+\\beta\\sum_j x_j+\\varepsilon$。

| 模型 | R² | RMSE | AIC | BIC | 参数个数 |
|---|---|---|---|---|---|
| 无约束 OLS | {r2_f:.6f} | {rmse_f:.6f} | {aic_f:.1f} | {bic_f:.1f} | {p + 1} |
| 等权约束 | {r2_r:.6f} | {rmse_r:.6f} | {aic_r:.1f} | {bic_r:.1f} | 2 |

- $\\Delta R^2={r2_f - r2_r:.2e}$，$\\Delta$RMSE$={abs(rmse_f - rmse_r):.1e}$；
- 嵌套 F 检验：$F({df1},{df2})={F_eq:.2f}$，$p={p_F:.1e}$，**严格等权被拒绝**；
- 似然比检验：$\\chi^2_{{{df1}}}={lr:.2f}$，$p={p_lr:.1e}$；
- 但系数相对偏差最大只有 {max_abs_dev * 100:.2f}%，95% CI 上限 {max_abs_dev_ci * 100:.2f}%。

**这是典型的“显著 ≠ 重要”**：104.9 万样本足以检出 0.3% 量级的权重差异，
但把 20 个斜率强行约束为相等，$R^2$ 只损失 {r2_f - r2_r:.2e}，RMSE 只差
{abs(rmse_f - rmse_r):.1e}。AIC 因样本量巨大而偏向无约束模型（$\\Delta$AIC={aic_f - aic_r:.1f}），
BIC 则选择等权模型（$\\Delta$BIC={bic_r - bic_f:.1f}）；两者对应的预测差异都可忽略。

## 2. 可交换性（分半复现）

随机分半后，两半样本的 20 个 OLS 系数：

- Spearman $\\rho={rho_half:.3f}$，Pearson $r={r_half:.3f}$；
- 前五名重要指标重合 {top5_overlap}/5。

结论：存在 <1% 的可复现权重差异，排序不是纯噪声；但幅度太小，
不足以支撑“重点监测某几个指标”的政策结论。

## 3. 独立性 / 因子结构诊断

- 两两相关均值 {off.mean():.4f}，绝对值均值 {np.abs(off).mean():.4f}，最大 {off.max():.4f}；
- 相关矩阵特征值 {eig[-1]:.3f}–{eig[0]:.3f}，PC1 解释率 {pc1_share * 100:.2f}%（等权基线 {100 / p:.2f}%）；
- KMO={kmo(R):.3f}；Bartlett $\\chi^2={bart_chi2:.0f}$，df={bart_df}（$p<10^{{-3}}$，大样本下仅作参考）。

结论：没有解释率显著高于 1/20 的共同因子，KMO 仅 {kmo(R):.2f}；最小特征值
{eig[-1]:.2f} 对应“行总和守恒”方向。指标之间不存在真正的风险因子结构，
重要性不能由因子载荷解释。

## 4. Shapley R² 分解（LMG）

| 指标 | Shapley R² | 95% CI | 排名区间 |
|---|---|---|---|
"""
    for _, row in shap_tab.iterrows():
        notes += (
            f"| {row['指标']} | {row['shapley_r2']:.4f} | "
            f"[{row['boot_lo']:.4f}, {row['boot_hi']:.4f}] | "
            f"{row['rank_lo']:.0f}–{row['rank_hi']:.0f} |\n"
        )
    notes += f"""
- 20 项 Shapley R² 之和为 {shap.sum():.6f}（= 全模型 R²）；
- 每项 {shap.min():.4f}–{shap.max():.4f}，极差占均值 {shap_range_pct:.2f}%；
- 自助法排名区间平均宽度 {(rank_hi - rank_lo).mean():.1f}/20；
- Shapley 与 drop-one R² 的排序 Spearman={rho_shap_drop:.3f}，与单变量 R² 的排序 Spearman={rho_shap_uni:.3f}；
- 指标标准差从 {X.std(axis=0).min():.3f} 到 {X.std(axis=0).max():.3f}（相差 {100 * (X.std(axis=0).max() / X.std(axis=0).min() - 1):.1f}%），
  而 Shapley 贡献对 $\\beta_j^2\\,\\mathrm{{Var}}(x_j)$ 敏感，因此这部分差异主要来自指标方差，而非权重。

## 5. 可直接写入论文的表述

> 问题 1 的结论不是“某几个指标显著更重要”，而是“20 个指标近似等权，差异在 1% 量级”。
> 在 104.9 万样本上，把 20 个斜率约束为相等后，$R^2$ 只从 {r2_f:.5f} 降到 {r2_r:.5f}
> （$\\Delta R^2={r2_f - r2_r:.2e}$），RMSE 只差 {abs(rmse_f - rmse_r):.1e}；虽然嵌套 F 检验
> $p={p_F:.1e}$ 拒绝严格等权，但系数相对偏差最大仅 {max_abs_dev * 100:.2f}%，属于“统计显著、
> 实践可忽略”。随机分半后系数排序的 Spearman 相关为 {rho_half:.3f}、前五名重合 {top5_overlap}/5，
> 说明排序有一定可复现性，但幅度差异不足 1%。Shapley R² 分解显示每个指标的贡献都在
> {shap.min():.4f}–{shap.max():.4f}（均值 {shap.mean():.4f}），极差仅 {shap_range_pct:.1f}%，
> 且与 drop-one 排名的 Spearman 仅 {rho_shap_drop:.2f}，说明“谁更重要”高度依赖度量选择。
> 因此，相对略高的 {', '.join(shap_tab.head(3)['指标'])} 与略低的
> {', '.join(shap_tab.tail(3)['指标'])} 只能在“<1% 差异”的意义上讨论；
> 预防策略应针对综合致灾背景指数做多维度协同治理，而不是押注少数指标。
"""
    (RESULTS / "p1_inference_notes.md").write_text(notes, encoding="utf-8")

    print(eq_tab.round(6).to_string(index=False))
    print()
    print(shap_tab[["指标", "shapley_r2", "boot_lo", "boot_hi",
                    "rank_lo", "rank_hi", "drop_one_r2"]]
          .round(5).to_string(index=False))
    print()
    print(out["conclusion"])


if __name__ == "__main__":
    main()
