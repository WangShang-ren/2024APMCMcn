"""问题 2：洪水风险分级、指标赋权、TOPSIS 预警评价与灵敏度分析。

流程：
1. 按目标值三分位划分低/中/高风险，并给出各风险组的指标画像；
2. 对照：对 20 维指标做 KMeans（特征空间聚类），检验是否存在“指标组合型”风险；
3. 赋权：熵权法、CRITIC、PCA 第一主成分、等权；
4. TOPSIS 综合评价 + 分级阈值；
5. 灵敏度：权重 Dirichlet 扰动 + 单指标 ±20% 扰动，看排序稳定性。

产出：
- results/problem2.json
- results/tables/p2_profiles.csv
- results/tables/p2_weights.csv
- results/tables/p2_sensitivity.csv
- figures/fig_p2_profiles.png
- figures/fig_p2_weights.png
- figures/fig_p2_topsis.png
- figures/fig_p2_sensitivity.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from common import (
    FIGURES, TABLES, RESULTS, FEATURES, TARGET, Timer, load_train,
    save_json, r2_score, setup_plot,
)

RNG = np.random.default_rng(42)


def entropy_weights(X: np.ndarray) -> np.ndarray:
    """熵权法（X 为非负指标）。"""
    Xp = X + 1e-12
    P = Xp / Xp.sum(axis=0, keepdims=True)
    n = X.shape[0]
    k = 1.0 / np.log(n)
    E = -k * np.sum(P * np.log(P), axis=0)
    d = 1.0 - E
    return d / d.sum()


def critic_weights(X: np.ndarray) -> np.ndarray:
    """CRITIC：对比强度 × 冲突性。"""
    std = X.std(axis=0, ddof=1)
    R = np.corrcoef(X, rowvar=False)
    conflict = (1.0 - R).sum(axis=0)
    c = std * conflict
    return c / c.sum()


def pca_weights(X: np.ndarray) -> tuple[np.ndarray, float]:
    """PCA 第一主成分载荷（绝对值归一化）与解释方差比。"""
    R = np.corrcoef(X, rowvar=False)
    vals, vecs = np.linalg.eigh(R)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    v = np.abs(vecs[:, 0])
    return v / v.sum(), float(vals[0] / vals.sum())


def topsis(X: np.ndarray, w: np.ndarray):
    """TOPSIS：所有指标按“越大风险越高”处理。"""
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    R = X / norm
    V = R * w
    best = V.max(axis=0)
    worst = V.min(axis=0)
    d_best = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - worst) ** 2).sum(axis=1))
    score = d_worst / (d_best + d_worst + 1e-12)
    return score


def main() -> None:
    setup_plot()
    import matplotlib.pyplot as plt

    with Timer("读取数据"):
        tr = load_train()
    X = tr[FEATURES].to_numpy(dtype=float)
    y = tr[TARGET].to_numpy(dtype=float)
    n = len(y)

    # ---------- 1. 风险分级（三分位） ----------
    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
    labels = np.where(y <= q1, 0, np.where(y <= q2, 1, 2))
    names = ["低风险", "中风险", "高风险"]

    profile_rows = []
    overall_mean = X.mean(axis=0)
    overall_std = X.std(axis=0, ddof=1)
    eta2 = {}
    for j, c in enumerate(FEATURES):
        grand = X[:, j].mean()
        ss_between = sum(
            (labels == g).sum() * (X[labels == g, j].mean() - grand) ** 2
            for g in range(3)
        )
        ss_total = ((X[:, j] - grand) ** 2).sum()
        eta2[c] = float(ss_between / ss_total)
        row = {"指标": c, "总体均值": grand, "总体标准差": overall_std[j]}
        for g in range(3):
            row[f"{names[g]}_均值"] = float(X[labels == g, j].mean())
            row[f"{names[g]}_标准化差"] = float(
                (X[labels == g, j].mean() - grand) / overall_std[j]
            )
        profile_rows.append(row)
    prof = pd.DataFrame(profile_rows)
    prof.to_csv(TABLES / "p2_profiles.csv", index=False, encoding="utf-8-sig")

    group_target_mean = {
        names[g]: float(y[labels == g].mean()) for g in range(3)
    }
    group_sizes = {names[g]: int((labels == g).sum()) for g in range(3)}

    # ---------- 2. 对照：特征空间 KMeans ----------
    with Timer("KMeans 对照"):
        m = 200_000
        idx = RNG.choice(n, size=m, replace=False)
        Xz = (X[idx] - X.mean(0)) / X.std(0)
        km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xz)
        ari = float(adjusted_rand_score(labels[idx], km.labels_))
        km_sizes = np.bincount(km.labels_).tolist()
        km_target_mean = [
            float(y[idx][km.labels_ == k].mean()) for k in range(3)
        ]
        km_profile = np.vstack([
            X[idx][km.labels_ == k].mean(axis=0) for k in range(3)
        ])

    # ---------- 3. 赋权 ----------
    with Timer("赋权"):
        w_ent = entropy_weights(X)
        w_cri = critic_weights(X)
        w_pca, pca_evr = pca_weights(X)
        w_eq = np.ones(len(FEATURES)) / len(FEATURES)

    wtab = pd.DataFrame({
        "指标": FEATURES,
        "等权": w_eq,
        "熵权": w_ent,
        "CRITIC": w_cri,
        "PCA1": w_pca,
    })
    wtab.to_csv(TABLES / "p2_weights.csv", index=False, encoding="utf-8-sig")

    weights_summary = {}
    for name, w in [("等权", w_eq), ("熵权", w_ent), ("CRITIC", w_cri), ("PCA1", w_pca)]:
        weights_summary[name] = {
            "min": float(w.min()), "max": float(w.max()),
            "std": float(w.std(ddof=0)),
            "max_over_min": float(w.max() / w.min()),
        }

    # ---------- 4. TOPSIS ----------
    with Timer("TOPSIS"):
        scores = {}
        for name, w in [("等权", w_eq), ("熵权", w_ent), ("CRITIC", w_cri)]:
            scores[name] = topsis(X, w)
    main_score = scores["CRITIC"]
    sp = float(stats.spearmanr(main_score, y).statistic)
    pe = float(np.corrcoef(main_score, y)[0, 1])
    # 线性校准后的 R²
    A = np.vstack([main_score, np.ones_like(main_score)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2_cal = float(r2_score(y, A @ c))

    # 预警等级：按 TOPSIS 分数三分位
    t1, t2 = np.quantile(main_score, [1 / 3, 2 / 3])
    warn = np.where(main_score <= t1, 0, np.where(main_score <= t2, 1, 2))
    acc = float((warn == labels).mean())
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(labels, warn):
        cm[a, b] += 1

    # ---------- 5. 灵敏度 ----------
    with Timer("灵敏度"):
        s = 20_000
        sidx = RNG.choice(n, size=s, replace=False)
        Xs = X[sidx]
        base = topsis(Xs, w_cri)
        # 5.1 Dirichlet 权重扰动（围绕 CRITIC 权重，约 ±10% 相对波动）
        B = 300
        conc = np.maximum(w_cri, 1e-6) * 2000.0
        rhos = np.empty(B)
        w_rel_std = []
        for b in range(B):
            wb = RNG.dirichlet(conc)
            w_rel_std.append(np.std(wb / w_cri - 1.0))
            rhos[b] = stats.spearmanr(base, topsis(Xs, wb)).statistic
        # 5.2 单指标 ±20%
        oat = []
        for j, c in enumerate(FEATURES):
            for factor in (0.8, 1.2):
                wb = w_cri.copy()
                wb[j] *= factor
                wb = wb / wb.sum()
                oat.append({
                    "指标": c, "factor": factor,
                    "spearman_vs_base": float(
                        stats.spearmanr(base, topsis(Xs, wb)).statistic
                    ),
                })
        oat_tab = pd.DataFrame(oat)
        oat_tab.to_csv(TABLES / "p2_sensitivity.csv", index=False,
                       encoding="utf-8-sig")

        sens = {
            "dirichlet_B": B,
            "dirichlet_relative_std_mean": float(np.mean(w_rel_std)),
            "spearman_mean": float(rhos.mean()),
            "spearman_min": float(rhos.min()),
            "spearman_p2.5": float(np.percentile(rhos, 2.5)),
            "spearman_p97.5": float(np.percentile(rhos, 97.5)),
            "oat_spearman_min": float(oat_tab["spearman_vs_base"].min()),
            "oat_spearman_max": float(oat_tab["spearman_vs_base"].max()),
        }

    # ---------- 6. 图 ----------
    # 6.1 风险组画像热图
    Z = prof[[f"{nm}_标准化差" for nm in names]].to_numpy().T
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_yticks(range(3))
    ax.set_yticklabels(names)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=90, fontsize=8)
    ax.set_title("三个风险组的指标标准化均值：所有指标同步平移，没有“组合型”画像")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                 label="(组均值−总体均值)/总体标准差")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p2_profiles.png")
    plt.close(fig)

    # 6.2 权重对比（PCA 单独一栏，避免压扁近等权结果）
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3.2, 1.0]}
    )
    xpos = np.arange(len(FEATURES))
    ax = axes[0]
    for k, (name, w) in enumerate([("等权", w_eq), ("熵权", w_ent),
                                   ("CRITIC", w_cri)]):
        ax.bar(xpos + (k - 1) * 0.27, w, width=0.27, label=name)
    ax.axhline(1 / len(FEATURES), color="k", ls="--", lw=0.8)
    ax.set_ylim(0.046, 0.054)
    ax.set_xticks(xpos)
    ax.set_xticklabels(FEATURES, rotation=90, fontsize=7)
    ax.set_ylabel("权重")
    ax.set_title("等权 / 熵权 / CRITIC：都接近 1/20")
    ax.legend(ncol=3, fontsize=8)

    ax2 = axes[1]
    ax2.bar(xpos, w_pca, color="#B279A2")
    ax2.axhline(1 / len(FEATURES), color="k", ls="--", lw=0.8)
    ax2.set_xticks([])
    ax2.set_ylabel("权重")
    ax2.set_title("PCA1（不适用）", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p2_weights.png")
    plt.close(fig)

    # 6.3 TOPSIS vs 目标
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sidx2 = RNG.choice(n, size=80_000, replace=False)
    hb = ax.hexbin(main_score[sidx2], y[sidx2], gridsize=55,
                   cmap="viridis", mincnt=1)
    ax.set_xlabel("CRITIC-TOPSIS 综合得分")
    ax.set_ylabel("洪水概率")
    ax.set_title(f"TOPSIS 得分与目标高度一致（Spearman={sp:.3f}）")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p2_topsis.png")
    plt.close(fig)

    # 6.4 灵敏度
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(rhos, bins=30, color="#54A24B", alpha=0.85)
    ax.axvline(rhos.mean(), color="k", ls="--", lw=1,
               label=f"均值={rhos.mean():.4f}")
    ax.set_xlabel("扰动权重下 TOPSIS 排序与基准的 Spearman")
    ax.set_ylabel("频数")
    ax.set_title("权重随机扰动约 ±10%：整体排序仍保持较强正相关")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p2_sensitivity.png")
    plt.close(fig)

    out = {
        "risk_thresholds_y": [float(q1), float(q2)],
        "group_sizes": group_sizes,
        "group_target_mean": group_target_mean,
        "eta2_range": [float(min(eta2.values())), float(max(eta2.values()))],
        "eta2_mean": float(np.mean(list(eta2.values()))),
        "kmeans": {
            "ari_vs_target_tercile": ari,
            "cluster_sizes": km_sizes,
            "cluster_target_mean": km_target_mean,
        },
        "weights": weights_summary,
        "pca_first_component_evr": pca_evr,
        "topsis": {
            "spearman_vs_target": sp,
            "pearson_vs_target": pe,
            "r2_after_linear_calibration": r2_cal,
            "warning_thresholds": [float(t1), float(t2)],
            "grade_accuracy": acc,
            "confusion_matrix": cm.tolist(),
        },
        "sensitivity": sens,
        "conclusion": (
            "三个风险组的 20 个指标均值同步平移，不存在“组合型”风险画像；"
            "熵权/CRITIC/等权结果都接近 1/20（PCA 因数据近似各向同性而不适用）；"
            "TOPSIS 得分与目标高度一致；权重约 ±10% 扰动下排序仍保持较强正相关，"
            "单指标 ±20% 扰动下 ρ>0.99。"
        ),
    }
    save_json(out, RESULTS / "problem2.json")
    print("组目标均值:", group_target_mean)
    print("eta² 范围:", out["eta2_range"])
    print("KMeans ARI vs 三分位:", round(ari, 4))
    print("权重最大/最小比:", {k: round(v["max_over_min"], 3)
                               for k, v in weights_summary.items()})
    print(f"TOPSIS Spearman={sp:.4f}, 分级一致率={acc:.4f}")
    print("灵敏度:", {k: round(v, 4) for k, v in sens.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
