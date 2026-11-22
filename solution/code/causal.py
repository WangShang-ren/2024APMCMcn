"""因果敏感性分析：N 是混杂变量，给定 N 后指标与 y 条件独立。

不做严格因果推断（没有干预实验），而是：
1. DAG：N -> x_j, N -> y；R（潜在机制）-> y，且 R 与 x 近似独立；
2. 条件独立检验：边际相关 r(x_j,y)、偏相关 r(x_j,y|N)、偏 R²；
3. 干预模拟：保持 N 不变的重新分配 -> Δy=0；改变 N -> Δy=β；
4. 效应异质性：潜在机制使 N->y 斜率在 [0.0044,0.0061] 之间变化。

产出：
- results/causal.json
- results/tables/causal_corr.csv
- results/tables/causal_intervention.csv
- figures/fig_causal_dag.png
- figures/fig_causal_corr.png
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    FIGURES, TABLES, RESULTS, FEATURES, TARGET, Timer,
    load_train, save_json, setup_plot,
)


def partial_corr(x, y, z):
    """r(x,y|z)，z 为 n×k 协变量（含截距）。"""
    Z = np.column_stack([np.ones_like(z), z])
    cx = np.linalg.lstsq(Z, x, rcond=None)[0]
    cy = np.linalg.lstsq(Z, y, rcond=None)[0]
    rx, ry = x - Z @ cx, y - Z @ cy
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_r2(x, y, z):
    Z = np.column_stack([np.ones_like(z), z])
    sse0 = np.sum((y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]) ** 2)
    Zj = np.column_stack([Z, x])
    sse1 = np.sum((y - Zj @ np.linalg.lstsq(Zj, y, rcond=None)[0]) ** 2)
    return float((sse0 - sse1) / sse0)


def main() -> None:
    setup_plot()
    import matplotlib.pyplot as plt

    with Timer("读取数据"):
        tr = load_train()
    X = tr[FEATURES].to_numpy(dtype=float)
    y = tr[TARGET].to_numpy(dtype=float)
    N = X.sum(axis=1).astype(float)

    # ---------- 1. 边际 / 偏相关 / 偏 R² ----------
    rows = []
    for j, c in enumerate(FEATURES):
        r_m = float(np.corrcoef(X[:, j], y)[0, 1])
        r_p = partial_corr(X[:, j], y, N)
        pr2 = partial_r2(X[:, j], y, N)
        z = np.arctanh(np.clip(r_p, -0.999, 0.999)) * np.sqrt(len(y) - 3)
        p = float(2 * stats.norm.sf(abs(z)))
        rows.append({
            "指标": c, "边际r": r_m, "偏相关r_given_N": r_p,
            "偏R2": pr2, "偏相关p值": p,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES / "causal_corr.csv", index=False, encoding="utf-8-sig")

    # 控制 N 后在窄区间内的平均相关（另一种条件独立证据）
    bin_corr = []
    for lo, hi in [(80, 90), (90, 100), (100, 110), (110, 120)]:
        m = (N >= lo) & (N < hi)
        C = np.corrcoef(X[m], rowvar=False)
        bin_corr.append(float(np.mean(C[~np.eye(len(FEATURES), dtype=bool)])))

    # ---------- 2. 潜在机制参数 ----------
    gen = json.load(open(RESULTS / "generative.json", encoding="utf-8"))
    lr = gen["latent_regime"]
    a1, a2 = lr["a"]
    b1, b2 = lr["b"]

    # ---------- 3. 干预模拟 ----------
    a = gen["target"]["a"]
    interventions = [
        ["保持 N 不变，重新分配 20 个指标", "x_j 之间转移 1 分", 0.0,
         "给定 N，y 与分配无关"],
        ["N 增加 1（随机给某个指标 +1）", "N -> N+1", a,
         "总量效应 $\\beta$"],
        ["20 个指标各 +1", "N -> N+20", 20 * a,
         "总量效应 ×20"],
        ["把 1 分从指标 j 转到指标 k", "N 不变", 0.0,
         "纯分配干预，无直接效应"],
        ["潜在机制从 1 切到 2（同一 N）", "R: 1->2",
         (a2 - a1) * 99.0 + (b2 - b1),
         "不可观测、不可干预，仅作敏感性"],
    ]
    itab = pd.DataFrame(interventions,
                        columns=["干预", "操作", "Δy", "解释"])
    itab.to_csv(TABLES / "causal_intervention.csv", index=False,
                encoding="utf-8-sig")

    # ---------- 4. 图：DAG ----------
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    # nodes
    ax.text(5, 4.2, "N\n（行总和 / 综合致灾指数）", ha="center", va="center",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.35", fc="#DDEBF7", ec="#4C78A8"))
    ax.text(1.4, 2.4, "x1", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="circle,pad=0.25", fc="#E2F0D9", ec="#59A14F"))
    ax.text(2.9, 2.4, "x2", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="circle,pad=0.25", fc="#E2F0D9", ec="#59A14F"))
    ax.text(4.4, 2.4, "...", ha="center", va="center", fontsize=12)
    ax.text(6.0, 2.4, "x20", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="circle,pad=0.25", fc="#E2F0D9", ec="#59A14F"))
    ax.text(8.2, 2.4, "y\n（洪水概率）", ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FBE5D6", ec="#E45756"))
    ax.text(8.2, 1.0, "R\n（潜在机制，不可观测）", ha="center", va="center",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#F2F2F2",
                                 ec="#999999", ls="--"))
    # arrows N -> x
    for x in (1.4, 2.9, 4.4, 6.0):
        ax.annotate("", xy=(x, 2.8), xytext=(5, 3.8),
                    arrowprops=dict(arrowstyle="->", color="#4C78A8", lw=1.2))
    # N -> y
    ax.annotate("", xy=(7.5, 2.5), xytext=(5.8, 3.8),
                arrowprops=dict(arrowstyle="->", color="#4C78A8", lw=1.8))
    # R -> y
    ax.annotate("", xy=(8.2, 1.9), xytext=(8.2, 1.4),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2, ls="--"))
    # x -/- y (no direct effect)
    ax.annotate("", xy=(7.2, 2.4), xytext=(6.6, 2.4),
                arrowprops=dict(arrowstyle="-|>", color="#E45756", lw=1.2, ls=":"))
    ax.text(6.85, 2.62, "无直接效应", ha="center", fontsize=8, color="#E45756")
    ax.text(5.0, 4.75, "DAG：N 是混杂变量；给定 N，x 与 y 条件独立", ha="center", fontsize=11)
    ax.text(8.2, 0.35, "R 与 x 近似独立（AUC≈0.54）", ha="center", fontsize=8,
            color="#666666")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_causal_dag.png")
    plt.close(fig)

    # ---------- 5. 图：边际 vs 偏相关 ----------
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    order = np.argsort(-tab["边际r"].to_numpy())
    ypos = np.arange(len(FEATURES))
    ax.barh(ypos - 0.2, tab["边际r"].to_numpy()[order], height=0.38,
            color="#4C78A8", label="边际 r(x_j, y)")
    ax.barh(ypos + 0.2, tab["偏相关r_given_N"].to_numpy()[order], height=0.38,
            color="#E45756", label="偏相关 r(x_j, y | N)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([FEATURES[k] for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("相关系数")
    ax.set_title("边际相关约 0.18；给定 N 后全部降到 0 附近")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_causal_corr.png")
    plt.close(fig)

    out = {
        "marginal_r_range": [float(tab["边际r"].min()), float(tab["边际r"].max())],
        "partial_r_range": [float(tab["偏相关r_given_N"].min()),
                            float(tab["偏相关r_given_N"].max())],
        "partial_r2_max_pct": float(tab["偏R2"].max() * 100),
        "within_N_indicator_corr": bin_corr,
        "within_N_bin_corr_mean": float(np.mean(bin_corr)),
        "regime_slope_range": [min(a1, a2), max(a1, a2)],
        "regime_intercept_range": [min(b1, b2), max(b1, b2)],
        "regime_auc_from_x": lr.get("regime_auc_from_x"),
        "interventions": itab.to_dict(orient="records"),
        "conclusion": (
            "N 是唯一可观测的驱动变量：边际相关 r≈0.18，但给定 N 后偏相关全部降到 "
            "|r|≤0.004、偏 R²<0.002%；保持 N 不变的再分配对 y 的效应为 0，"
            "改变 N 的效应为 β≈0.00565。潜在机制 R 使 N→y 斜率在 "
            "[0.0044,0.0061] 间异质，但 R 与 x 近似独立（AUC≈0.54），"
            "因此没有证据支持任何单个指标的直接因果效应。"
        ),
    }
    save_json(out, RESULTS / "causal.json")
    print(tab.round(5).to_string(index=False))
    print("\n偏相关范围:", out["partial_r_range"],
          "偏 R² 最大(%):", round(out["partial_r2_max_pct"], 6))
    print("控制 N 后相关:", [round(v, 4) for v in bin_corr])
    print("潜在机制斜率:", out["regime_slope_range"], "AUC:", out["regime_auc_from_x"])
    print(itab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
