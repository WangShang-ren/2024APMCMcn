"""问题 3：洪水概率预测模型 + 仅用 5 个关键指标的改进模型。

内容：
1. 80/20 留出集，比较：均值基线、等权指数、OLS、Ridge、随机森林、HistGradientBoosting；
2. 标准化系数 + 置换重要性；
3. 穷举 C(20,5)=15504 个五指标组合，给出最优/最差/分布；
4. 贪心前向选择与学习曲线；
5. 保存最终模型（OLS 全 20 指标）供问题 4 使用。

产出：
- results/problem3.json
- results/tables/p3_models.csv
- results/tables/p3_feature_importance.csv
- results/tables/p3_five_feature.csv
- results/tables/p3_learning_curve.csv
- results/final_model.npz
- figures/fig_p3_model_compare.png
- figures/fig_p3_learning_curve.png
- figures/fig_p3_five_feature.png
- figures/fig_p3_pred_vs_true.png
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from common import (
    FIGURES, TABLES, RESULTS, FEATURES, TARGET, Timer, load_train,
    save_json, rmse, mae, r2_score, setup_plot,
)

RNG = np.random.default_rng(42)


def fit_ols_centered(X, y):
    Xm = X.mean(axis=0)
    ym = y.mean()
    coef, *_ = np.linalg.lstsq(X - Xm, y - ym, rcond=None)
    intercept = float(ym - Xm @ coef)
    return coef, intercept


def eval_pred(y, p):
    return {"rmse": rmse(y, p), "mae": mae(y, p), "r2": r2_score(y, p)}


def main() -> None:
    setup_plot()
    import matplotlib.pyplot as plt

    with Timer("读取数据"):
        tr = load_train()
    X = tr[FEATURES].to_numpy(dtype=float)
    y = tr[TARGET].to_numpy(dtype=float)
    p = len(FEATURES)

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    ntr, nva = len(ytr), len(yva)

    models = {}

    # 0. 均值基线
    models["均值基线"] = np.full(nva, ytr.mean())

    # 1. 等权指数（对行均值做一元线性回归）
    xbar_tr = Xtr.mean(axis=1)
    A = np.vstack([xbar_tr, np.ones_like(xbar_tr)]).T
    (a_eq, b_eq), *_ = np.linalg.lstsq(A, ytr, rcond=None)
    models["等权指数"] = a_eq * Xva.mean(axis=1) + b_eq

    # 2. OLS
    coef, intercept = fit_ols_centered(Xtr, ytr)
    models["OLS(20指标)"] = Xva @ coef + intercept

    # 3. Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xtr, ytr)
    models["Ridge(α=1)"] = ridge.predict(Xva)

    # 4. 随机森林（子样本，控制时间）
    with Timer("随机森林"):
        rs = RNG.choice(ntr, size=min(150_000, ntr), replace=False)
        rf = RandomForestRegressor(
            n_estimators=80, max_depth=18, min_samples_leaf=20,
            n_jobs=-1, random_state=42,
        )
        rf.fit(Xtr[rs], ytr[rs])
        models["随机森林"] = rf.predict(Xva)

    # 5. HistGradientBoosting（子样本）
    with Timer("HistGradientBoosting"):
        hs = RNG.choice(ntr, size=min(400_000, ntr), replace=False)
        hgb = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.1, random_state=42,
        )
        hgb.fit(Xtr[hs], ytr[hs])
        models["HistGB"] = hgb.predict(Xva)

    rows = []
    for name, pred in models.items():
        m = eval_pred(yva, pred)
        rows.append({"模型": name, **m})
    model_tab = pd.DataFrame(rows).sort_values("rmse")
    model_tab.to_csv(TABLES / "p3_models.csv", index=False, encoding="utf-8-sig")

    # ---------- 特征重要性 ----------
    with Timer("置换重要性"):
        m = 200_000
        idx = RNG.choice(nva, size=min(m, nva), replace=False)
        Xp, yp = Xva[idx], yva[idx]
        base = rmse(yp, Xp @ coef + intercept)
        reps = 10
        perm = np.zeros((reps, p))
        for r in range(reps):
            for j in range(p):
                Xq = Xp.copy()
                RNG.shuffle(Xq[:, j])
                perm[r, j] = rmse(yp, Xq @ coef + intercept) - base
        perm_mean = perm.mean(axis=0)
        perm_std = perm.std(axis=0, ddof=1)

    std_x = Xtr.std(axis=0, ddof=1)
    std_y = ytr.std(ddof=1)
    imp = pd.DataFrame({
        "指标": FEATURES,
        "ols_coef": coef,
        "std_coef": coef * std_x / std_y,
        "perm_rmse_increase": perm_mean,
        "perm_rmse_std": perm_std,
    }).sort_values("perm_rmse_increase", ascending=False)
    imp.to_csv(TABLES / "p3_feature_importance.csv", index=False,
               encoding="utf-8-sig")

    # ---------- 5 指标穷举（利用充分统计量） ----------
    with Timer("5 指标穷举"):
        Xtr_c = Xtr - Xtr.mean(axis=0)
        ytr_c = ytr - ytr.mean()
        Gtr = Xtr_c.T @ Xtr_c
        btr = Xtr_c.T @ ytr_c
        yytr = float(ytr_c @ ytr_c)

        Xva_c = Xva - Xtr.mean(axis=0)
        yva_c = yva - ytr.mean()
        Gva = Xva_c.T @ Xva_c
        bva = Xva_c.T @ yva_c
        yyva = float(yva_c @ yva_c)
        var_yva = float(np.var(yva))

        recs = []
        for combo in combinations(range(p), 5):
            S = list(combo)
            try:
                c = np.linalg.solve(Gtr[np.ix_(S, S)], btr[S])
            except np.linalg.LinAlgError:
                continue
            mse = (yyva - 2 * c @ bva[S] + c @ Gva[np.ix_(S, S)] @ c) / nva
            r2_tr = float((c @ btr[S]) / yytr)
            recs.append({
                "subset": ",".join(FEATURES[j] for j in S),
                "train_r2": r2_tr,
                "val_rmse": float(np.sqrt(max(mse, 0))),
                "val_r2": float(1 - mse / var_yva),
            })
        five = pd.DataFrame(recs)
        five.to_csv(TABLES / "p3_five_feature.csv", index=False,
                    encoding="utf-8-sig")

        best5 = five.loc[five["val_rmse"].idxmin()].to_dict()
        worst5 = five.loc[five["val_rmse"].idxmax()].to_dict()
        # 按边际相关选出的 5 个
        r = np.array([abs(np.corrcoef(Xtr[:, j], ytr)[0, 1]) for j in range(p)])
        top_marg = list(np.argsort(-r)[:5])
        S = top_marg
        c = np.linalg.solve(Gtr[np.ix_(S, S)], btr[S])
        mse = (yyva - 2 * c @ bva[S] + c @ Gva[np.ix_(S, S)] @ c) / nva
        marg5 = {
            "subset": ",".join(FEATURES[j] for j in S),
            "val_rmse": float(np.sqrt(max(mse, 0))),
            "val_r2": float(1 - mse / var_yva),
        }

    # ---------- 贪心前向选择 ----------
    chosen, remaining = [], list(range(p))
    greedy = []
    for k in range(1, p + 1):
        best_j, best_rmse = None, np.inf
        for j in remaining:
            S = chosen + [j]
            c = np.linalg.solve(Gtr[np.ix_(S, S)], btr[S])
            mse = (yyva - 2 * c @ bva[S] + c @ Gva[np.ix_(S, S)] @ c) / nva
            if mse < best_rmse:
                best_rmse, best_j = mse, j
        chosen.append(best_j)
        remaining.remove(best_j)
        greedy.append({
            "k": k, "added": FEATURES[best_j],
            "val_rmse": float(np.sqrt(max(best_rmse, 0))),
        })

    # ---------- 学习曲线 ----------
    sizes = [10_000, 30_000, 100_000, 300_000, 600_000, ntr]
    lc = []
    for s in sizes:
        sub = RNG.choice(ntr, size=s, replace=False)
        c_s, b_s = fit_ols_centered(Xtr[sub], ytr[sub])
        lc.append({
            "n_train": int(s),
            "rmse": rmse(yva, Xva @ c_s + b_s),
            "r2": r2_score(yva, Xva @ c_s + b_s),
        })

    # ---------- 保存最终模型 ----------
    np.savez(
        RESULTS / "final_model.npz",
        coef=coef, intercept=intercept, features=np.array(FEATURES),
        rmse_val=rmse(yva, models["OLS(20指标)"]),
    )

    # ---------- 图 ----------
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    order = model_tab.sort_values("rmse", ascending=True)
    ax.barh(np.arange(len(order)), order["rmse"], color="#4C78A8", alpha=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order["模型"])
    ax.set_xlabel("验证集 RMSE")
    ax.set_title("模型对比：线性模型已达噪声下限，非线性模型无增益")
    ax.set_xlim(0, max(order["rmse"]) * 1.15)
    for i, v in enumerate(order["rmse"]):
        ax.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p3_model_compare.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(five["val_rmse"], bins=60, color="#F58518", alpha=0.85)
    ax.axvline(best5["val_rmse"], color="g", ls="--", lw=1.5,
               label=f"最优 5 指标 RMSE={best5['val_rmse']:.4f}")
    ax.axvline(marg5["val_rmse"], color="b", ls=":", lw=1.5,
               label=f"边际相关 Top5 RMSE={marg5['val_rmse']:.4f}")
    ax.set_xlim(five["val_rmse"].min() - 0.00005,
                five["val_rmse"].max() + 0.00005)
    ax.text(0.02, 0.95,
            f"20 指标 RMSE={rmse(yva, models['OLS(20指标)']):.4f}\n（远低于本图范围）",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_xlabel("验证集 RMSE")
    ax.set_ylabel("组合数")
    ax.set_title("全部 15504 个五指标组合：RMSE 集中在 0.0460–0.0465")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p3_five_feature.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot([d["n_train"] for d in lc], [d["rmse"] for d in lc],
            "o-", color="#4C78A8")
    ax.set_xscale("log")
    ax.set_xlabel("训练样本数")
    ax.set_ylabel("验证集 RMSE")
    ax.set_title("学习曲线：1 万样本已接近饱和（模型仅 20 个参数）")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p3_learning_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    hb = ax.hexbin(yva, models["OLS(20指标)"], gridsize=55,
                   cmap="viridis", mincnt=1)
    lo, hi = 0.25, 0.75
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实洪水概率")
    ax.set_ylabel("预测洪水概率")
    ax.set_title(f"OLS 验证集：R²={r2_score(yva, models['OLS(20指标)']):.4f}")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_p3_pred_vs_true.png")
    plt.close(fig)

    out = {
        "n_train": int(ntr), "n_valid": int(nva),
        "models": model_tab.to_dict(orient="records"),
        "feature_importance": imp.to_dict(orient="records"),
        "five_feature": {
            "n_subsets": int(len(five)),
            "best": best5,
            "worst": worst5,
            "marginal_top5": marg5,
            "val_rmse_min": float(five["val_rmse"].min()),
            "val_rmse_max": float(five["val_rmse"].max()),
            "val_rmse_median": float(five["val_rmse"].median()),
        },
        "greedy": greedy,
        "learning_curve": lc,
        "final_model": {
            "type": "OLS(20指标)",
            "coef": coef.tolist(),
            "intercept": intercept,
            "features": FEATURES,
            "valid_rmse": rmse(yva, models["OLS(20指标)"]),
            "valid_r2": r2_score(yva, models["OLS(20指标)"]),
        },
        "conclusion": (
            "20 指标线性模型（等价于等权指数）在验证集上达到 RMSE≈0.0201、"
            "R²≈0.845，已触及噪声下限；随机森林/HistGB 无显著增益。"
            "仅用 5 个指标时，即使穷举最优组合，RMSE 也显著上升到约 0.046，"
            "说明 20 个指标等权共同贡献，无法用 5 个指标无损替代。"
        ),
    }
    save_json(out, RESULTS / "problem3.json")

    print(model_tab.round(5).to_string(index=False))
    print(f"\n最优 5 指标: {best5['subset']}  RMSE={best5['val_rmse']:.4f}")
    print(f"最差 5 指标: {worst5['subset']}  RMSE={worst5['val_rmse']:.4f}")
    print(f"边际 Top5: {marg5['subset']}  RMSE={marg5['val_rmse']:.4f}")
    print(f"5 指标 RMSE 中位数: {five['val_rmse'].median():.4f}")
    print("贪心前 5:", [d["added"] for d in greedy[:5]])


if __name__ == "__main__":
    main()
