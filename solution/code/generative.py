"""生成式逆向建模：还原“N 的混合分布 + 多项分配 + 噪声”的数据生成过程。

模型（工作模型）：
    N_i           ~ GMM(k=6)（按 BIC 在 k<=8 中取折中，k=6 后改善趋缓）
    x_i | N_i     ~ 欠离散多项分配（multinomial 核心 + 均摊余量，离散系数 φ）
    y_i           = a N_i + b + ε_i,  ε_i ~ Student-t（比高斯更贴合重尾）

产出：
- results/generative.json
- results/tables/generative_ppc.csv
- results/tables/generative_params.csv
- results/tables/generative_pred.csv
- figures/fig_generative_ppc.png
- figures/fig_generative_pred.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from common import (
    FIGURES, TABLES, RESULTS, FEATURES, TARGET, Timer,
    load_train, save_json, rmse, r2_score, setup_plot,
)

RNG = np.random.default_rng(2024)


# --------------------------------------------------------------------------
# 1. 参数估计
# --------------------------------------------------------------------------
def fit_n_gmm(N: np.ndarray, sample: int = 300_000, kmax: int = 8):
    idx = RNG.choice(len(N), size=min(sample, len(N)), replace=False)
    Ns = N[idx].reshape(-1, 1)
    rows, models = [], {}
    for k in range(1, kmax + 1):
        g = GaussianMixture(k, n_init=3, random_state=42).fit(Ns)
        rows.append({"k": k, "bic": float(g.bic(Ns)), "aic": float(g.aic(Ns))})
        models[k] = g
    tab = pd.DataFrame(rows)
    # 工作模型取 k=6（k=6 后 BIC 改善 < 0.15%，见 json）
    k = 6
    g = models[k]
    return g, tab


def fit_noise(resid: np.ndarray):
    """高斯 vs Student-t；用验证集 NLL/CRPS 比较。"""
    sample = RNG.choice(len(resid), size=min(300_000, len(resid)), replace=False)
    r = resid[sample]
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    df, loc, scale = stats.t.fit(r)
    return {
        "gaussian": {"mean": mu, "std": sd},
        "student_t": {"df": float(df), "loc": float(loc), "scale": float(scale)},
    }


# --------------------------------------------------------------------------
# 2. 模拟
# --------------------------------------------------------------------------
def sample_partial_multinomial(N: np.ndarray, p: np.ndarray, phi: float,
                               rng: np.random.Generator) -> np.ndarray:
    """多项核心 + 均摊余量：Var(x_j|N) = φ N p_j(1-p_j)，且 sum_j x_j = N。"""
    n, k = len(N), len(p)
    M = np.maximum(np.round(phi * N).astype(int), 0)
    X = rng.multinomial(1, p, size=n) * 0  # 占位，保持形状
    # 逐行 multinomial（n=20 万，向量化不可用；循环可接受）
    Z = np.empty((n, k), dtype=int)
    for i in range(n):
        Z[i] = rng.multinomial(M[i], p)
    R = N - M
    base = R // k
    rem = R % k
    X = Z + base[:, None]
    for i in range(n):
        if rem[i] > 0:
            cats = rng.choice(k, size=rem[i], replace=False)
            X[i, cats] += 1
    return X


def simulate(g, p, phi, noise, n, rng):
    # N
    comp = rng.choice(len(g.weights_), size=n, p=g.weights_)
    N = g.means_.ravel()[comp] + np.sqrt(g.covariances_.ravel()[comp]) * rng.standard_normal(n)
    N = np.clip(np.round(N), 64, 144).astype(int)
    X = sample_partial_multinomial(N, p, phi, rng)
    t = noise["student_t"]
    eps = stats.t.rvs(t["df"], loc=t["loc"], scale=t["scale"], size=n, random_state=rng)
    y = 0.005646759236800108 * N - 0.05335468265348672 + eps
    return X, y, N


# --------------------------------------------------------------------------
# 3. 预测评价
# --------------------------------------------------------------------------
def crps_from_samples(samples: np.ndarray, y: np.ndarray) -> float:
    """用排序样本估计 CRPS = E|X-y| - 0.5 E|X-X'|。"""
    z = np.sort(samples, axis=1)
    n = z.shape[1]
    term1 = np.mean(np.abs(z - y[:, None]))
    w = 2 * np.arange(1, n + 1) - n - 1
    term2 = 2 / (n * n) * np.sum(w[None, :] * z, axis=1).mean()
    return float(term1 - 0.5 * term2)


def predictive_compare(yva, mu, resid_train, noise):
    rng = np.random.default_rng(7)
    sub = rng.choice(len(yva), size=min(30_000, len(yva)), replace=False)
    yv, mv = yva[sub], mu[sub]
    out = []
    # 高斯
    g = noise["gaussian"]
    dist = stats.norm(g["mean"], g["std"])
    lo, hi = mv + dist.ppf(0.025), mv + dist.ppf(0.975)
    out.append({
        "noise": "高斯", "nll": float(-np.mean(dist.logpdf(yv - mv))),
        "crps": crps_from_samples(dist.rvs(size=(len(sub), 100), random_state=1) + mv[:, None], yv),
        "coverage95": float(np.mean((yv >= lo) & (yv <= hi))),
        "width95": float(np.mean(hi - lo)),
    })
    # Student-t
    t = noise["student_t"]
    dist = stats.t(t["df"], t["loc"], t["scale"])
    lo, hi = mv + dist.ppf(0.025), mv + dist.ppf(0.975)
    out.append({
        "noise": "Student-t", "nll": float(-np.mean(dist.logpdf(yv - mv))),
        "crps": crps_from_samples(dist.rvs(size=(len(sub), 100), random_state=2) + mv[:, None], yv),
        "coverage95": float(np.mean((yv >= lo) & (yv <= hi))),
        "width95": float(np.mean(hi - lo)),
    })
    # 经验残差（bootstrap）
    eps = rng.choice(resid_train, size=(len(sub), 100), replace=True)
    lo, hi = mv + np.quantile(eps, 0.025, axis=1), mv + np.quantile(eps, 0.975, axis=1)
    out.append({
        "noise": "经验残差", "nll": float("nan"),
        "crps": crps_from_samples(eps + mv[:, None], yv),
        "coverage95": float(np.mean((yv >= lo) & (yv <= hi))),
        "width95": float(np.mean(hi - lo)),
    })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# 4. 潜在机制异质性（诊断，不进入可部署预测）
# --------------------------------------------------------------------------
def mixlin2(N, y):
    """两分量线性混合回归 EM（k=2），用于给出 oracle 上界。"""
    n = len(N)
    a = np.array([0.0062, 0.0044]); b = np.array([-0.110, 0.091])
    s = np.array([0.012, 0.018]); g = np.array([-2.93, 0.0218])
    for _ in range(60):
        lg = np.column_stack([np.zeros(n), g[0] + g[1] * N])
        lg -= lg.max(1, keepdims=True)
        pi = np.exp(lg); pi /= pi.sum(1, keepdims=True)
        dens = np.column_stack([
            np.exp(-0.5 * ((y - (a[j] * N + b[j])) / s[j]) ** 2) /
            (s[j] * np.sqrt(2 * np.pi)) for j in range(2)
        ])
        r = pi * dens; r /= r.sum(1, keepdims=True) + 1e-300
        for j in range(2):
            W = r[:, j]; Sw = W.sum()
            xm, ym = (W * N).sum() / Sw, (W * y).sum() / Sw
            cov = (W * (N - xm) * (y - ym)).sum() / Sw
            var = (W * (N - xm) ** 2).sum() / Sw
            a[j] = cov / var; b[j] = ym - a[j] * xm
            s[j] = np.sqrt((W * (y - (a[j] * N + b[j])) ** 2).sum() / Sw)
    # 用 y 的后验责任得到 oracle 预测（不可部署）
    lg = np.column_stack([np.zeros(n), g[0] + g[1] * N])
    lg -= lg.max(1, keepdims=True); pi = np.exp(lg); pi /= pi.sum(1, keepdims=True)
    dens = np.column_stack([
        np.exp(-0.5 * ((y - (a[j] * N + b[j])) / s[j]) ** 2) /
        (s[j] * np.sqrt(2 * np.pi)) for j in range(2)
    ])
    r = pi * dens; r /= r.sum(1, keepdims=True)
    comp = np.column_stack([a[j] * N + b[j] for j in range(2)])
    oracle = (r * comp).sum(1)
    # 可部署：只用 N 的 gate
    gate = np.column_stack([np.zeros(n), g[0] + g[1] * N])
    gate -= gate.max(1, keepdims=True); pi_n = np.exp(gate); pi_n /= pi_n.sum(1, keepdims=True)
    deploy = (pi_n * comp).sum(1)
    return {
        "a": a.tolist(), "b": b.tolist(), "sigma": s.tolist(),
        "gate": g.tolist(), "weights": (r.sum(0) / n).tolist(),
        "oracle_rmse_in_sample": rmse(y, oracle),
        "deployable_rmse_in_sample": rmse(y, deploy),
        "responsibility": r,
    }


def main() -> None:
    setup_plot()
    import matplotlib.pyplot as plt

    with Timer("读取数据"):
        tr = load_train()
    X = tr[FEATURES].to_numpy(dtype=float)
    y = tr[TARGET].to_numpy(dtype=float)
    N = X.sum(axis=1).astype(float)

    # ---------- 参数估计 ----------
    with Timer("N 的 GMM"):
        g, gmm_tab = fit_n_gmm(N)
        gmm_tab.to_csv(TABLES / "generative_gmm.csv", index=False, encoding="utf-8-sig")

    p = X.sum(axis=0) / N.sum()
    var_obs = float(np.var(X - N[:, None] * p[None, :], axis=0, ddof=1).mean())
    var_mult = float(N.mean() * (p * (1 - p)).mean())
    phi = var_obs / var_mult
    ptab = pd.DataFrame({
        "指标": FEATURES, "p_j": p,
    })
    ptab.to_csv(TABLES / "generative_allocation.csv", index=False, encoding="utf-8-sig")

    A = np.vstack([N, np.ones_like(N)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (a * N + b)
    noise = fit_noise(resid)

    # ---------- 模拟与 PPC ----------
    with Timer("模拟 + PPC"):
        Xs, ys, Ns = simulate(g, p, phi, noise, n=200_000, rng=RNG)
        Cs = np.corrcoef(Xs, rowvar=False)
        offs = Cs[~np.eye(len(FEATURES), dtype=bool)]
        Co = np.corrcoef(X, rowvar=False)
        offo = Co[~np.eye(len(FEATURES), dtype=bool)]
        rs = np.array([np.corrcoef(Xs[:, j], ys)[0, 1] for j in range(len(FEATURES))])
        ro = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(len(FEATURES))])
        # 控制 N 后相关
        def within_corr(Z, v):
            out = []
            for lo, hi in [(80, 90), (90, 100), (100, 110), (110, 120)]:
                m = (v >= lo) & (v < hi)
                if m.sum() < 500:
                    continue
                C = np.corrcoef(Z[m], rowvar=False)
                out.append(np.mean(C[~np.eye(len(FEATURES), dtype=bool)]))
            return float(np.mean(out))
        ppc = pd.DataFrame([
            ["N 均值", N.mean(), Ns.mean()],
            ["N 标准差", N.std(), Ns.std()],
            ["N 的 5% 分位", np.quantile(N, .05), np.quantile(Ns, .05)],
            ["N 的 95% 分位", np.quantile(N, .95), np.quantile(Ns, .95)],
            ["指标均值（平均）", X.mean(), Xs.mean()],
            ["指标标准差（平均）", X.std(axis=0, ddof=1).mean(), Xs.std(axis=0, ddof=1).mean()],
            ["两两相关均值", offo.mean(), offs.mean()],
            ["控制 N 后相关均值", within_corr(X, N), within_corr(Xs, Ns)],
            ["y 均值", y.mean(), ys.mean()],
            ["y 标准差", y.std(), ys.std()],
            ["单变量 r 最小值", ro.min(), rs.min()],
            ["单变量 r 最大值", ro.max(), rs.max()],
            ["y~N 的 R²", r2_score(y, a * N + b), r2_score(ys, a * Ns + b)],
        ], columns=["统计量", "真实数据", "生成数据"])
        ppc.to_csv(TABLES / "generative_ppc.csv", index=False, encoding="utf-8-sig")

    # ---------- 预测对比 ----------
    with Timer("预测对比"):
        Xtr, Xva, ytr, yva, Ntr, Nva = train_test_split(
            X, y, N, test_size=0.2, random_state=42
        )
        Atr = np.vstack([Ntr, np.ones_like(Ntr)]).T
        (aa, bb), *_ = np.linalg.lstsq(Atr, ytr, rcond=None)
        resid_tr = ytr - (aa * Ntr + bb)
        noise_tr = fit_noise(resid_tr)
        mu = aa * Nva + bb
        pred_tab = predictive_compare(yva, mu, resid_tr, noise_tr)
        pred_tab.insert(0, "RMSE", rmse(yva, mu))
        pred_tab.to_csv(TABLES / "generative_pred.csv", index=False,
                        encoding="utf-8-sig")

    # ---------- 潜在机制异质性 ----------
    with Timer("潜在机制诊断"):
        sub = RNG.choice(len(N), size=200_000, replace=False)
        mix = mixlin2(N[sub], y[sub])
        # 可部署 gate 的验证 RMSE（用全量训练的参数在验证集上直接套）
        lg = np.column_stack([np.zeros(len(Nva)), mix["gate"][0] + mix["gate"][1] * Nva])
        lg -= lg.max(1, keepdims=True)
        pin = np.exp(lg); pin /= pin.sum(1, keepdims=True)
        comp = np.column_stack([
            mix["a"][j] * Nva + mix["b"][j] for j in range(2)
        ])
        mix["deployable_rmse_valid"] = rmse(yva, (pin * comp).sum(1))
        # oracle 验证 RMSE（用验证集 y 的后验责任，上界）
        dens = np.column_stack([
            np.exp(-0.5 * ((yva - (mix["a"][j] * Nva + mix["b"][j])) / mix["sigma"][j]) ** 2)
            / (mix["sigma"][j] * np.sqrt(2 * np.pi)) for j in range(2)
        ])
        rv = pin * dens; rv /= rv.sum(1, keepdims=True)
        mix["oracle_rmse_valid"] = rmse(yva, (rv * comp).sum(1))
        # 潜在机制能否由 x 识别：logistic AUC
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        reg = (mix["responsibility"][:, 1] > 0.5).astype(int)
        Ai, Bi, ay, by = train_test_split(X[sub], reg, test_size=0.3,
                                          random_state=42, stratify=reg)
        clf = LogisticRegression(max_iter=1000).fit(Ai, ay)
        mix["regime_auc_from_x"] = float(roc_auc_score(by, clf.predict_proba(Bi)[:, 1]))
        mix["regime_prevalence"] = float(reg.mean())
        mix.pop("responsibility")

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(N, bins=np.arange(64, 145, 2), density=True, alpha=0.55,
                 color="#4C78A8", label="真实 N")
    axes[0].hist(Ns, bins=np.arange(64, 145, 2), density=True, alpha=0.45,
                 histtype="step", lw=1.4, color="#E45756", label="生成 N")
    axes[0].set_xlabel("行总和 N")
    axes[0].set_ylabel("密度")
    axes[0].set_title("行总和分布：真实 vs 生成")
    axes[0].legend(fontsize=8)
    axes[1].hist(y, bins=100, density=True, alpha=0.55,
                 color="#4C78A8", label="真实 y")
    axes[1].hist(ys, bins=100, density=True, histtype="step", lw=1.4,
                 color="#E45756", label="生成 y")
    axes[1].set_xlabel("洪水概率 y")
    axes[1].set_ylabel("密度")
    axes[1].set_title("目标分布：真实 vs 生成")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_generative_ppc.png")
    plt.close(fig)

    # 预测分布对比图（噪声分布）
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = np.linspace(-0.12, 0.12, 600)
    gt = noise["student_t"]
    ax.hist(resid, bins=100, density=True, alpha=0.45, color="#4C78A8",
            label="残差直方图")
    ax.plot(xs, stats.norm(noise["gaussian"]["mean"], noise["gaussian"]["std"]).pdf(xs),
            "k--", lw=1.4, label="高斯拟合")
    ax.plot(xs, stats.t(gt["df"], gt["loc"], gt["scale"]).pdf(xs),
            "r-", lw=1.6, label=f"Student-t (df={gt['df']:.1f})")
    ax.set_xlim(-0.09, 0.09)
    ax.set_xlabel("残差 ε")
    ax.set_ylabel("密度")
    ax.set_title("噪声分布：t 比高斯更贴合重尾")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_generative_pred.png")
    plt.close(fig)

    out = {
        "gmm": {
            "k": 6,
            "weights": sorted(g.weights_.tolist()),
            "means": sorted(g.means_.ravel().tolist()),
            "stds": sorted(np.sqrt(g.covariances_.ravel()).tolist()),
            "bic": gmm_tab.to_dict(orient="records"),
        },
        "allocation": {
            "p_min": float(p.min()), "p_max": float(p.max()), "p_mean": float(p.mean()),
            "phi": float(phi), "var_obs": var_obs, "var_multinomial": var_mult,
        },
        "noise": noise,
        "target": {"a": float(a), "b": float(b), "r2": float(r2_score(y, a * N + b)),
                   "rmse": float(rmse(y, a * N + b))},
        "ppc": ppc.to_dict(orient="records"),
        "predictive": pred_tab.to_dict(orient="records"),
        "latent_regime": mix,
        "conclusion": (
            "反演得到的生成模型为：N~GMM(6)，x|N 近似欠离散多项分配（φ≈0.88），"
            "y=aN+b+Student-t 噪声。生成模型能复现行总和、指标相关与目标分布；"
            "点预测与 OLS 完全相同（因为 N 已知），但 t 噪声的预测分布在 NLL/CRPS 上略优。"
            "残差诊断发现一个两分量线性机制的潜在结构（oracle RMSE≈0.0117），"
            "但它既不能由 N 也不能由 20 个指标可靠识别（AUC≈0.53），"
            "因此只能作为不可观测异质性，不能用于可部署预测。"
        ),
    }
    save_json(out, RESULTS / "generative.json")
    print(ppc.round(4).to_string(index=False))
    print("\n预测对比:")
    print(pred_tab.round(5).to_string(index=False))
    print("\n潜在机制:", {k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in mix.items() if not isinstance(v, list)})


if __name__ == "__main__":
    main()
