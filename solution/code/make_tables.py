"""把 results/ 中的结果整理成论文用 LaTeX 表格（paper/tables/*.tex）。"""
from __future__ import annotations

import json

import pandas as pd

from common import RESULTS, TABLES, PAPER

OUT = PAPER / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def sci(x: float, nd: int = 1) -> str:
    """把 1.2e-10 写成 LaTeX 的 1.2\\times10^{-10}。"""
    m, e = f"{x:.{nd}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def esc(s) -> str:
    """转义普通文本；$...$ 内的数学公式保持原样。"""
    s = str(s)
    parts = s.split("$")
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append("$" + part + "$")
            continue
        for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#")]:
            part = part.replace(a, b)
        out.append(part)
    return "".join(out)


def write_table(name: str, caption: str, label: str, colspec: str,
                header: list[str], rows: list[list], note: str = "",
                small: bool = True) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        (r"\small" if small else ""),
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(esc(x) for x in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\par\vspace{{2pt}}\footnotesize {note}")
    lines.append(r"\end{table}")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    eda = json.load(open(RESULTS / "eda_summary.json", encoding="utf-8"))
    st = json.load(open(RESULTS / "structure.json", encoding="utf-8"))
    p1 = json.load(open(RESULTS / "problem1.json", encoding="utf-8"))
    p1i = json.load(open(RESULTS / "problem1_inference.json", encoding="utf-8"))
    p2 = json.load(open(RESULTS / "problem2.json", encoding="utf-8"))
    p3 = json.load(open(RESULTS / "problem3.json", encoding="utf-8"))
    p4 = json.load(open(RESULTS / "problem4.json", encoding="utf-8"))

    # 1. 数据总览
    ts = eda["target_stats"]
    rows = [
        ["训练集", f"{eda['train_shape'][0]:,}", f"{eda['train_shape'][1]}"],
        ["测试集", f"{eda['test_shape'][0]:,}", f"{eda['test_shape'][1]}"],
        ["缺失值（特征）", eda["missing_train"], eda["missing_test_features"]],
        ["目标均值", f"{ts['mean']:.4f}", "—"],
        ["目标标准差", f"{ts['std']:.4f}", "—"],
        ["目标范围", f"[{ts['min']:.3f}, {ts['max']:.3f}]", "—"],
        ["指标范围", f"[{eda['feature_value_min']:.0f}, {eda['feature_value_max']:.0f}]",
         "同左"],
        ["指标均值范围", f"[{eda['feature_mean_min']:.3f}, {eda['feature_mean_max']:.3f}]",
         "同左"],
    ]
    write_table("tab_data", "数据规模与基本统计", "tab:data", "lrr",
                ["项目", "训练集", "测试集"], rows)

    # 2. 结构
    rs = st["row_sum"]
    td = st["target_decomposition"]
    cs = st["correlation_structure"]
    rows = [
        ["指标均值极差/均值", f"{st['exchangeability']['feature_mean_range_over_mean']*100:.2f}%"],
        ["指标标准差极差/均值", f"{st['exchangeability']['feature_std_range_over_mean']*100:.2f}%"],
        ["两两相关均值", f"{cs['mean_pairwise_r']:.4f}"],
        ["两两相关范围", f"[{cs['min_pairwise_r']:.4f}, {cs['max_pairwise_r']:.4f}]"],
        ["控制 N 后两两相关", f"{st['within_N_bins']['bins'][2]['mean_pairwise_r']:.4f}"],
        ["多项分配理论值 $-1/19$", f"{-1/19:.4f}"],
        ["行总和 $N$ 均值 $\\pm$ 标准差", f"{rs['mean']:.2f} $\\pm$ {rs['std']:.2f}"],
        ["行总和 N 范围", f"[{rs['min']:.0f}, {rs['max']:.0f}]"],
        ["$y\\approx aN+b$ 的 $R^2$", f"{td['r2']:.4f}"],
        ["信号标准差 / 噪声标准差", f"{td['signal_std']:.4f} / {td['noise_std']:.4f}"],
        ["噪声占方差比例", f"{td['noise_share_of_variance']*100:.2f}%"],
    ]
    write_table("tab_structure", "数据结构与生成机制的关键证据", "tab:structure",
                "lr", ["项目", "数值"], rows)

    # 3. 问题 1 重要性
    imp = pd.read_csv(TABLES / "p1_importance.csv", encoding="utf-8-sig")
    rows = []
    for _, r in imp.iterrows():
        rows.append([
            r["指标"], f"{r['pearson_r']:.4f}", f"{r['mutual_info']:.4f}",
            f"{r['perm_rmse_increase']:.4f}", f"{r['drop_one_r2']:.4f}",
        ])
    write_table("tab_importance", "20 个指标的关联强度（按 Pearson 相关排序）",
                "tab:importance", "lrrrr",
                ["指标", "$r$", "互信息", "置换 $\\Delta$RMSE", "留一 $\\Delta R^2$"],
                rows,
                note=f"单变量 $r\\in[{p1['pearson_r_range'][0]:.4f},"
                     f"{p1['pearson_r_range'][1]:.4f}]$；自助法排名 95\\% 区间平均宽度"
                     f"约 {p1['rank_interval_width_mean']:.1f}（满量程 20）。")

    # 3b. 问题 1 等权约束检验
    eq = p1i["equal_weight_test"]
    sh = p1i["shapley"]
    rows = [
        ["无约束 OLS（20 指标）", f"{eq['unrestricted']['r2']:.5f}",
         f"{eq['unrestricted']['rmse']:.6f}", f"{eq['unrestricted']['n_params']}",
         f"{eq['coef_rel_range'] * 100:.2f}%"],
        ["等权约束（行总和 $N$）", f"{eq['restricted']['r2']:.5f}",
         f"{eq['restricted']['rmse']:.6f}", "2", "0.00%"],
    ]
    write_table(
        "tab_p1equal", "等权约束与无约束 OLS 的对比", "tab:p1equal", "lrrrr",
        ["模型", "$R^2$", "RMSE", "参数个数", "系数极差/均值"], rows,
        note=(f"嵌套 F 检验 $F(19,1048554)={eq['F']:.2f}$，"
              f"$p={sci(eq['p_value'])}$；"
              f"$\\Delta R^2={sci(eq['delta_r2'])}$。"
              f"随机分半的系数排序 Spearman $\\rho={p1i['exchangeability']['split_half_spearman']:.3f}$；"
              f"Shapley $R^2$ 每项 {sh['min']:.4f}–{sh['max']:.4f}，"
              f"极差仅 {sh['range_over_mean'] * 100:.1f}\\% 于均值。"),
    )

    # 4. 问题 2 赋权
    rows = []
    for k, v in p2["weights"].items():
        rows.append([k, f"{v['min']:.4f}", f"{v['max']:.4f}",
                     f"{v['max_over_min']:.3f}", f"{v['std']:.5f}"])
    write_table("tab_weights", "四种赋权方法的权重分布", "tab:weights", "lrrrr",
                ["方法", "最小", "最大", "最大/最小", "标准差"], rows,
                note="理想等权为 1/20=0.05；PCA 因数据近似各向同性而不适用。")

    # 5. 问题 2 TOPSIS
    tp = p2["topsis"]
    rows = [
        ["Spearman（TOPSIS vs 目标）", f"{tp['spearman_vs_target']:.4f}"],
        ["Pearson（TOPSIS vs 目标）", f"{tp['pearson_vs_target']:.4f}"],
        ["线性校准后 R²", f"{tp['r2_after_linear_calibration']:.4f}"],
        ["三级预警一致率", f"{tp['grade_accuracy']*100:.2f}%"],
        ["权重扰动（$\\pm10$%）后 Spearman 均值", f"{p2['sensitivity']['spearman_mean']:.4f}"],
        ["单指标 $\\pm20$% 扰动后 Spearman 范围",
         f"[{p2['sensitivity']['oat_spearman_min']:.4f},"
         f"{p2['sensitivity']['oat_spearman_max']:.4f}]"],
        ["KMeans 与三分位的 ARI", f"{p2['kmeans']['ari_vs_target_tercile']:.4f}"],
        ["风险组 $\\eta^2$ 范围", f"[{p2['eta2_range'][0]:.4f}, {p2['eta2_range'][1]:.4f}]"],
    ]
    write_table("tab_topsis", "TOPSIS 预警模型与灵敏度", "tab:topsis", "lr",
                ["指标", "数值"], rows)

    # 6. 问题 3 模型
    rows = []
    for m in p3["models"]:
        name = m["模型"].replace("α=1", r"$\alpha=1$")
        rows.append([name, f"{m['rmse']:.5f}", f"{m['mae']:.5f}", f"{m['r2']:.4f}"])
    write_table("tab_models", "预测模型在 20\\% 验证集上的表现", "tab:models",
                "lrrr", ["模型", "RMSE", "MAE", "$R^2$"], rows)

    # 7. 五指标
    ff = p3["five_feature"]
    rows = [
        ["20 指标（OLS/等权）",
         f"{p3['final_model']['valid_rmse']:.5f}",
         f"{p3['final_model']['valid_r2']:.4f}", "全部 20 个"],
        ["最优 5 指标", f"{ff['best']['val_rmse']:.5f}",
         f"{ff['best']['val_r2']:.4f}", ff["best"]["subset"]],
        ["最差 5 指标", f"{ff['worst']['val_rmse']:.5f}",
         f"{ff['worst']['val_r2']:.4f}", ff["worst"]["subset"]],
        ["边际相关 Top5", f"{ff['marginal_top5']['val_rmse']:.5f}",
         f"{ff['marginal_top5']['val_r2']:.4f}", ff["marginal_top5"]["subset"]],
        ["全部 15504 组合中位数", f"{ff['val_rmse_median']:.5f}", "—", "—"],
    ]
    write_table("tab_five", "仅用 5 个指标的精度损失（穷举 15504 个组合）",
                "tab:five", "lrrp{0.42\\linewidth}",
                ["方案", "RMSE", "$R^2$", "指标组合"], rows,
                note="5 指标组合的 RMSE 全部落在 "
                     f"[{ff['val_rmse_min']:.4f}, {ff['val_rmse_max']:.4f}]，"
                     "说明任选 5 个指标的表现几乎相同。", small=True)

    # 8. 问题 4 正态性
    nm = p4["normality"]
    tt = nm["train_target"]
    ps = p4["prediction_stats"]
    rows = [
        ["均值", f"{ps['mean']:.4f}", f"{eda['target_stats']['mean']:.4f}"],
        ["标准差", f"{ps['std']:.4f}", f"{eda['target_stats']['std']:.4f}"],
        ["偏度", f"{nm['skew']:.4f}", f"{tt['skew']:.4f}"],
        ["峰度", f"{nm['kurtosis_excess']:.4f}", f"{tt['kurtosis_excess']:.4f}"],
        ["KS $D$（vs 正态）", f"{nm['ks_D_vs_fitted_normal']:.4f}",
         f"{tt['ks_D_vs_fitted_normal']:.4f}"],
        ["JB 统计量", f"{nm['jarque_bera_stat']:.1f}", "—"],
        ["Shapiro $W$（$n=5000$）", f"{nm['shapiro_W_5000']:.4f}", "—"],
    ]
    write_table("tab_normality", "问题 4 预测分布与观测目标的正态性对比",
                "tab:normality", "lrr",
                ["统计量", "预测概率（信号）", "训练集观测值（信号+噪声）"], rows,
                note="预测信号峰度约 0.84，明显偏离正态；观测值叠加噪声后峰度约 $-0.04$。")

    print("表格已写入", OUT)


if __name__ == "__main__":
    main()
