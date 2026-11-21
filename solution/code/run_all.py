"""一键复现：依次运行 EDA、结构分析、问题 1–4、表格生成。

用法：
    /home/shaun/miniforge3/bin/python run_all.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
LOG = ROOT / "results" / "logs" / "run_all.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    ("eda.py", "数据总览与 EDA"),
    ("structure.py", "数据结构与生成机制"),
    ("problem1.py", "问题 1：指标关联分析"),
    ("problem1_inference.py", "问题 1 补充：等权检验与 Shapley 分解"),
    ("problem2.py", "问题 2：风险分级与 TOPSIS"),
    ("problem3.py", "问题 3：预测模型与 5 指标"),
    ("problem4.py", "问题 4：预测与分布检验"),
    ("make_tables.py", "生成论文表格"),
]


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    t_all = time.time()
    for script, desc in SCRIPTS:
        t0 = time.time()
        msg = f"\n===== {desc} ({script}) ====="
        print(msg, flush=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
            r = subprocess.run(
                [sys.executable, str(CODE / script)],
                cwd=CODE, stdout=f, stderr=subprocess.STDOUT,
            )
        dt = time.time() - t0
        status = "OK" if r.returncode == 0 else f"FAILED({r.returncode})"
        print(f"[{status}] {script}: {dt:.1f}s", flush=True)
        if r.returncode != 0:
            print(f"详见 {LOG}", flush=True)
            sys.exit(r.returncode)

    print(f"\n全部完成，总耗时 {time.time() - t_all:.1f}s")
    print(f"日志：{LOG}")
    print("\n关键结果：")
    for name in ("eda_summary", "structure", "problem1", "problem1_inference",
                 "problem2", "problem3", "problem4"):
        p = ROOT / "results" / f"{name}.json"
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            if name == "eda_summary":
                print(f"  R²(线性)={d['ols']['r2_train']:.4f}，"
                      f"噪声 std={d['ols']['resid_std']:.4f}")
            elif name == "structure":
                print(f"  行总和 N={d['row_sum']['mean']:.2f}±{d['row_sum']['std']:.2f}，"
                      f"y~N 的 R²={d['target_decomposition']['r2']:.4f}")
            elif name == "problem1_inference":
                eq = d["equal_weight_test"]
                sh = d["shapley"]
                print(f"  等权检验 F={eq['F']:.2f}, p={eq['p_value']:.1e}, "
                      f"ΔR²={eq['delta_r2']:.1e}; "
                      f"Shapley 极差/均值={sh['range_over_mean']:.2%}")
            elif name == "problem2":
                print(f"  TOPSIS Spearman={d['topsis']['spearman_vs_target']:.4f}，"
                      f"KMeans ARI={d['kmeans']['ari_vs_target_tercile']:.4f}")
            elif name == "problem3":
                print(f"  OLS RMSE={d['final_model']['valid_rmse']:.5f}，"
                      f"最优 5 指标 RMSE={d['five_feature']['best']['val_rmse']:.5f}")
            elif name == "problem4":
                print(f"  预测均值={d['prediction_stats']['mean']:.4f}，"
                      f"峰度={d['normality']['kurtosis_excess']:.4f}")
    print(f"\n提交文件：{ROOT / 'submit.csv'}")


if __name__ == "__main__":
    main()
