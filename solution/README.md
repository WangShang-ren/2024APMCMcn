# APMCM 2024 中文赛项 B 题 · 洪水灾害的数据分析与预测

本目录是 B 题的完整解题工程，包含数据、代码、结果、图表、论文与提交文件。

## 1. 核心发现

对 `train.csv` 的 104.9 万条数据做取证式分析后，发现它不是普通的“多因子真实数据”，而是一个高度结构化的合成数据：

1. **20 个指标近似可交换**：均值都在 4.92–4.96，标准差都在 2.05–2.09，两两相关全部为弱负值（均值 −0.0104）。
2. **控制行总和 N 后，两两相关 ≈ −1/19 = −0.0526**，与“把 N 个计数单元多项分配到 20 个指标”的机制一致。
3. **行总和 N 本身是多峰分布**（GMM 的 BIC 支持 4 个成分，均值约 82.5 / 94.7 / 103.5 / 110.6）。
4. **目标几乎只由 N 决定**：
   \[
   \hat y = 0.00564676\,N - 0.05335,\quad R^2=0.845,\quad \text{噪声 std}=0.0201
   \]
   残差与 20 个指标全部无关（|r| < 1e−15）。
5. 因此：
   - 问题 1 不存在实质意义上的“强指标/弱指标”：严格等权被统计拒绝（$p=1.2\times10^{-10}$），但 $\Delta R^2=1.3\times10^{-5}$、系数偏差 <1%，实践上等价；
   - 问题 2 的高/中/低风险组只是所有指标同步平移，不存在“组合型”画像；
   - 问题 3 线性/等权模型最优，复杂模型无增益；
   - 问题 4 的预测信号是多峰的，观测值因噪声叠加才接近正态。

## 2. 主要结果

| 项目 | 结果 |
|---|---|
| OLS / 等权指数 验证集 RMSE | 0.02011 |
| OLS 验证集 R² | 0.8445 |
| 等权约束 vs 无约束 OLS | ΔR²=1.3e-5，F=4.58（p=1.2e-10），系数偏差 ≤1% |
| Shapley R² 极差 / 均值 | 5.9% |
| 随机森林 / HistGB 验证集 RMSE | 0.03533 / 0.02116 |
| 最优 5 指标 RMSE（穷举 15504 组合） | 0.04604 |
| 最差 5 指标 RMSE | 0.04651 |
| 20 指标相比 5 指标的 RMSE 降幅 | 约 56% |
| TOPSIS 与目标 Spearman | 0.8805 |
| KMeans 与风险三分位 ARI | 0.0001 |
| 预测概率均值 / 标准差 | 0.5045 / 0.0469 |
| 预测分布峰度 / KS D | 0.841 / 0.0714 |

## 3. 目录结构

```
solution/
  code/
    common.py        # 路径、读取、指标、绘图风格
    eda.py           # 数据总览与 EDA
    structure.py     # 数据结构与生成机制（核心证据）
    problem1.py      # 指标关联分析
    problem1_inference.py  # 等权约束检验、分半复现、Shapley R² 分解
    problem2.py      # 风险分级、赋权、TOPSIS、灵敏度
    problem3.py      # 预测模型、5 指标穷举、学习曲线
    problem4.py      # test 预测、submit.csv、正态性分析
    make_tables.py   # 生成 paper/tables/*.tex
    run_all.py       # 一键复现
    requirements.txt
  data/              # train.csv / test.csv / submit.csv（原始附件）
  results/
    *.json           # 各问结果
    tables/*.csv     # 明细表
    logs/run_all.log # 运行日志
    submit.csv       # 填写后的提交文件
  figures/           # 论文图（PNG，200 dpi）
  paper/             # 官方 Word 模板版 + LaTeX 工作稿 + 表格
  submit.csv         # 与 results/submit.csv 相同的提交文件
```

## 4. 复现

```bash
cd solution/code
/home/shaun/miniforge3/bin/python run_all.py
```

约 1–2 分钟（不含 LaTeX 编译）。依赖见 `requirements.txt`。

单独运行某一问：

```bash
/home/shaun/miniforge3/bin/python problem3.py
```

## 5. 提交文件

`submit.csv` 与 `results/submit.csv` 已按原附件格式填写：

- GBK 编码；
- CRLF 换行；
- 保留末尾两个空列；
- 745,305 行数据，id 与 `test.csv` 完全一致；
- 概率列保留 6 位小数。

## 6. 论文（官方 Word 模板版，提交用）

正式提交文件按官方 Word 模板生成，第一页包含官方要求的

```
选题 | 2024年第十四届APMCM亚太地区大学生数学建模竞赛（中文赛项） | 参赛编号
  B  |                    （合并单元格）                            | apmcm241*****
```

的表格，正文宋体小四、单倍行距、A4、页边距 2.5 cm，共 17 页（含参考文献与附录）。

- `paper/Bapmcm241*****.docx`：可直接用 Word 打开、修改；
- `paper/Bapmcm241*****.pdf`：由该 docx 导出，可直接提交；
- `paper/build_official.py`：从 `main.tex` 自动生成上述 docx 与 pdf。

**提交前必须修改**：

1. 把封面表格中的 `apmcm241*****` 换成你们队的 13 位参赛编号；
2. 把文件名 `Bapmcm241*****` 改成 `B` + 实际队号，例如 `Bapmcm24123456.pdf`；
3. 在 Word 中确认封面表格、页码、图表编号后，重新导出 PDF。

重新生成：

```bash
cd paper
/home/shaun/miniforge3/bin/python build_official.py
```

`paper/main.tex` / `paper/main.pdf` 是 LaTeX 工作稿（21 页，图表更全），不是官方模板格式；正式提交以 `Bapmcm241*****.docx/pdf` 为准。

## 7. 说明与局限

- 本题数据高度疑似合成/等权构造。论文没有迎合“找关键指标”的常规叙事，而是用证据说明指标等权、只存在总量效应；这是本文与多数队伍最大的区别。
- 5 指标模型不是“越少越好”：穷举证明任选 5 个指标的精度几乎相同，且都显著低于 20 指标。论文给出了精度损失与可解释的折中建议。
- 预测分布的“非正态”结论针对的是**预测信号**；观测目标因噪声叠加在实用意义上接近正态。两者都报告，避免误读。
