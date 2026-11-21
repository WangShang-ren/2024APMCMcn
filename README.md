# APMCM 2024 中文赛项 B 题 · 洪水灾害的数据分析与预测

2024 年第十四届 APMCM 亚太地区大学生数学建模竞赛（中文赛项）B 题解答工程。

## 目录结构

```
SolOnB/
├── B题 洪水灾害的数据分析与预测/   # 赛题原文与附件原始数据
└── solution/                        # 代码、结果、图表、论文、提交文件
    ├── code/       # 全部脚本，run_all.py 一键复现
    ├── results/    # JSON、明细表、日志、最终提交文件
    ├── figures/    # 论文图
    └── paper/      # LaTeX 工作稿与官方 Word/PDF
```

原始附件（`train.csv` / `test.csv` / `submit.csv`）直接读取 `B题.../附件 .../` 下的文件，不再在 `solution/` 内重复存放。

## 复现

```bash
cd solution/code
python run_all.py
```

分析结论、结果表与更详细的说明见 [solution/README.md](solution/README.md)。

模型整体框架（母模型、四问子模型、检验体系与边界）见 [solution/模型整体设计.md](solution/模型整体设计.md)。
