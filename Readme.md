# Instacart 用户复购预测项目

本项目基于 Instacart Market Basket Analysis 数据集，完成用户-商品复购预测流程：

1. 生成带特征的训练样本表
2. 训练多种机器学习模型
3. 汇总评估指标、显著性检验、可视化图表和错误分析报告

## 1. 数据准备

请先下载 [Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) 数据集，并将以下 6 个 CSV 文件放到 `data/` 目录：

```text
data/
├── aisles.csv
├── departments.csv
├── orders.csv
├── order_products__prior.csv
├── order_products__train.csv
└── products.csv
```

## 2. 环境检查

建议确认以下依赖可正常导入：

```powershell
python -c "import pandas; print('pandas', pandas.__version__)"
python -c "import sklearn; print('sklearn', sklearn.__version__)"
python -c "import joblib; print('joblib', joblib.__version__)"
python -c "import matplotlib; print('matplotlib', matplotlib.__version__)"
python -c "import scipy; print('scipy', scipy.__version__)"
python -c "import xgboost; print('xgboost', xgboost.__version__)"
python -c "import lightgbm; print('lightgbm', lightgbm.__version__)"
```

如缺少依赖，可安装：

```powershell
pip install pandas scikit-learn joblib matplotlib scipy xgboost lightgbm
```

## 3. 流程总览

| 阶段 | 前置文件 | 执行代码 | 生成文件 |
|------|----------|----------|----------|
| 生成特征表 | `data/` 下 6 个原始 CSV | `code/feature_engineering.py` | `output/feature_table.csv` |
| 训练模型 | `output/feature_table.csv`，首次训练会自动生成或复用 `data/splits/` | `code/train/train_*.py` | `data/splits/*.csv`，`output/models/*`，`output/predictions/*_pred.csv`，`output/metrics_summary.csv` |
| 评估与分析 | `output/metrics_summary.csv`，`output/predictions/*_pred.csv`，错误分析还会读取 `output/feature_table.csv` | `code/evaluate_analysis.py` | `output/analysis/*`，`figure/evaluation/*`，`doc/评估与分析报告.md` |

## 4. 生成特征表

### 前置文件

需要先准备好 `data/` 目录下的 6 个原始数据文件：

```text
data/
├── aisles.csv
├── departments.csv
├── orders.csv
├── order_products__prior.csv
├── order_products__train.csv
└── products.csv
```

### 执行代码

在项目根目录运行：

```powershell
python code\feature_engineering.py
```

执行文件位置：

```text
code/feature_engineering.py
```

该脚本会读取 `data/` 下的原始数据，构造用户特征、商品特征、用户-商品交互特征和部门特征，并生成：

### 生成文件

```text
output/feature_table.csv
```

当前训练脚本默认从 `output/feature_table.csv` 读取特征表。

## 5. 训练模型

### 前置文件

训练模型依赖特征工程阶段生成的特征表：

```text
output/feature_table.csv
```

首次训练时，代码会自动按 `user_id` 做 60/20/20 划分并生成：

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

如果这些划分文件已经存在，后续训练会直接复用，保证不同模型在同一测试集上比较。

### 执行代码

训练脚本位于 `code/train/`。首次训练会自动按 `user_id` 做 60/20/20 划分，并保存到：

```text
code/train/
├── train_logistic_regression.py
├── train_decision_tree.py
├── train_random_forest.py
├── train_lightgbm.py
└── train_xgboost.py
```

### 单独训练某个模型

```powershell
python code\train\train_logistic_regression.py
python code\train\train_decision_tree.py
python code\train\train_random_forest.py
python code\train\train_lightgbm.py
python code\train\train_xgboost.py
```

建议先跑较快的模型验证流程：

```powershell
python code\train\train_logistic_regression.py
python code\train\train_decision_tree.py
python code\train\train_lightgbm.py
```

然后再运行耗时更长的模型：

```powershell
python code\train\train_xgboost.py
python code\train\train_random_forest.py
```

### 生成文件

模型文件保存到：

```text
output/models/
```

预测结果保存到：

```text
output/predictions/
```

每个预测文件包含：

```text
user_id, product_id, y_true, y_pred, y_proba
```

汇总指标保存到：

```text
output/metrics_summary.csv
```

训练阶段的完整生成文件包括：

```text
data/splits/
├── train.csv
├── val.csv
└── test.csv

output/models/
├── logistic_regression.pkl
├── decision_tree.pkl
├── random_forest.pkl
├── lightgbm.pkl
└── xgboost.json

output/predictions/
├── logistic_regression_pred.csv
├── decision_tree_pred.csv
├── random_forest_pred.csv
├── lightgbm_pred.csv
└── xgboost_pred.csv

output/metrics_summary.csv
```

## 6. 评估与分析

### 前置文件

评估与分析依赖训练阶段生成的指标汇总和预测结果：

```text
output/metrics_summary.csv
output/predictions/*_pred.csv
```

错误分析还会回连特征表，因此需要：

```text
output/feature_table.csv
```

如果某个模型缺少对应的 `*_pred.csv`，它不会参与 ROC/PR 曲线、混淆矩阵、McNemar 检验、Top-K 和错误切片分析。

### 执行代码

训练完成并生成预测文件后，运行：

```powershell
python code\evaluate_analysis.py
```

执行文件位置：

```text
code/evaluate_analysis.py
```

该脚本会基于 `output/metrics_summary.csv` 和 `output/predictions/*_pred.csv` 生成：

### 生成文件

评估分析表：

```text
output/analysis/
├── evaluation_metrics_recomputed.csv
├── confusion_matrices.csv
├── bootstrap_ci.csv
├── mcnemar_pairwise_tests.csv
├── topk_metrics.csv
├── error_type_feature_summary.csv
└── error_slice_analysis.csv
```

可视化图表：

```text
figure/evaluation/
├── model_metric_comparison.png
├── roc_pr_curves.png
├── model_error_rates.png
├── topk_precision.png
├── best_model_error_types.png
└── recency_error_slice.png
```

报告章节：

```text
doc/评估与分析报告.md
```

报告内容包括：

- 评估指标设计
- 模型综合对比
- McNemar 统计显著性检验
- Bootstrap 置信区间
- Top-K 推荐指标
- 错误类型和错误切片分析
- 结论与改进建议

## 7. 推荐完整运行顺序

从零开始时，按以下顺序运行：

```powershell
python code\feature_engineering.py

python code\train\train_logistic_regression.py
python code\train\train_decision_tree.py
python code\train\train_lightgbm.py
python code\train\train_xgboost.py
python code\train\train_random_forest.py

python code\evaluate_analysis.py
```

## 8. 主要目录说明

```text
code/
├── feature_engineering.py        # 特征工程
├── evaluate_analysis.py          # 评估、显著性检验、可视化、错误分析
└── train/                        # 模型训练脚本和共享工具

data/                             # 原始数据和训练划分
doc/                              # 项目文档与报告
figure/                           # EDA 和评估图表
output/
├── feature_table.csv             # 特征表
├── metrics_summary.csv           # 模型指标汇总
├── models/                       # 训练好的模型
├── predictions/                  # 测试集预测结果
├── preprocessor/                 # 预处理器
└── analysis/                     # 评估分析结果
```

## 9. 注意事项

- `output/feature_table.csv` 文件较大，生成时间和读取时间会比较长。
- `Random Forest` 和 `XGBoost` 训练耗时较长，可最后运行。
- 评估脚本只会对已有的 `output/predictions/*_pred.csv` 做逐样本分析；如果某个模型缺少预测文件，它不会参与 McNemar 检验、Top-K 和错误分析。
- 若 Windows 控制台出现编码问题，可尝试：

```powershell
set PYTHONIOENCODING=utf-8
```

## 10. 交互界面

项目已补充基于 `Streamlit` 的交互界面，可用于答辩演示和工程交付展示。

### 安装界面依赖

```powershell
pip install streamlit
```

### 启动方式

在项目根目录运行：

```powershell
streamlit run app.py
```

### 界面功能

- `项目总览`：查看原始数据、模型、预测结果、分析表和当前最佳模型状态
- `特征工程`：检查原始数据并运行 `code/feature_engineering.py`
- `模型训练`：按模型单独训练，或按推荐顺序批量训练全部模型
- `评估分析`：运行 `code/evaluate_analysis.py` 并查看指标表、Top-K、McNemar 检验和报告
- `单条样本预测`：从特征表中选择单条样本，横向对比各模型预测标签、正类概率和命中情况
- `文档与图表`：浏览 README、训练报告、评估报告及图表 PNG

### 说明

- 界面内部通过子进程调用现有脚本，不改变原有训练与评估逻辑。
- 如需完整展示流程，建议按以下顺序操作：

```text
1. 特征工程
2. 模型训练
3. 评估分析
4. 文档与图表展示
```
