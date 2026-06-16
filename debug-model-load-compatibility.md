# [OPEN] model-load-compatibility

## 问题概述
- 症状：`Streamlit` 可启动，但“单条样本预测”中仅 `XGBoost` 可用，`Logistic Regression`、`Decision Tree`、`Random Forest`、`LightGBM` 加载失败。
- 现象：运行时出现 `No module named 'numpy._core'`，并伴随 `scikit-learn` 版本不一致警告。
- 目标：定位模型序列化兼容问题，恢复当前环境下的模型加载与单样本预测能力。

## 当前状态
- 状态：证据收集中
- 约束：在完成证据确认前，不修改业务逻辑

## 假设
1. 模型或预处理器是在不同版本的 `numpy` / `scikit-learn` 环境下序列化，当前环境反序列化失败。
2. 某些 `.pkl` 文件内部引用了旧版 `numpy` 模块路径（如 `numpy._core`），当前环境中不存在对应模块别名。
3. 训练脚本与当前运行环境依赖版本不一致，导致重新加载历史产物时出现 ABI 或模块路径问题。
4. `LightGBM` 与 `scikit-learn` 模型失败原因本质相同，但 `XGBoost` 因使用独立 `json` 模型格式而不受影响。
5. 当前问题只影响“加载历史模型”，重新在本地环境训练并导出后可恢复兼容性。

## 证据记录
- 当前环境版本：
  - Python 3.8.6
  - numpy 1.24.4
  - scikit-learn 1.3.2
  - pandas 2.0.3
  - joblib 1.4.2
- 运行时加载结果：
  - `output/preprocessor/lr_preprocessor.pkl` -> `ModuleNotFoundError: No module named 'numpy._core'`
  - `output/models/logistic_regression.pkl` -> `ModuleNotFoundError: No module named 'numpy._core'`
  - `output/models/decision_tree.pkl` -> `ModuleNotFoundError: No module named 'numpy._core'`
  - `output/models/random_forest.pkl` -> `ModuleNotFoundError: No module named 'numpy._core'`
  - `output/models/lightgbm.pkl` -> `ModuleNotFoundError: No module named 'numpy._core'`
  - `output/models/xgboost.json` -> 可正常加载并预测
- 附加观察：
  - 反序列化时出现 `InconsistentVersionWarning`，提示部分模型/预处理器来自 `scikit-learn 1.8.0`
  - 训练脚本以 `output/feature_table.csv` 为数据源，即使原始 `data/` 不在，也可以在当前环境重训并重导出兼容模型
- 后续修复验证：
  - 当前环境重新训练并导出后，`lr_preprocessor.pkl` 已可正常加载
  - 当前环境重新训练并导出后，`logistic_regression.pkl` 已可正常加载并完成预测
  - 当前环境重新训练并导出后，`decision_tree.pkl` 已可正常加载并完成预测
  - 当前环境重新训练并导出后，`lightgbm.pkl` 已可正常加载并完成预测
  - `random_forest.pkl` 与 `random_forest_pred.csv` 已在当前环境重新生成
  - 最终对比结果已恢复为：`Logistic Regression` / `Decision Tree` / `Random Forest` / `LightGBM` / `XGBoost` 全部可用

## 结论
- 假设 1、2、4、5 当前被证据支持。
- 最小修复路径：在当前环境重新训练并覆盖导出 `Logistic Regression`、`Decision Tree`、`Random Forest`、`LightGBM` 相关产物，使其与本地 `numpy` / `scikit-learn` 版本重新对齐。
- 当前状态：修复方案已验证有效，单条样本预测链路已恢复为 5/5 模型可用，等待用户在界面中确认问题已解决。
