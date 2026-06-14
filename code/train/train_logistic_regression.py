"""
训练脚本 - Logistic Regression (线性基线模型)

预处理:
  - StandardScaler (17 个数值特征)
  - OneHotEncoder (department_id, 21 类, sparse)

超参数微调: 在 Val 上对比 C ∈ {0.1, 1.0, 10}
最终模型:  最佳 C, Train+Val 全量重训, Test 评估

产出:
  - output/models/logistic_regression.pkl
  - output/preprocessor/lr_preprocessor.pkl
  - output/predictions/logistic_regression_pred.csv
"""
import sys
import os

# 确保项目根在 sys.path 中，以便 code.train 可导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from code.train.config import LR_PARAMS, LR_C_CANDIDATES
from code.train.data_loader import load_and_split, extract_xy
from code.train.preprocessing import build_lr_preprocessor, fit_preprocessor
from code.train.evaluation import evaluate_model
from code.train.model_io import save_model, save_predictions


def main():
    print("\n" + "=" * 60)
    print("  Logistic Regression 训练")
    print("=" * 60)

    # 1. 加载并划分数据
    train_data, val_data, test_data = load_and_split()

    X_train, y_train, _ = extract_xy(train_data)
    X_val, y_val, _ = extract_xy(val_data)
    X_test, y_test, ids_test = extract_xy(test_data)

    # 2. 构建并拟合预处理器
    print("\n3. 构建预处理器 (StandardScaler + OneHot)...")
    preprocessor = build_lr_preprocessor(X_train)
    preprocessor = fit_preprocessor(preprocessor, X_train)

    X_train_pp = preprocessor.transform(X_train)
    X_val_pp = preprocessor.transform(X_val)
    X_test_pp = preprocessor.transform(X_test)

    # 3. 超参数微调 - 在 Val 上对比 C 值
    print("\n4. 超参数微调 (Val 上对比 C)...")
    best_c = None
    best_f1 = -1
    best_model_candidate = None

    for c in LR_C_CANDIDATES:
        params = LR_PARAMS.copy()
        params["C"] = c
        lr = LogisticRegression(**params)
        lr.fit(X_train_pp, y_train)

        y_val_pred = lr.predict(X_val_pp)
        y_val_proba = lr.predict_proba(X_val_pp)[:, 1]

        from sklearn.metrics import f1_score

        f1 = f1_score(y_val, y_val_pred)
        print(f"   C={c:.1f}  ->  Val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_c = c
            best_model_candidate = lr

    print(f"\n   最佳 C = {best_c} (Val F1 = {best_f1:.4f})")

    # 4. 最终模型 - 用最佳 C 在 Train+Val 上重训
    print("\n5. 用最佳参数在 Train+Val 上训练最终模型...")
    X_trainval_pp = pd.concat([X_train_pp, X_val_pp], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    final_params = LR_PARAMS.copy()
    final_params["C"] = best_c
    final_model = LogisticRegression(**final_params)
    final_model.fit(X_trainval_pp, y_trainval)

    # 5. 测试集评估
    print("\n6. 测试集评估...")
    y_test_pred = final_model.predict(X_test_pp)
    y_test_proba = final_model.predict_proba(X_test_pp)[:, 1]

    evaluate_model(y_test, y_test_pred, y_test_proba, "Logistic Regression")

    # 6. 保存
    print("\n7. 保存模型与预测...")
    save_model(final_model, "logistic_regression")
    save_predictions(ids_test, y_test, y_test_pred, y_test_proba, "logistic_regression")

    print("\nOK Logistic Regression 训练完成!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
