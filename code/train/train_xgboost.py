"""
训练脚本 — XGBoost (Boosting 集成)

预处理: 无, department_id 通过 enable_categorical=True 处理

超参数微调: 在 Val 上对比 (max_depth, learning_rate) 组合
            使用 early_stopping_rounds=30 自动确定树的数量

最终模型:  最佳参数, Train+Val 全量重训 (以 Val 为早停集),
            Test 评估时使用 best_iteration

产出:
  - output/models/xgboost.json
  - output/predictions/xgboost_pred.csv
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import xgboost as xgb

from sklearn.metrics import f1_score

from code.train.config import XGB_PARAMS, XGB_CANDIDATES
from code.train.data_loader import load_and_split, extract_xy
from code.train.evaluation import evaluate_model
from code.train.model_io import save_model, save_predictions


def main():
    print("\n" + "=" * 60)
    print("  XGBoost 训练")
    print(f"  XGBoost 版本: {xgb.__version__}")
    print("=" * 60)

    # 1. 加载并划分数据
    train_data, val_data, test_data = load_and_split()

    X_train, y_train, _ = extract_xy(train_data)
    X_val, y_val, _ = extract_xy(val_data)
    X_test, y_test, ids_test = extract_xy(test_data)

    # 2. 超参数微调 — (max_depth, learning_rate) 组合, early_stopping
    print("\n3. 超参数微调 (Val 上对比 max_depth × learning_rate)...")
    best_params = None
    best_f1 = -1
    best_boost_rounds = None

    for combo in XGB_CANDIDATES:
        params = XGB_PARAMS.copy()
        params.update(combo)
        # 去掉 early_stopping_rounds, 它作为 fit 参数传入
        early_stop = params.pop("early_stopping_rounds", 30)
        params.pop("enable_categorical", None)

        model = xgb.XGBClassifier(
            **params,
            enable_categorical=True,
            early_stopping_rounds=early_stop,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_val_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_val_pred)
        n_trees = model.best_iteration
        print(
            f"   max_depth={combo['max_depth']}, "
            f"lr={combo['learning_rate']:.2f}  →  "
            f"Val F1={f1:.4f}  (best_iter={n_trees})"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_params = combo
            best_boost_rounds = n_trees

    print(
        f"\n   最佳: max_depth={best_params['max_depth']}, "
        f"lr={best_params['learning_rate']:.2f} "
        f"(Val F1 = {best_f1:.4f}, best_iter = {best_boost_rounds})"
    )

    # 3. 最终模型 — Train+Val 重训（用最佳迭代数，无早停，避免 Test 泄露）
    print("\n4. 用最佳参数在 Train+Val 上训练最终模型...")
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    final_params = XGB_PARAMS.copy()
    final_params.update(best_params)
    final_params["n_estimators"] = best_boost_rounds
    final_params.pop("early_stopping_rounds", None)
    final_params.pop("enable_categorical", None)

    final_model = xgb.XGBClassifier(
        **final_params,
        enable_categorical=True,
    )
    final_model.fit(
        X_trainval,
        y_trainval,
        verbose=False,
    )
    print(f"   树数量: {best_boost_rounds} (来自 Val 早停)")

    # 4. 测试集评估
    print("\n5. 测试集评估...")
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_test_pred, y_test_proba, "XGBoost")

    # 5. 保存
    print("\n6. 保存模型与预测...")
    save_model(final_model, "xgboost")
    save_predictions(ids_test, y_test, y_test_pred, y_test_proba, "xgboost")

    print("\n✓ XGBoost 训练完成!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
