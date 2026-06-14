"""
训练脚本 - LightGBM (Boosting 集成)

预处理: 无, department_id 通过 categorical_feature 参数原生处理

超参数微调: 在 Val 上对比 (num_leaves, max_depth) 组合
            使用 early_stopping_rounds=30, callbacks 记录 best_iteration

最终模型:  最佳参数, Train+Val 全量重训 (以 Val 为早停集),
            Test 评估

产出:
  - output/models/lightgbm.txt
  - output/predictions/lightgbm_pred.csv
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import lightgbm as lgb

from sklearn.metrics import f1_score

from code.train.config import LGB_PARAMS, LGB_CANDIDATES
from code.train.data_loader import load_and_split, extract_xy
from code.train.evaluation import evaluate_model
from code.train.model_io import save_model, save_predictions


def main():
    print("\n" + "=" * 60)
    print("  LightGBM 训练")
    print(f"  LightGBM 版本: {lgb.__version__}")
    print("=" * 60)

    # 1. 加载并划分数据
    train_data, val_data, test_data = load_and_split()

    X_train, y_train, _ = extract_xy(train_data)
    X_val, y_val, _ = extract_xy(val_data)
    X_test, y_test, ids_test = extract_xy(test_data)

    # 2. 找到 department_id 的列索引（LightGBM 4.6+ 要求数字或 "name:" 前缀）
    cat_col_idx = X_train.columns.get_loc("department_id")

    # 3. 超参数微调 - (num_leaves, max_depth) 组合
    print("\n3. 超参数微调 (Val 上对比 num_leaves x max_depth)...")
    best_params = None
    best_f1 = -1
    best_boost_rounds = None

    for combo in LGB_CANDIDATES:
        params = LGB_PARAMS.copy()
        params.update(combo)
        early_stop = params.pop("early_stopping_rounds", 30)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            categorical_feature=[cat_col_idx],
            callbacks=[
                lgb.early_stopping(early_stop, min_delta=0.001),
                lgb.log_evaluation(0),
            ],
        )

        # 用概率找最优阈值，而非默认 0.5
        y_val_proba = model.predict_proba(X_val)[:, 1]
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        best_thresh_f1 = -1
        best_thresh = 0.5
        for t in thresholds:
            f1_t = f1_score(y_val, y_val_proba >= t)
            if f1_t > best_thresh_f1:
                best_thresh_f1 = f1_t
                best_thresh = t

        n_trees = model.best_iteration_
        print(
            f"   num_leaves={combo['num_leaves']:3d}, "
            f"max_depth={combo['max_depth']}  ->  "
            f"Val F1={best_thresh_f1:.4f}  (best_iter={n_trees}, thresh={best_thresh})"
        )

        if best_thresh_f1 > best_f1:
            best_f1 = best_thresh_f1
            best_params = combo
            best_boost_rounds = n_trees

    print(
        f"\n   最佳: num_leaves={best_params['num_leaves']}, "
        f"max_depth={best_params['max_depth']} "
        f"(Val F1 = {best_f1:.4f}, best_iter = {best_boost_rounds})"
    )

    # 3. 最终模型 - Train+Val 重训（用最佳迭代数，无早停，避免 Test 泄露）
    print("\n4. 用最佳参数在 Train+Val 上训练最终模型...")
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    final_params = LGB_PARAMS.copy()
    final_params.update(best_params)
    final_params["n_estimators"] = best_boost_rounds
    final_params.pop("early_stopping_rounds", None)

    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(
        X_trainval,
        y_trainval,
        categorical_feature=[cat_col_idx],
        callbacks=[lgb.log_evaluation(0)],
    )
    print(f"   树数量: {best_boost_rounds} (来自 Val 早停)")

    # 4. 测试集评估（用 Val 上选出的最佳阈值）
    print("\n5. 测试集评估...")
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    # 在 Val 上找最佳阈值
    y_val_proba_final = final_model.predict_proba(X_val)[:, 1]
    best_test_thresh = 0.5
    best_test_f1 = -1
    for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        f1_t = f1_score(y_val, y_val_proba_final >= t)
        if f1_t > best_test_f1:
            best_test_f1 = f1_t
            best_test_thresh = t
    y_test_pred = (y_test_proba >= best_test_thresh).astype(int)
    print(f"   使用阈值: {best_test_thresh}")

    evaluate_model(y_test, y_test_pred, y_test_proba, "LightGBM")

    # 5. 保存
    print("\n6. 保存模型与预测...")
    save_model(final_model, "lightgbm")
    save_predictions(ids_test, y_test, y_test_pred, y_test_proba, "lightgbm")

    print("\nOK LightGBM 训练完成!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
