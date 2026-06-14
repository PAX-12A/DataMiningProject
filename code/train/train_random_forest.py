"""
训练脚本 - Random Forest (Bagging 集成)

预处理: 无

超参数微调: 在 Val 上对比 (n_estimators, max_depth) 组合
最终模型:  最佳参数, Train+Val 全量重训, Test 评估

注意: 100 棵树 x 80 万行 内存较大，joblib compress=3 压缩保存。
      max_depth=12 限制以控制模型体积。

产出:
  - output/models/random_forest.pkl
  - output/predictions/random_forest_pred.csv
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from code.train.config import RF_PARAMS, RF_CANDIDATES
from code.train.data_loader import load_and_split, extract_xy
from code.train.evaluation import evaluate_model
from code.train.model_io import save_model, save_predictions


def main():
    print("\n" + "=" * 60)
    print("  Random Forest 训练")
    print("=" * 60)

    # 1. 加载并划分数据
    train_data, val_data, test_data = load_and_split()

    X_train, y_train, _ = extract_xy(train_data)
    X_val, y_val, _ = extract_xy(val_data)
    X_test, y_test, ids_test = extract_xy(test_data)

    # 2. 超参数微调 - (n_estimators, max_depth) 组合
    print("\n3. 超参数微调 (Val 上对比 n_estimators x max_depth)...")
    best_params = None
    best_f1 = -1

    for combo in RF_CANDIDATES:
        params = RF_PARAMS.copy()
        params.update(combo)
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        y_val_pred = rf.predict(X_val)
        f1 = f1_score(y_val, y_val_pred)
        print(
            f"   n_estimators={combo['n_estimators']:3d}, "
            f"max_depth={combo['max_depth']:2d}  ->  Val F1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_params = combo

    print(
        f"\n   最佳: n_estimators={best_params['n_estimators']}, "
        f"max_depth={best_params['max_depth']} (Val F1 = {best_f1:.4f})"
    )

    # 3. 最终模型 - Train+Val 重训
    print("\n4. 用最佳参数在 Train+Val 上训练最终模型...")
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    final_params = RF_PARAMS.copy()
    final_params.update(best_params)
    print(f"   开始训练 (n_estimators={final_params['n_estimators']}, "
          f"n_jobs=-1, 全 CPU 并行)...")
    final_model = RandomForestClassifier(**final_params)
    final_model.fit(X_trainval, y_trainval)

    # 4. 测试集评估
    print("\n5. 测试集评估...")
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_test_pred, y_test_proba, "Random Forest")

    # 5. 保存
    print("\n6. 保存模型与预测...")
    save_model(final_model, "random_forest")
    save_predictions(ids_test, y_test, y_test_pred, y_test_proba, "random_forest")

    print("\nOK Random Forest 训练完成!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
