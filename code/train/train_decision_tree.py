"""
训练脚本 - Decision Tree (单棵树基线)

预处理: 无（原始特征直接训练）

超参数微调: 在 Val 上对比 max_depth ∈ {8, 12, 15, None}
最终模型:  最佳 max_depth, Train+Val 全量重训, Test 评估

产出:
  - output/models/decision_tree.pkl
  - output/predictions/decision_tree_pred.csv
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

from code.train.config import DT_PARAMS, DT_MAX_DEPTH_CANDIDATES
from code.train.data_loader import load_and_split, extract_xy
from code.train.evaluation import evaluate_model
from code.train.model_io import save_model, save_predictions


def main():
    print("\n" + "=" * 60)
    print("  Decision Tree 训练")
    print("=" * 60)

    # 1. 加载并划分数据
    train_data, val_data, test_data = load_and_split()

    X_train, y_train, _ = extract_xy(train_data)
    X_val, y_val, _ = extract_xy(val_data)
    X_test, y_test, ids_test = extract_xy(test_data)

    # 2. 超参数微调 - max_depth
    print("\n3. 超参数微调 (Val 上对比 max_depth)...")
    best_depth = None
    best_f1 = -1

    for depth in DT_MAX_DEPTH_CANDIDATES:
        params = DT_PARAMS.copy()
        params["max_depth"] = depth
        dt = DecisionTreeClassifier(**params)
        dt.fit(X_train, y_train)

        y_val_pred = dt.predict(X_val)
        f1 = f1_score(y_val, y_val_pred)
        depth_label = "None" if depth is None else str(depth)
        print(f"   max_depth={depth_label:4s}  ->  Val F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_depth = depth

    depth_label = "None" if best_depth is None else str(best_depth)
    print(f"\n   最佳 max_depth = {depth_label} (Val F1 = {best_f1:.4f})")

    # 3. 最终模型 - Train+Val 重训
    print("\n4. 用最佳参数在 Train+Val 上训练最终模型...")
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    final_params = DT_PARAMS.copy()
    final_params["max_depth"] = best_depth
    final_model = DecisionTreeClassifier(**final_params)
    final_model.fit(X_trainval, y_trainval)

    # 4. 测试集评估
    print("\n5. 测试集评估...")
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_test_pred, y_test_proba, "Decision Tree")

    # 5. 保存
    print("\n6. 保存模型与预测...")
    save_model(final_model, "decision_tree")
    save_predictions(ids_test, y_test, y_test_pred, y_test_proba, "decision_tree")

    print("\nOK Decision Tree 训练完成!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
