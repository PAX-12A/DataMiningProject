"""
模型与预测结果保存。

- scikit-learn 模型: joblib
- XGBoost: 自有 .json 格式
- LightGBM: 自有 .txt 格式
- 预测结果: CSV (user_id, product_id, y_true, y_pred, y_proba)
"""
import os
import joblib
import pandas as pd

from .config import MODEL_DIR, PRED_DIR


def save_model(model, model_name):
    """
    按框架选择最佳格式保存模型。

    Parameters
    ----------
    model : 训练好的模型对象
    model_name : str (如 "logistic_regression", "xgboost")
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # XGBoost
    if model_name == "xgboost":
        path = os.path.join(MODEL_DIR, "xgboost.json")
        model.save_model(path)
        print(f"  模型已保存 (XGBoost JSON): {path}")

    # LightGBM（用 joblib 以避免中文路径下原生 save 报错）
    elif model_name == "lightgbm":
        path = os.path.join(MODEL_DIR, "lightgbm.pkl")
        joblib.dump(model, path)
        print(f"  模型已保存 (joblib): {path}")

    # scikit-learn 系（LR, DT, RF）
    else:
        path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        if model_name == "random_forest":
            joblib.dump(model, path, compress=3)
        else:
            joblib.dump(model, path)
        print(f"  模型已保存 (joblib): {path}")


def save_predictions(ids, y_true, y_pred, y_proba, model_name):
    """
    保存预测结果 CSV。

    Parameters
    ----------
    ids : DataFrame (user_id, product_id)
    y_true : array-like
    y_pred : array-like
    y_proba : array-like
    model_name : str
    """
    os.makedirs(PRED_DIR, exist_ok=True)

    pred_df = ids.copy()
    pred_df["y_true"] = y_true.values
    pred_df["y_pred"] = y_pred
    pred_df["y_proba"] = y_proba

    path = os.path.join(PRED_DIR, f"{model_name}_pred.csv")
    pred_df.to_csv(path, index=False)
    print(f"  预测结果已保存: {path}  ({len(pred_df):,} 行)")
