"""
统一评估函数 - 计算所有指标，打印格式化表格，追加到汇总 CSV。
"""
import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)

from .config import METRICS_PATH


def evaluate_model(y_true, y_pred, y_proba, model_name):
    """
    计算全部 7 个指标，打印结果，并追加到 metrics_summary.csv。

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    y_proba : array-like (正类概率)
    model_name : str

    Returns
    -------
    metrics : dict
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
    }

    _print_metrics(metrics)
    _append_to_summary(metrics)

    return metrics


def _print_metrics(metrics):
    """格式化打印指标表。"""
    print(f"\n{'=' * 50}")
    print(f"  {metrics['model']} - 测试集指标")
    print(f"{'=' * 50}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}    * 主要")
    print(f"  F1-score:   {metrics['f1']:.4f}        * 主要")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}   * 主要")
    print(f"  PR-AUC:     {metrics['pr_auc']:.4f}    * 主要")
    print(f"  LogLoss:    {metrics['log_loss']:.4f}")
    print(f"{'=' * 50}\n")


def _append_to_summary(metrics):
    """追加一行到汇总 CSV。"""
    row = pd.DataFrame([metrics])

    if os.path.exists(METRICS_PATH):
        existing = pd.read_csv(METRICS_PATH)
        # 如已有同名模型行则覆盖
        existing = existing[existing["model"] != metrics["model"]]
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row

    combined.to_csv(METRICS_PATH, index=False)
    print(f"  指标已追加到: {METRICS_PATH}")
