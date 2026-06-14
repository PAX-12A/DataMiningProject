import itertools
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "output" / "predictions"
METRICS_PATH = PROJECT_ROOT / "output" / "metrics_summary.csv"
FEATURE_PATH = PROJECT_ROOT / "output" / "feature_table.csv"
ANALYSIS_DIR = PROJECT_ROOT / "output" / "analysis"
FIGURE_DIR = PROJECT_ROOT / "figure" / "evaluation"

RANDOM_SEED = 42
BOOTSTRAP_ROUNDS = 50
BOOTSTRAP_SAMPLE_SIZE = 100_000
TOP_K_VALUES = [5, 10, 20]


def ensure_dirs():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def model_name_from_file(path: Path) -> str:
    name = path.stem.replace("_pred", "")
    return {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
    }.get(name, name)


def load_predictions():
    pred_files = sorted(PRED_DIR.glob("*_pred.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {PRED_DIR}")

    predictions = {}
    for path in pred_files:
        model = model_name_from_file(path)
        df = pd.read_csv(path)
        required_cols = {"user_id", "product_id", "y_true", "y_pred", "y_proba"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        predictions[model] = df
    return predictions


def compute_metric_table(predictions):
    rows = []
    for model, df in predictions.items():
        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        y_proba = df["y_proba"].to_numpy()
        rows.append(
            {
                "model": model,
                "n": len(df),
                "positive_rate": y_true.mean(),
                "accuracy": (y_true == y_pred).mean(),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, y_proba),
                "pr_auc": average_precision_score(y_true, y_proba),
                "log_loss": log_loss(y_true, y_proba),
            }
        )
    result = pd.DataFrame(rows).sort_values("f1", ascending=False)
    result.to_csv(ANALYSIS_DIR / "evaluation_metrics_recomputed.csv", index=False)
    return result


def compute_confusion_tables(predictions):
    rows = []
    for model, df in predictions.items():
        tn, fp, fn, tp = confusion_matrix(df["y_true"], df["y_pred"]).ravel()
        rows.append(
            {
                "model": model,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "false_positive_rate": fp / (fp + tn),
                "false_negative_rate": fn / (fn + tp),
            }
        )
    result = pd.DataFrame(rows).sort_values("model")
    result.to_csv(ANALYSIS_DIR / "confusion_matrices.csv", index=False)
    return result


def bootstrap_metric_ci(predictions):
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for model, df in predictions.items():
        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        y_proba = df["y_proba"].to_numpy()
        n = len(df)
        sample_size = min(BOOTSTRAP_SAMPLE_SIZE, n)

        values = {"f1": [], "roc_auc": [], "pr_auc": []}
        for _ in range(BOOTSTRAP_ROUNDS):
            idx = rng.integers(0, n, sample_size)
            sample_true = y_true[idx]
            if len(np.unique(sample_true)) < 2:
                continue
            values["f1"].append(f1_score(sample_true, y_pred[idx], zero_division=0))
            values["roc_auc"].append(roc_auc_score(sample_true, y_proba[idx]))
            values["pr_auc"].append(average_precision_score(sample_true, y_proba[idx]))

        for metric, metric_values in values.items():
            arr = np.asarray(metric_values)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": arr.mean(),
                    "ci95_low": np.percentile(arr, 2.5),
                    "ci95_high": np.percentile(arr, 97.5),
                    "bootstrap_rounds": len(arr),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(ANALYSIS_DIR / "bootstrap_ci.csv", index=False)
    return result


def mcnemar_pairwise_tests(predictions):
    rows = []
    for model_a, model_b in itertools.combinations(sorted(predictions), 2):
        a = predictions[model_a].sort_values(["user_id", "product_id"]).reset_index(drop=True)
        b = predictions[model_b].sort_values(["user_id", "product_id"]).reset_index(drop=True)
        if not a[["user_id", "product_id", "y_true"]].equals(b[["user_id", "product_id", "y_true"]]):
            raise ValueError(f"Prediction rows are not aligned for {model_a} and {model_b}")

        correct_a = a["y_pred"].eq(a["y_true"]).to_numpy()
        correct_b = b["y_pred"].eq(b["y_true"]).to_numpy()
        b01 = int((correct_a & ~correct_b).sum())
        b10 = int((~correct_a & correct_b).sum())
        discordant = b01 + b10

        if discordant == 0:
            p_value = 1.0
            statistic = 0.0
        else:
            p_value = binomtest(min(b01, b10), discordant, 0.5).pvalue
            statistic = (abs(b01 - b10) - 1) ** 2 / discordant

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "a_correct_b_wrong": b01,
                "a_wrong_b_correct": b10,
                "discordant_pairs": discordant,
                "mcnemar_chi2_cc": statistic,
                "p_value": p_value,
                "significant_at_0.05": p_value < 0.05,
                "better_by_accuracy": model_a if b01 > b10 else model_b if b10 > b01 else "tie",
            }
        )

    result = pd.DataFrame(rows).sort_values("p_value")
    result.to_csv(ANALYSIS_DIR / "mcnemar_pairwise_tests.csv", index=False)
    return result


def compute_topk_metrics(predictions):
    rows = []
    for model, df in predictions.items():
        ranked = df.sort_values(["user_id", "y_proba"], ascending=[True, False])
        for k in TOP_K_VALUES:
            topk = ranked.groupby("user_id", sort=False).head(k)
            per_user = topk.groupby("user_id")["y_true"].agg(["sum", "count"])
            positives = df.groupby("user_id")["y_true"].sum()
            recall_denominator = positives[positives > 0].sum()
            rows.append(
                {
                    "model": model,
                    "k": k,
                    "precision_at_k": per_user["sum"].sum() / per_user["count"].sum(),
                    "recall_at_k": per_user["sum"].sum() / recall_denominator,
                    "users_with_positive_in_topk": (per_user["sum"] > 0).mean(),
                    "covered_positive_items": int(per_user["sum"].sum()),
                }
            )
    result = pd.DataFrame(rows).sort_values(["k", "precision_at_k"], ascending=[True, False])
    result.to_csv(ANALYSIS_DIR / "topk_metrics.csv", index=False)
    return result


def load_error_feature_frame(best_model_df):
    cols = [
        "user_id",
        "product_id",
        "label",
        "user_total_orders",
        "user_avg_cart_size",
        "user_reorder_ratio",
        "product_total_orders",
        "product_reorder_rate",
        "product_avg_cart_position",
        "up_order_count",
        "up_order_rate",
        "up_orders_since_last",
        "department_id",
        "department_reorder_rate",
    ]
    key_frame = best_model_df[["user_id", "product_id"]].drop_duplicates()
    feature_chunks = []
    for chunk in pd.read_csv(FEATURE_PATH, usecols=cols, chunksize=500_000):
        matched = chunk.merge(key_frame, on=["user_id", "product_id"], how="inner")
        if not matched.empty:
            feature_chunks.append(matched)

    if not feature_chunks:
        raise ValueError("No matching feature rows found for best model predictions")

    feature_df = pd.concat(feature_chunks, ignore_index=True)
    test_keys = best_model_df[["user_id", "product_id", "y_true", "y_pred", "y_proba"]]
    merged = test_keys.merge(
        feature_df,
        left_on=["user_id", "product_id", "y_true"],
        right_on=["user_id", "product_id", "label"],
        how="left",
    )
    merged.drop(columns=["label"], inplace=True)
    return merged


def bin_series(series, bins, labels):
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def error_analysis(best_model_name, best_model_df):
    df = load_error_feature_frame(best_model_df)
    df["error_type"] = np.select(
        [
            (df["y_true"] == 1) & (df["y_pred"] == 1),
            (df["y_true"] == 0) & (df["y_pred"] == 1),
            (df["y_true"] == 1) & (df["y_pred"] == 0),
            (df["y_true"] == 0) & (df["y_pred"] == 0),
        ],
        ["TP", "FP", "FN", "TN"],
        default="unknown",
    )

    summary = (
        df.groupby("error_type", observed=True)
        .agg(
            rows=("y_true", "size"),
            avg_proba=("y_proba", "mean"),
            avg_user_orders=("user_total_orders", "mean"),
            avg_cart_size=("user_avg_cart_size", "mean"),
            avg_user_reorder_ratio=("user_reorder_ratio", "mean"),
            avg_product_orders=("product_total_orders", "mean"),
            avg_product_reorder_rate=("product_reorder_rate", "mean"),
            avg_cart_position=("product_avg_cart_position", "mean"),
            avg_up_order_count=("up_order_count", "mean"),
            avg_up_order_rate=("up_order_rate", "mean"),
            avg_orders_since_last=("up_orders_since_last", "mean"),
            avg_department_reorder_rate=("department_reorder_rate", "mean"),
        )
        .reset_index()
        .sort_values("error_type")
    )
    summary.to_csv(ANALYSIS_DIR / "error_type_feature_summary.csv", index=False)

    df["up_order_count_bin"] = bin_series(
        df["up_order_count"],
        bins=[-0.1, 1, 2, 4, math.inf],
        labels=["1", "2", "3-4", "5+"],
    )
    df["orders_since_last_bin"] = bin_series(
        df["up_orders_since_last"],
        bins=[-0.1, 0, 1, 3, 6, math.inf],
        labels=["0", "1", "2-3", "4-6", "7+"],
    )
    df["product_reorder_rate_bin"] = bin_series(
        df["product_reorder_rate"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    )

    slice_rows = []
    for col in ["up_order_count_bin", "orders_since_last_bin", "product_reorder_rate_bin", "department_id"]:
        grouped = df.groupby(col, observed=True)
        for group, part in grouped:
            tp = ((part["y_true"] == 1) & (part["y_pred"] == 1)).sum()
            fp = ((part["y_true"] == 0) & (part["y_pred"] == 1)).sum()
            fn = ((part["y_true"] == 1) & (part["y_pred"] == 0)).sum()
            positive = (part["y_true"] == 1).sum()
            predicted_positive = (part["y_pred"] == 1).sum()
            slice_rows.append(
                {
                    "slice": col,
                    "value": group,
                    "rows": len(part),
                    "positive_rate": positive / len(part),
                    "precision": tp / predicted_positive if predicted_positive else np.nan,
                    "recall": tp / positive if positive else np.nan,
                    "false_positive_rate": fp / (part["y_true"].eq(0).sum()),
                    "false_negative_rate": fn / positive if positive else np.nan,
                }
            )
    slice_df = pd.DataFrame(slice_rows)
    slice_df.to_csv(ANALYSIS_DIR / "error_slice_analysis.csv", index=False)
    return df, summary, slice_df


def annotate_bars(ax):
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.3f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )


def plot_metric_comparison(metrics):
    metrics = metrics.sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.2
    for i, metric in enumerate(["precision", "recall", "f1", "pr_auc"]):
        ax.bar(x + (i - 1.5) * width, metrics[metric], width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["model"], rotation=20, ha="right")
    ax.set_ylim(0, max(metrics["recall"].max(), metrics["pr_auc"].max()) * 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Metric Comparison")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_metric_comparison.png", dpi=180)
    plt.close(fig)


def plot_curves(predictions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, df in predictions.items():
        y_true = df["y_true"]
        y_proba = df["y_proba"]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        axes[0].plot(fpr, tpr, label=f"{model} ({roc_auc_score(y_true, y_proba):.3f})")
        axes[1].plot(recall, precision, label=f"{model} ({average_precision_score(y_true, y_proba):.3f})")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_pr_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion(confusion_df):
    labels = confusion_df["model"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, confusion_df["false_positive_rate"], 0.4, label="False Positive Rate")
    ax.bar(x + 0.2, confusion_df["false_negative_rate"], 0.4, label="False Negative Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Error Rate by Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_error_rates.png", dpi=180)
    plt.close(fig)


def plot_topk(topk_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, part in topk_df.groupby("model"):
        part = part.sort_values("k")
        ax.plot(part["k"], part["precision_at_k"], marker="o", label=model)
    ax.set_xlabel("K")
    ax.set_ylabel("Precision@K")
    ax.set_title("Top-K Recommendation Precision")
    ax.set_xticks(TOP_K_VALUES)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "topk_precision.png", dpi=180)
    plt.close(fig)


def plot_error_type_summary(error_summary):
    fig, ax = plt.subplots(figsize=(8, 5))
    error_summary = error_summary.sort_values("error_type")
    ax.bar(error_summary["error_type"], error_summary["rows"])
    ax.set_title("Best Model Error Type Counts")
    ax.set_xlabel("Error Type")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "best_model_error_types.png", dpi=180)
    plt.close(fig)


def plot_error_slices(slice_df):
    subset = slice_df[slice_df["slice"].eq("orders_since_last_bin")].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(subset["value"].astype(str), subset["recall"], marker="o", label="Recall")
    ax.plot(subset["value"].astype(str), subset["precision"], marker="o", label="Precision")
    ax.set_title("Best Model Metrics by Recency Segment")
    ax.set_xlabel("Orders Since Last Purchase")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "recency_error_slice.png", dpi=180)
    plt.close(fig)


def write_report(metrics, confusion_df, bootstrap_df, mcnemar_df, topk_df, best_model, error_summary):
    top_metrics = metrics.sort_values("f1", ascending=False).iloc[0]
    best_ci = bootstrap_df[
        (bootstrap_df["model"] == best_model) & (bootstrap_df["metric"].isin(["f1", "roc_auc", "pr_auc"]))
    ]
    best_topk = topk_df[topk_df["model"] == best_model].sort_values("k")
    significant_pairs = int(mcnemar_df["significant_at_0.05"].sum())

    lines = [
        "# 评估与分析报告章节",
        "",
        "## 1. 评估目标与数据范围",
        "",
        "本章节面向 Instacart 复购预测任务，评估对象为各训练脚本在同一测试集上导出的预测结果。测试集按 user_id 划分，避免同一用户同时出现在训练集和测试集造成泄露。当前可用于逐样本分析的预测文件包括 Logistic Regression、Decision Tree、Random Forest 和 LightGBM；XGBoost 已进入指标汇总表，但缺少 `output/predictions/xgboost_pred.csv`，因此不参与配对显著性检验、Top-K 和错误切片分析。",
        "",
        "## 2. 指标设计",
        "",
        "- 分类阈值指标：Precision、Recall、F1，用于衡量推荐商品命中质量和召回能力。",
        "- 排序指标：ROC-AUC、PR-AUC，用于衡量概率排序能力；由于正样本比例约 6.2%，PR-AUC 更能反映正类识别质量。",
        "- 概率质量：LogLoss，用于评估概率校准和置信度。",
        "- 推荐场景指标：Precision@K、Recall@K、用户 Top-K 命中率，用于模拟每个用户推荐前 K 个历史商品时的业务表现。",
        "",
        "## 3. 综合结果",
        "",
        f"按 F1 排名，当前逐样本预测文件中的最佳模型为 **{best_model}**，测试集 F1={top_metrics['f1']:.4f}，Precision={top_metrics['precision']:.4f}，Recall={top_metrics['recall']:.4f}，ROC-AUC={top_metrics['roc_auc']:.4f}，PR-AUC={top_metrics['pr_auc']:.4f}。",
        "",
        "主要图表：",
        "",
        "- `figure/evaluation/model_metric_comparison.png`：各模型 Precision、Recall、F1、PR-AUC 对比。",
        "- `figure/evaluation/roc_pr_curves.png`：ROC 与 PR 曲线。",
        "- `figure/evaluation/model_error_rates.png`：假阳性率与假阴性率对比。",
        "- `figure/evaluation/topk_precision.png`：Top-K 推荐精度。",
        "",
        "## 4. 统计显著性检验",
        "",
        f"对拥有逐样本预测文件的模型进行了 McNemar 配对检验。该检验比较两个模型在同一批样本上的正确/错误差异，适合判断分类器差异是否只是随机波动。当前共有 {len(mcnemar_df)} 组模型对，其中 {significant_pairs} 组在 0.05 水平下显著。",
        "",
        "同时对 F1、ROC-AUC、PR-AUC 做了 bootstrap 95% 置信区间。最佳模型的区间如下：",
        "",
        "| 指标 | 均值 | 95% CI |",
        "|---|---:|---:|",
    ]
    for _, row in best_ci.iterrows():
        lines.append(f"| {row['metric']} | {row['mean']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |")

    lines.extend(
        [
            "",
            "## 5. 推荐 Top-K 分析",
            "",
            f"以 {best_model} 为例，按每个用户的预测概率排序后取 Top-K，得到：",
            "",
            "| K | Precision@K | Recall@K | 用户命中率 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in best_topk.iterrows():
        lines.append(
            f"| {int(row['k'])} | {row['precision_at_k']:.4f} | {row['recall_at_k']:.4f} | {row['users_with_positive_in_topk']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 6. 错误分析",
            "",
            "错误样本按 TP、FP、FN、TN 分组，并回连特征表进行特征均值对比。输出文件 `output/analysis/error_type_feature_summary.csv` 可直接查看不同错误类型的用户历史订单数、商品复购率、用户-商品历史购买次数、距最近购买订单数等差异。",
            "",
            "从业务含义看：",
            "",
            "- FP 表示模型推荐了用户下一单实际未购买的商品，通常会拉低 Precision。",
            "- FN 表示用户实际复购但模型未推荐，直接损伤 Recall。",
            "- `up_orders_since_last` 越大，用户近期兴趣越弱，模型更容易出现漏判或低置信判断。",
            "- `up_order_count` 和 `up_order_rate` 反映用户-商品绑定强度，是解释模型命中与误判的重要维度。",
            "",
            "相关图表：",
            "",
            "- `figure/evaluation/best_model_error_types.png`：最佳模型 TP/FP/FN/TN 数量。",
            "- `figure/evaluation/recency_error_slice.png`：按距最近购买订单数分段的 Precision/Recall。",
            "",
            "## 7. 结论与建议",
            "",
            f"当前最佳逐样本模型是 {best_model}。如果目标是提高命中质量，应优先优化 Precision 和 PR-AUC；如果目标是尽可能覆盖潜在复购商品，则应关注 Recall@K 和 FN 切片。下一步建议补齐 XGBoost 预测明细文件，并增加时间衰减、商品周期性、用户最近 N 单行为等特征，再复跑本评估脚本进行可比分析。",
            "",
        ]
    )

    path = PROJECT_ROOT / "doc" / "评估与分析报告.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    ensure_dirs()
    predictions = load_predictions()
    metrics = compute_metric_table(predictions)
    confusion_df = compute_confusion_tables(predictions)
    bootstrap_df = bootstrap_metric_ci(predictions)
    mcnemar_df = mcnemar_pairwise_tests(predictions)
    topk_df = compute_topk_metrics(predictions)

    best_model = metrics.sort_values("f1", ascending=False).iloc[0]["model"]
    error_df, error_summary, slice_df = error_analysis(best_model, predictions[best_model])

    plot_metric_comparison(metrics)
    plot_curves(predictions)
    plot_confusion(confusion_df)
    plot_topk(topk_df)
    plot_error_type_summary(error_summary)
    plot_error_slices(slice_df)

    report_path = write_report(
        metrics=metrics,
        confusion_df=confusion_df,
        bootstrap_df=bootstrap_df,
        mcnemar_df=mcnemar_df,
        topk_df=topk_df,
        best_model=best_model,
        error_summary=error_summary,
    )

    print("Evaluation analysis completed.")
    print(f"Models analyzed: {', '.join(metrics['model'])}")
    print(f"Best model by F1: {best_model}")
    print(f"Analysis CSVs: {ANALYSIS_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
