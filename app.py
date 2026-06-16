from __future__ import annotations

import os
import subprocess
import sys
from html import escape
from pathlib import Path
from textwrap import dedent

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DOC_DIR = PROJECT_ROOT / "doc"
FIGURE_DIR = PROJECT_ROOT / "figure"
EVAL_FIGURE_DIR = FIGURE_DIR / "evaluation"
OUTPUT_DIR = PROJECT_ROOT / "output"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
FEATURE_TABLE_PATH = OUTPUT_DIR / "feature_table.csv"
PREPROCESSOR_PATH = OUTPUT_DIR / "preprocessor" / "lr_preprocessor.pkl"
INFERENCE_EXCLUDE_COLS = {"user_id", "product_id", "label", "reordered"}

RAW_DATA_FILES = [
    "aisles.csv",
    "departments.csv",
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
]

TRAINING_JOBS = {
    "Logistic Regression": {
        "script": PROJECT_ROOT / "code" / "train" / "train_logistic_regression.py",
        "model": OUTPUT_DIR / "models" / "logistic_regression.pkl",
        "prediction": OUTPUT_DIR / "predictions" / "logistic_regression_pred.csv",
    },
    "Decision Tree": {
        "script": PROJECT_ROOT / "code" / "train" / "train_decision_tree.py",
        "model": OUTPUT_DIR / "models" / "decision_tree.pkl",
        "prediction": OUTPUT_DIR / "predictions" / "decision_tree_pred.csv",
    },
    "Random Forest": {
        "script": PROJECT_ROOT / "code" / "train" / "train_random_forest.py",
        "model": OUTPUT_DIR / "models" / "random_forest.pkl",
        "prediction": OUTPUT_DIR / "predictions" / "random_forest_pred.csv",
    },
    "LightGBM": {
        "script": PROJECT_ROOT / "code" / "train" / "train_lightgbm.py",
        "model": OUTPUT_DIR / "models" / "lightgbm.pkl",
        "prediction": OUTPUT_DIR / "predictions" / "lightgbm_pred.csv",
    },
    "XGBoost": {
        "script": PROJECT_ROOT / "code" / "train" / "train_xgboost.py",
        "model": OUTPUT_DIR / "models" / "xgboost.json",
        "prediction": OUTPUT_DIR / "predictions" / "xgboost_pred.csv",
    },
}

DOC_FILES = {
    "README": PROJECT_ROOT / "Readme.md",
    "特征说明": DOC_DIR / "feature_description.md",
    "训练计划书": DOC_DIR / "训练计划书.md",
    "训练报告": DOC_DIR / "训练报告.md",
    "训练运行指南": DOC_DIR / "训练运行指南.md",
    "评估与分析报告": DOC_DIR / "评估与分析报告.md",
}


st.set_page_config(
    page_title="Instacart 项目交互界面",
    page_icon="🛒",
    layout="wide",
)


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #f3f6fb;
            --bg-accent: #edf3ff;
            --surface: rgba(255, 255, 255, 0.96);
            --surface-strong: #ffffff;
            --text-primary: #18253d;
            --text-secondary: #5c6b82;
            --brand-700: #1849c6;
            --brand-600: #2f6fed;
            --brand-500: #5c8df6;
            --success-bg: rgba(15, 157, 88, 0.12);
            --success-text: #0a6b3c;
            --warning-bg: rgba(242, 153, 74, 0.14);
            --warning-text: #9b5d12;
            --border-soft: rgba(47, 111, 237, 0.12);
            --shadow-soft: 0 10px 30px rgba(24, 37, 61, 0.08);
            --shadow-strong: 0 18px 38px rgba(24, 73, 198, 0.18);
        }
        .stApp {
            background: linear-gradient(180deg, #f8faff 0%, var(--bg-primary) 100%);
            color: var(--text-primary);
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.2rem;
            max-width: 1440px;
        }
        .hero-card {
            padding: 1.6rem 1.7rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #123689 0%, #2458d3 55%, #6e9fff 100%);
            color: #ffffff;
            box-shadow: var(--shadow-strong);
            margin-bottom: 1.1rem;
        }
        .hero-eyebrow {
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 999px;
            padding: 0.28rem 0.68rem;
            margin-bottom: 0.85rem;
        }
        .hero-title {
            font-size: 2.05rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.45rem;
        }
        .hero-desc {
            font-size: 1rem;
            line-height: 1.65;
            opacity: 0.94;
            margin-bottom: 0.85rem;
            max-width: 860px;
        }
        .hero-caption {
            font-size: 0.92rem;
            opacity: 0.88;
            margin-bottom: 0;
        }
        .section-card {
            padding: 1.05rem 1.15rem;
            border-radius: 18px;
            background: var(--surface);
            border: 1px solid var(--border-soft);
            box-shadow: var(--shadow-soft);
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.4rem;
        }
        .section-desc {
            color: var(--text-secondary);
            line-height: 1.65;
            margin-bottom: 0;
        }
        .spotlight-card {
            padding: 1.1rem 1.2rem;
            border-radius: 18px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid rgba(24, 73, 198, 0.14);
            box-shadow: var(--shadow-soft);
            margin-bottom: 1rem;
        }
        .spotlight-label {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--brand-700);
            margin-bottom: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .spotlight-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.55rem;
        }
        .spotlight-meta {
            color: var(--text-secondary);
            margin-bottom: 0.25rem;
        }
        .spotlight-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.85rem;
        }
        .spotlight-metric {
            padding: 0.8rem 0.85rem;
            border-radius: 14px;
            background: rgba(24, 73, 198, 0.05);
            border: 1px solid rgba(24, 73, 198, 0.1);
        }
        .spotlight-metric-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
        }
        .spotlight-metric-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.2;
        }
        .list-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(47, 111, 237, 0.1);
            margin-bottom: 1rem;
        }
        .list-title {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.7rem;
        }
        .list-card ul {
            margin: 0;
            padding-left: 1.1rem;
            color: var(--text-secondary);
            line-height: 1.7;
        }
        .list-card li + li {
            margin-top: 0.25rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.34rem 0.76rem;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            border-radius: 999px;
            font-size: 0.88rem;
            font-weight: 600;
        }
        .status-ok {
            background: var(--success-bg);
            color: var(--success-text);
        }
        .status-warn {
            background: var(--warning-bg);
            color: var(--warning-text);
        }
        .caption-text {
            color: var(--text-secondary);
            font-size: 0.92rem;
            margin-bottom: 0.7rem;
        }
        div[data-testid="stMetric"] {
            background: var(--surface-strong);
            border: 1px solid var(--border-soft);
            padding: 0.9rem 1rem;
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(31, 46, 89, 0.07);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 12px;
            min-height: 2.8rem;
            font-weight: 600;
        }
        div[data-testid="stTabs"] button {
            border-radius: 10px 10px 0 0;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f4f7ff 100%);
            border-right: 1px solid rgba(47, 111, 237, 0.1);
        }
        .sidebar-brand {
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 1rem;
            border-radius: 18px;
            background: linear-gradient(145deg, #123689 0%, #2458d3 100%);
            color: #ffffff;
            box-shadow: var(--shadow-soft);
        }
        .sidebar-brand-tag {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.9;
            margin-bottom: 0.35rem;
        }
        .sidebar-brand-title {
            font-size: 1.18rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .sidebar-brand-desc {
            font-size: 0.9rem;
            line-height: 1.6;
            opacity: 0.92;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_read_markdown(path: Path) -> str:
    if not path.exists():
        return f"文件不存在: `{path}`"
    return path.read_text(encoding="utf-8")


def safe_read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, nrows=nrows)


@st.cache_data(show_spinner=False)
def load_feature_samples(limit: int) -> pd.DataFrame | None:
    if not FEATURE_TABLE_PATH.exists():
        return None
    return pd.read_csv(FEATURE_TABLE_PATH, nrows=limit)


@st.cache_resource(show_spinner=False)
def load_saved_model(model_name: str, model_path: str):
    if model_name == "XGBoost":
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_lr_preprocessor_resource(preprocessor_path: str):
    return joblib.load(preprocessor_path)


def get_inference_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in INFERENCE_EXCLUDE_COLS]


def compare_models_on_sample(sample_row: pd.Series, feature_columns: list[str]) -> pd.DataFrame:
    sample_features = pd.DataFrame([sample_row[feature_columns]])
    true_label = sample_row.get("label")
    rows = []

    for model_name, cfg in TRAINING_JOBS.items():
        model_path = cfg["model"]
        if not model_path.exists():
            rows.append(
                {
                    "模型": model_name,
                    "状态": "模型文件缺失",
                    "预测标签": None,
                    "正类概率": None,
                    "置信度": None,
                    "真实标签": int(true_label) if pd.notna(true_label) else None,
                    "是否命中": None,
                }
            )
            continue

        try:
            model = load_saved_model(model_name, str(model_path))
            model_input = sample_features

            if model_name == "Logistic Regression":
                if not PREPROCESSOR_PATH.exists():
                    raise FileNotFoundError(f"预处理器不存在: {PREPROCESSOR_PATH}")
                preprocessor = load_lr_preprocessor_resource(str(PREPROCESSOR_PATH))
                model_input = preprocessor.transform(sample_features)

            pred_label = int(model.predict(model_input)[0])
            if hasattr(model, "predict_proba"):
                positive_proba = float(model.predict_proba(model_input)[0][1])
            else:
                positive_proba = float(pred_label)

            rows.append(
                {
                    "模型": model_name,
                    "状态": "可用",
                    "预测标签": pred_label,
                    "正类概率": positive_proba,
                    "置信度": max(positive_proba, 1 - positive_proba),
                    "真实标签": int(true_label) if pd.notna(true_label) else None,
                    "是否命中": (
                        "是"
                        if pd.notna(true_label) and pred_label == int(true_label)
                        else "否"
                        if pd.notna(true_label)
                        else "-"
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "模型": model_name,
                    "状态": f"加载失败: {exc}",
                    "预测标签": None,
                    "正类概率": None,
                    "置信度": None,
                    "真实标签": int(true_label) if pd.notna(true_label) else None,
                    "是否命中": None,
                }
            )

    result = pd.DataFrame(rows)
    available_df = result[result["正类概率"].notna()].sort_values("正类概率", ascending=False)
    unavailable_df = result[result["正类概率"].isna()]
    return pd.concat([available_df, unavailable_df], ignore_index=True)


def build_raw_data_status() -> pd.DataFrame:
    rows = []
    for name in RAW_DATA_FILES:
        path = DATA_DIR / name
        rows.append(
            {
                "文件": name,
                "是否存在": "是" if path.exists() else "否",
                "大小(MB)": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else None,
            }
        )
    return pd.DataFrame(rows)


def build_training_status() -> pd.DataFrame:
    rows = []
    for model_name, cfg in TRAINING_JOBS.items():
        rows.append(
            {
                "模型": model_name,
                "训练脚本": str(cfg["script"].relative_to(PROJECT_ROOT)),
                "模型文件": "已生成" if cfg["model"].exists() else "未生成",
                "预测文件": "已生成" if cfg["prediction"].exists() else "未生成",
            }
        )
    return pd.DataFrame(rows)


def build_overview_metrics() -> dict[str, int]:
    model_count = sum(1 for cfg in TRAINING_JOBS.values() if cfg["model"].exists())
    prediction_count = sum(1 for cfg in TRAINING_JOBS.values() if cfg["prediction"].exists())
    analysis_count = len(list(ANALYSIS_DIR.glob("*.csv"))) if ANALYSIS_DIR.exists() else 0
    eval_figure_count = len(list(EVAL_FIGURE_DIR.glob("*.png"))) if EVAL_FIGURE_DIR.exists() else 0
    raw_count = sum(1 for name in RAW_DATA_FILES if (DATA_DIR / name).exists())
    metrics = {
        "已训练模型": model_count,
        "预测结果文件": prediction_count,
        "分析表": analysis_count,
        "评估图": eval_figure_count,
    }
    if raw_count > 0:
        metrics = {"原始数据文件": raw_count, **metrics}
    return metrics


def get_metrics_summary() -> pd.DataFrame | None:
    metrics_summary = safe_read_csv(OUTPUT_DIR / "metrics_summary.csv")
    if metrics_summary is None or metrics_summary.empty:
        return None
    return metrics_summary


def get_best_model_row(metrics_summary: pd.DataFrame | None) -> pd.Series | None:
    if metrics_summary is None or metrics_summary.empty:
        return None
    if "f1" not in metrics_summary.columns:
        return metrics_summary.iloc[0]
    return metrics_summary.sort_values("f1", ascending=False).iloc[0]


def render_topk_snapshot(topk_df: pd.DataFrame | None) -> None:
    st.subheader("Top-K 业务指标")
    if topk_df is None or topk_df.empty:
        st.info("暂无可展示的 Top-K 图表。")
        return

    required_columns = {"model", "k", "precision_at_k", "recall_at_k"}
    if not required_columns.issubset(topk_df.columns):
        st.info("当前 Top-K 结果字段不完整。")
        return

    precision_chart = (
        topk_df.pivot(index="k", columns="model", values="precision_at_k")
        .sort_index()
    )
    recall_chart = (
        topk_df.pivot(index="k", columns="model", values="recall_at_k")
        .sort_index()
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Precision@K")
        st.line_chart(precision_chart, use_container_width=True)
    with col2:
        st.markdown("#### Recall@K")
        st.line_chart(recall_chart, use_container_width=True)


def render_significance_snapshot(
    mcnemar_df: pd.DataFrame | None,
    bootstrap_df: pd.DataFrame | None,
) -> None:
    st.subheader("统计检验摘要")
    left_col, right_col = st.columns(2)

    with left_col:
        if mcnemar_df is None or mcnemar_df.empty:
            st.info("暂无 McNemar 检验结果。")
        else:
            significant_count = int(mcnemar_df["significant_at_0.05"].fillna(False).sum())
            pair_count = len(mcnemar_df)
            best_pair = mcnemar_df.sort_values("p_value", ascending=True).iloc[0]
            st.metric("显著差异模型对", f"{significant_count}/{pair_count}")
            st.dataframe(
                mcnemar_df[["model_a", "model_b", "better_by_accuracy", "p_value"]].sort_values("p_value"),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "最显著的一组: "
                f"{best_pair['model_a']} vs {best_pair['model_b']}，"
                f"更优模型为 {best_pair['better_by_accuracy']}。"
            )

    with right_col:
        if bootstrap_df is None or bootstrap_df.empty:
            st.info("暂无 Bootstrap 置信区间结果。")
        else:
            focus_df = bootstrap_df[bootstrap_df["metric"].isin(["f1", "pr_auc", "roc_auc"])].copy()
            if focus_df.empty:
                st.dataframe(bootstrap_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(
                    focus_df.sort_values(["metric", "mean"], ascending=[True, False]),
                    use_container_width=True,
                    hide_index=True,
                )


def render_embedded_figure(path: Path, title: str, caption: str) -> None:
    st.markdown(f"#### {title}")
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"尚未生成 `{path.relative_to(PROJECT_ROOT)}`。")


def render_page_header(title: str, description: str, eyebrow: str, caption: str | None = None) -> None:
    st.markdown(
        dedent(
            f"""
            <div class="hero-card">
                <div class="hero-eyebrow">{escape(eyebrow)}</div>
                <div class="hero-title">{escape(title)}</div>
                <p class="hero-desc">{escape(description)}</p>
                {f'<p class="hero-caption">{escape(caption)}</p>' if caption else ''}
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_spotlight_card(label: str, title: str, lines: list[str]) -> None:
    meta_html = "".join(f'<div class="spotlight-meta">{escape(line)}</div>' for line in lines)
    st.markdown(
        dedent(
            f"""
            <div class="spotlight-card">
                <div class="spotlight-label">{escape(label)}</div>
                <div class="spotlight-title">{escape(title)}</div>
                {meta_html}
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_best_model_card(best_row: pd.Series) -> None:
    metrics = [
        ("F1", f"{best_row['f1']:.4f}"),
        ("Recall", f"{best_row['recall']:.4f}"),
        ("PR-AUC", f"{best_row['pr_auc']:.4f}"),
    ]
    metrics_html = "".join(
        dedent(
            f"""
            <div class="spotlight-metric">
                <div class="spotlight-metric-label">{escape(label)}</div>
                <div class="spotlight-metric-value">{escape(value)}</div>
            </div>
            """
        ).strip()
        for label, value in metrics
    )
    st.markdown(
        dedent(
            f"""
            <div class="spotlight-card">
                <div class="spotlight-label">当前最佳模型</div>
                <div class="spotlight-title">{escape(str(best_row["model"]))}</div>
                <div class="spotlight-metrics">{metrics_html}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_pipeline_status_banner() -> None:
    feature_ready = FEATURE_TABLE_PATH.exists()
    model_count = sum(1 for cfg in TRAINING_JOBS.values() if cfg["model"].exists())
    prediction_count = sum(1 for cfg in TRAINING_JOBS.values() if cfg["prediction"].exists())
    analysis_ready = ANALYSIS_DIR.exists() and any(ANALYSIS_DIR.glob("*.csv"))

    pills = [
        (
            "特征表已生成" if feature_ready else "特征表缺失",
            "status-ok" if feature_ready else "status-warn",
        ),
        (
            f"模型产物 {model_count}/{len(TRAINING_JOBS)}",
            "status-ok" if model_count else "status-warn",
        ),
        (
            f"预测产物 {prediction_count}/{len(TRAINING_JOBS)}",
            "status-ok" if prediction_count else "status-warn",
        ),
        (
            "分析结果已生成" if analysis_ready else "分析结果缺失",
            "status-ok" if analysis_ready else "status-warn",
        ),
    ]
    status_html = "".join(
        f'<span class="status-pill {css_class}">{label}</span>'
        for label, css_class in pills
    )
    st.markdown(status_html, unsafe_allow_html=True)


def run_python_script(script_path: Path, title: str) -> bool:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    with st.spinner(f"正在执行: {title}"):
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    st.subheader(f"{title} 日志")
    output = result.stdout.strip()
    error_output = result.stderr.strip()

    if output:
        st.code(output, language="text")
    else:
        st.info("标准输出为空。")

    if error_output:
        st.subheader("错误输出")
        st.code(error_output, language="text")

    if result.returncode == 0:
        st.success(f"{title} 执行完成。")
        return True

    st.error(f"{title} 执行失败，退出码: {result.returncode}")
    return False


def render_overview() -> None:
    render_page_header(
        "Instacart 用户复购预测工作台",
        "统一查看项目状态、模型结果与关键产出。",
        "Overview",
    )
    render_pipeline_status_banner()
    metrics_summary = get_metrics_summary()
    best_row = get_best_model_row(metrics_summary)
    metrics = build_overview_metrics()

    if best_row is not None:
        render_best_model_card(best_row)
    else:
        render_spotlight_card(
            "当前最佳模型",
            "尚无可用训练结果",
            ["未检测到 output/metrics_summary.csv", "建议先完成模型训练后再查看结果。"],
        )

    cols = st.columns(len(metrics))
    for idx, (label, value) in enumerate(metrics.items()):
        cols[idx].metric(label, value)

    render_embedded_figure(
        EVAL_FIGURE_DIR / "model_metric_comparison.png",
        "模型指标图",
        "各模型 Precision、Recall、F1 和 PR-AUC 对比",
    )

    summary_tab, metrics_tab, data_tab = st.tabs(["训练状态", "模型表现", "数据资产"])
    with summary_tab:
        st.dataframe(build_training_status(), use_container_width=True, hide_index=True)
    with metrics_tab:
        if metrics_summary is not None:
            st.dataframe(
                metrics_summary.sort_values("f1", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            st.divider()
            render_topk_snapshot(safe_read_csv(ANALYSIS_DIR / "topk_metrics.csv"))
        else:
            st.warning("暂未发现 `output/metrics_summary.csv`，请先完成模型训练。")
    with data_tab:
        st.markdown('<div class="caption-text">检查原始数据文件是否齐备，并确认后续流程的输入基础。</div>', unsafe_allow_html=True)
        st.dataframe(build_raw_data_status(), use_container_width=True, hide_index=True)


def render_feature_engineering() -> None:
    render_page_header(
        "特征工程中心",
        "围绕数据准备、特征表产出和样例预览组织流程，让页面既能操作也能解释当前状态。",
        "Feature Engineering",
        "该页面优先回答两个问题：数据是否齐备，特征表是否已生成。",
    )

    feature_path = OUTPUT_DIR / "feature_table.csv"

    if feature_path.exists():
        st.success(f"已检测到特征表: `{feature_path.relative_to(PROJECT_ROOT)}`")
        preview_df = safe_read_csv(feature_path, nrows=20)
        if preview_df is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("样例行数", len(preview_df))
            c2.metric("列数", preview_df.shape[1])
            c3.metric("文件大小(MB)", round(feature_path.stat().st_size / 1024 / 1024, 2))
            st.dataframe(preview_df, use_container_width=True)
    else:
        st.warning("尚未生成 `output/feature_table.csv`。")

    if st.button("运行特征工程", type="primary", use_container_width=True):
        success = run_python_script(PROJECT_ROOT / "code" / "feature_engineering.py", "特征工程")
        if success:
            st.rerun()


def render_training() -> None:
    render_page_header(
        "模型训练中心",
        "将选择模型、执行训练和查看结果放在一个页面内，减少操作跳转并保留清晰的状态反馈。",
        "Model Training",
        "推荐先完成特征工程，再按顺序执行基线模型与集成模型训练。",
    )
    control_col, guide_col = st.columns([1.25, 0.9])
    with control_col:
        with st.form("training_form"):
            selected_models = st.multiselect(
                "选择要训练的模型",
                list(TRAINING_JOBS.keys()),
                default=["Logistic Regression"],
            )
            run_all_recommended = st.checkbox("使用推荐顺序训练全部模型")
            submitted = st.form_submit_button("开始训练", type="primary", use_container_width=True)

        if submitted:
            execution_list = (
                list(TRAINING_JOBS.keys())
                if run_all_recommended
                else selected_models
            )
            if not execution_list:
                st.warning("请先选择至少一个模型。")
                return

            for model_name in execution_list:
                cfg = TRAINING_JOBS[model_name]
                st.divider()
                run_python_script(cfg["script"], f"{model_name} 训练")

            st.rerun()
    with guide_col:
        render_spotlight_card(
            "推荐顺序",
            "推荐顺序",
            [
                "Logistic Regression -> Decision Tree -> Random Forest",
                "再执行 LightGBM 与 XGBoost 对比提升空间",
                "训练完成后立即回看 F1、Recall、PR-AUC",
            ],
        )

    st.subheader("当前训练状态")
    st.dataframe(build_training_status(), use_container_width=True, hide_index=True)

    metrics_summary = get_metrics_summary()
    if metrics_summary is not None:
        st.subheader("训练结果汇总")
        st.dataframe(
            metrics_summary.sort_values("f1", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def render_analysis() -> None:
    render_page_header(
        "评估分析中心",
        "集中展示指标重算、Top-K、显著性检验和 Bootstrap 结果，让结论展示与统计佐证在同一视图完成。",
        "Evaluation",
        "分析页负责回答“哪个模型更好、差异是否显著、结论是否稳定”。",
    )
    if st.button("运行评估分析", type="primary", use_container_width=True):
        success = run_python_script(PROJECT_ROOT / "code" / "evaluate_analysis.py", "评估分析")
        if success:
            st.rerun()

    metrics_df = safe_read_csv(ANALYSIS_DIR / "evaluation_metrics_recomputed.csv")
    topk_df = safe_read_csv(ANALYSIS_DIR / "topk_metrics.csv")
    mcnemar_df = safe_read_csv(ANALYSIS_DIR / "mcnemar_pairwise_tests.csv")
    bootstrap_df = safe_read_csv(ANALYSIS_DIR / "bootstrap_ci.csv")

    tab_metrics, tab_tests, tab_report = st.tabs(["指标结论", "统计检验", "评估报告"])
    with tab_metrics:
        st.subheader("重算指标表")
        if metrics_df is not None:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.info("尚未发现 `output/analysis/evaluation_metrics_recomputed.csv`。")

        st.subheader("Top-K 指标")
        if topk_df is not None:
            st.dataframe(topk_df, use_container_width=True, hide_index=True)
            st.divider()
            render_topk_snapshot(topk_df)
        else:
            st.info("尚未发现 Top-K 结果。")

        st.subheader("关键评估图")
        figure_col1, figure_col2 = st.columns(2)
        with figure_col1:
            render_embedded_figure(
                EVAL_FIGURE_DIR / "roc_pr_curves.png",
                "ROC / PR 曲线",
                "用于观察各模型的排序区分能力。",
            )
        with figure_col2:
            render_embedded_figure(
                EVAL_FIGURE_DIR / "topk_precision.png",
                "Top-K 精度图",
                "适合在答辩中直接说明推荐质量。",
            )

    with tab_tests:
        st.subheader("McNemar 检验")
        if mcnemar_df is not None:
            st.dataframe(mcnemar_df, use_container_width=True, hide_index=True)
        else:
            st.info("尚未发现显著性检验结果。")

        st.subheader("Bootstrap 置信区间")
        if bootstrap_df is not None:
            st.dataframe(bootstrap_df, use_container_width=True, hide_index=True)
        else:
            st.info("尚未发现 Bootstrap 结果。")

        st.divider()
        render_significance_snapshot(mcnemar_df, bootstrap_df)

    with tab_report:
        report_path = DOC_DIR / "评估与分析报告.md"
        st.markdown(safe_read_markdown(report_path))


def render_single_prediction() -> None:
    render_page_header(
        "单条样本预测台",
        "从特征表中抽取一个样本，比较各模型在同一输入下的输出，适合演示模型差异和结果解释。",
        "Single Prediction",
        "交互上保留样本选择与结果展示两块区域，减少用户重复操作。",
    )
    if not FEATURE_TABLE_PATH.exists():
        st.warning("尚未生成 `output/feature_table.csv`，请先在“特征工程”页面生成特征表。")
        return

    if "prediction_compare_df" not in st.session_state:
        st.session_state["prediction_compare_df"] = None
        st.session_state["prediction_sample_key"] = None

    sample_pool_size = st.slider("读取样本池大小", 100, 5000, 1000, 100)
    sample_df = load_feature_samples(sample_pool_size)
    if sample_df is None or sample_df.empty:
        st.warning("当前无法读取演示样本。")
        return

    feature_columns = get_inference_feature_columns(sample_df)
    selected_index = st.slider("选择样本行", 0, len(sample_df) - 1, 0)
    sample_row = sample_df.iloc[selected_index]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("user_id", int(sample_row["user_id"]) if "user_id" in sample_row else "-")
    c2.metric("product_id", int(sample_row["product_id"]) if "product_id" in sample_row else "-")
    c3.metric("真实标签", int(sample_row["label"]) if "label" in sample_row else "-")
    c4.metric("历史 reordered", int(sample_row["reordered"]) if "reordered" in sample_row else "-")

    if st.button("执行单条样本预测", type="primary", use_container_width=True):
        st.session_state["prediction_compare_df"] = compare_models_on_sample(sample_row, feature_columns)
        st.session_state["prediction_sample_key"] = (sample_pool_size, selected_index)

    compare_df = st.session_state.get("prediction_compare_df")
    sample_key = st.session_state.get("prediction_sample_key")
    if compare_df is not None and sample_key == (sample_pool_size, selected_index):
        st.subheader("模型对比结果")
        available_df = compare_df[compare_df["状态"] == "可用"].copy()
        unavailable_df = compare_df[compare_df["状态"] != "可用"].copy()

        if not available_df.empty:
            available_df = available_df.sort_values("正类概率", ascending=False).reset_index(drop=True)
            card_columns = st.columns(min(3, len(available_df)))
            for idx, row in available_df.iterrows():
                with card_columns[idx % len(card_columns)]:
                    st.markdown(f"### {row['模型']}")
                    st.metric("预测标签", int(row["预测标签"]))
                    st.metric("正类概率", f"{row['正类概率']:.4f}")
                    st.metric("置信度", f"{row['置信度']:.4f}")
                    st.metric("是否命中真实标签", row["是否命中"])

        if not unavailable_df.empty:
            st.subheader("不可用模型")
            st.dataframe(unavailable_df, use_container_width=True, hide_index=True)

        st.subheader("完整对比表")
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

    st.subheader("当前样本特征")
    feature_view = sample_row[feature_columns].to_frame(name="value")
    st.dataframe(feature_view, use_container_width=True)


def render_docs_and_figures() -> None:
    render_page_header(
        "文档与图表",
        "把项目文档和图表收敛到同一页面，方便汇报时切换材料，也减少导航层级。",
        "Documentation",
        "文档负责解释方法，图表负责展示结论，两者分标签展示更符合汇报场景。",
    )
    tab_docs, tab_figures = st.tabs(["项目文档", "图表浏览"])
    with tab_docs:
        selected_doc = st.selectbox("选择文档", list(DOC_FILES.keys()))
        st.subheader(selected_doc)
        st.markdown(safe_read_markdown(DOC_FILES[selected_doc]))

    with tab_figures:
        image_paths = sorted(FIGURE_DIR.glob("*.png")) + sorted(EVAL_FIGURE_DIR.glob("*.png"))
        if not image_paths:
            st.info("当前没有可展示的 PNG 图表。")
            return

        selected_image = st.selectbox(
            "选择图表",
            [str(path.relative_to(PROJECT_ROOT)) for path in image_paths],
        )
        image_path = PROJECT_ROOT / selected_image
        st.image(str(image_path), caption=selected_image, use_container_width=True)


def main() -> None:
    inject_global_styles()
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-tag">Data Mining Project</div>
            <div class="sidebar-brand-title">Instacart 复购预测</div>
            <p class="sidebar-brand-desc">统一查看数据、模型、评估与样本推断，按仪表盘方式组织导航与状态。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "工作台导航",
        ["项目总览", "特征工程", "模型训练", "评估分析", "单条样本预测", "文档与图表"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 项目状态")
    st.sidebar.write(f"- 特征表：{'已生成' if FEATURE_TABLE_PATH.exists() else '未生成'}")
    st.sidebar.write(
        f"- 模型文件：{sum(1 for cfg in TRAINING_JOBS.values() if cfg['model'].exists())}/{len(TRAINING_JOBS)}"
    )
    st.sidebar.write(
        f"- 预测文件：{sum(1 for cfg in TRAINING_JOBS.values() if cfg['prediction'].exists())}/{len(TRAINING_JOBS)}"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 推荐流程")
    st.sidebar.write("1. 特征工程")
    st.sidebar.write("2. 模型训练")
    st.sidebar.write("3. 评估分析")
    st.sidebar.write("4. 单条样本预测")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**运行说明**")
    st.sidebar.code("streamlit run app.py", language="bash")
    st.sidebar.markdown("建议在项目根目录运行。")

    if page == "项目总览":
        render_overview()
    elif page == "特征工程":
        render_feature_engineering()
    elif page == "模型训练":
        render_training()
    elif page == "评估分析":
        render_analysis()
    elif page == "单条样本预测":
        render_single_prediction()
    else:
        render_docs_and_figures()


if __name__ == "__main__":
    main()
