"""
全局配置 - 路径、随机种子、模型超参数默认值。
所有训练脚本统一引用此模块，保证一致性。
"""
import os

# ==================== 项目根目录与路径 ====================

# 项目根目录：当前文件位于 code/train/，向上两级即项目根
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# 数据路径
DATA_PATH = os.path.join(PROJECT_ROOT, "output", "feature_table.csv")

# 划分后数据集存放目录
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
SPLIT_TRAIN_PATH = os.path.join(SPLIT_DIR, "train.csv")
SPLIT_VAL_PATH = os.path.join(SPLIT_DIR, "val.csv")
SPLIT_TEST_PATH = os.path.join(SPLIT_DIR, "test.csv")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
PREPROCESSOR_DIR = os.path.join(OUTPUT_DIR, "preprocessor")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics_summary.csv")

# 确保输出目录存在
for _dir in [OUTPUT_DIR, MODEL_DIR, PRED_DIR, PREPROCESSOR_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ==================== 随机种子 ====================

RANDOM_SEED = 42

# ==================== 数据划分比例 ====================

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ==================== 列名定义 ====================

IDENTIFIER_COLS = ["user_id", "product_id"]
TARGET_COL = "label"
CATEGORICAL_COLS = ["department_id"]

# ==================== 模型超参数默认值 ====================

# Logistic Regression
LR_PARAMS = {
    "solver": "saga",
    "l1_ratio": 0,
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

# Logistic Regression 候选 C 值（用于手工微调）
LR_C_CANDIDATES = [0.1, 1.0, 10.0]

# Decision Tree
DT_PARAMS = {
    "criterion": "gini",
    "min_samples_leaf": 1000,
    "min_samples_split": 5000,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

# Decision Tree 候选 max_depth
DT_MAX_DEPTH_CANDIDATES = [8, 12, 15, None]

# Random Forest
RF_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 12,
    "min_samples_leaf": 1000,
    "min_samples_split": 5000,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "bootstrap": True,
}

# Random Forest 候选参数组合
RF_CANDIDATES = [
    {"n_estimators": 50, "max_depth": 8},
    {"n_estimators": 50, "max_depth": 12},
    {"n_estimators": 100, "max_depth": 12},
]

# XGBoost
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 15.2,  # neg/pos ≈ 15.2
    "tree_method": "hist",
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "random_state": RANDOM_SEED,
    "enable_categorical": True,
}

# XGBoost 候选参数组合
XGB_CANDIDATES = [
    {"max_depth": 4, "learning_rate": 0.1},
    {"max_depth": 6, "learning_rate": 0.1},
    {"max_depth": 6, "learning_rate": 0.05},
    {"max_depth": 8, "learning_rate": 0.1},
]

# LightGBM
LGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 7,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 15.2,  # neg/pos ≈ 15.2
    "boosting_type": "gbdt",
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "force_col_wise": True,
}

# LightGBM 候选参数组合
LGB_CANDIDATES = [
    {"num_leaves": 31, "max_depth": 5},
    {"num_leaves": 31, "max_depth": 7},
    {"num_leaves": 63, "max_depth": 7},
]
