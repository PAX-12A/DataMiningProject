"""
预处理模块 - Logistic Regression 专用。

树模型（DT / RF / XGBoost / LightGBM）不需要标准化和 One-Hot，
直接使用原始特征即可。LightGBM/XGBoost 可通过 categorical_feature
参数指定 department_id 为类别特征。
"""
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .config import CATEGORICAL_COLS, PREPROCESSOR_DIR, RANDOM_SEED


def build_lr_preprocessor(X):
    """
    为 Logistic Regression 构建预处理流水线。

    - 数值特征：StandardScaler（17 列）
    - 类别特征（department_id）：OneHotEncoder，sparse_output=True 节省内存

    Returns 拟合好的 ColumnTransformer。
    """
    # 分类出数值列和类别列
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
    categorical_cols = [c for c in X.columns if c in CATEGORICAL_COLS]

    print(f"   数值特征: {len(numeric_cols)} 列")
    print(f"   类别特征: {len(categorical_cols)} 列 -> One-Hot")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            (
                "categorical",
                OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore"
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def fit_preprocessor(preprocessor, X_train):
    """拟合预处理器并保存。"""
    print("   拟合 ColumnTransformer...")
    preprocessor.fit(X_train)

    # 保存预处理器
    path = os.path.join(PREPROCESSOR_DIR, "lr_preprocessor.pkl")
    joblib.dump(preprocessor, path)
    print(f"   预处理器已保存到: {path}")

    return preprocessor


def load_preprocessor():
    """加载已保存的 LR 预处理器。"""
    path = os.path.join(PREPROCESSOR_DIR, "lr_preprocessor.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"预处理器文件不存在: {path}")
    return joblib.load(path)
