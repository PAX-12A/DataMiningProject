"""
数据加载与用户级 train/val/test 划分。

关键设计：按 user_id 划分（60/20/20），确保同一用户的所有样本
只出现在一个集合中，防止数据泄露。

首次运行：从 feature_table.csv 读取 -> 用户级划分 -> 保存到 data/splits/
后续运行：直接从 data/splits/ 加载（5 个模型共用同一份划分）
"""
import os
import numpy as np
import pandas as pd

from .config import (
    DATA_PATH,
    SPLIT_DIR,
    SPLIT_TRAIN_PATH,
    SPLIT_VAL_PATH,
    SPLIT_TEST_PATH,
    IDENTIFIER_COLS,
    TARGET_COL,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def load_and_split():
    """
    加载特征表，按用户级划分 train/val/test。

    首次调用时从 CSV 划分并保存结果到 data/splits/；
    后续调用时直接从 data/splits/ 加载。

    Returns
    -------
    train_data : DataFrame
    val_data : DataFrame
    test_data : DataFrame
    """
    # ---- 如果已有划分结果，直接加载 ----
    if (os.path.exists(SPLIT_TRAIN_PATH)
            and os.path.exists(SPLIT_VAL_PATH)
            and os.path.exists(SPLIT_TEST_PATH)):
        print("=" * 60)
        print("1. 从 data/splits/ 加载已有划分...")
        train_data = pd.read_csv(SPLIT_TRAIN_PATH)
        val_data = pd.read_csv(SPLIT_VAL_PATH)
        test_data = pd.read_csv(SPLIT_TEST_PATH)
        print(f"   Train: {len(train_data):,} 行")
        print(f"   Val:   {len(val_data):,} 行")
        print(f"   Test:  {len(test_data):,} 行")
        print(f"   reordered=1 比例: Train={train_data[TARGET_COL].mean():.4f}, "
              f"Val={val_data[TARGET_COL].mean():.4f}, "
              f"Test={test_data[TARGET_COL].mean():.4f}")
        return train_data, val_data, test_data

    # ---- 首次运行：从原始数据划分并保存 ----
    print("=" * 60)
    print("1. 加载特征表（首次划分）...")
    df = pd.read_csv(DATA_PATH)
    print(f"   总行数: {len(df):,}")
    print(f"   总用户数: {df['user_id'].nunique():,}")
    print(f"   reordered=1 比例: {df[TARGET_COL].mean():.4f}")

    # ---- 用户级划分 ----
    print("\n2. 用户级划分 (60/20/20)...")
    unique_users = df["user_id"].unique()
    n_users = len(unique_users)
    print(f"   唯一用户数: {n_users:,}")

    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(unique_users)

    n_train_users = int(n_users * TRAIN_RATIO)
    n_val_users = int(n_users * VAL_RATIO)

    train_users = set(unique_users[:n_train_users])
    val_users = set(unique_users[n_train_users:n_train_users + n_val_users])
    test_users = set(unique_users[n_train_users + n_val_users:])

    # 验证无交集
    assert train_users & val_users == set(), "Train 和 Val 的 user_id 有交集!"
    assert train_users & test_users == set(), "Train 和 Test 的 user_id 有交集!"
    assert val_users & test_users == set(), "Val 和 Test 的 user_id 有交集!"
    print("   OK Train/Val/Test 的 user_id 互不相交")

    # 筛选
    train_data = df[df["user_id"].isin(train_users)].copy()
    val_data = df[df["user_id"].isin(val_users)].copy()
    test_data = df[df["user_id"].isin(test_users)].copy()

    print(f"   Train: {len(train_data):,} 行, {len(train_users):,} 用户")
    print(f"   Val:   {len(val_data):,} 行, {len(val_users):,} 用户")
    print(f"   Test:  {len(test_data):,} 行, {len(test_users):,} 用户")

    # ---- 保存到 data/splits/ ----
    print("\n3. 保存划分结果到 data/splits/ ...")
    os.makedirs(SPLIT_DIR, exist_ok=True)

    train_data.to_csv(SPLIT_TRAIN_PATH, index=False)
    print(f"   -> {SPLIT_TRAIN_PATH}")

    val_data.to_csv(SPLIT_VAL_PATH, index=False)
    print(f"   -> {SPLIT_VAL_PATH}")

    test_data.to_csv(SPLIT_TEST_PATH, index=False)
    print(f"   -> {SPLIT_TEST_PATH}")
    print("   OK 划分已保存，后续模型将直接加载")

    # 释放原始 df
    del df

    return train_data, val_data, test_data


def extract_xy(data: pd.DataFrame):
    """
    从 DataFrame 中提取特征 X、目标 y 和标识列。

    Parameters
    ----------
    data : DataFrame
        包含 user_id, product_id, reordered 及特征列的数据

    Returns
    -------
    X : DataFrame (仅特征列)
    y : Series (目标列)
    ids : DataFrame (user_id, product_id)
    """
    feature_cols = [
        c
        for c in data.columns
        if c not in IDENTIFIER_COLS + [TARGET_COL]
    ]
    X = data[feature_cols]
    y = data[TARGET_COL]
    ids = data[IDENTIFIER_COLS]
    return X, y, ids
