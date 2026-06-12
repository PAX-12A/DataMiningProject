import pandas as pd
import numpy as np

# ==================================
# 1. 读取数据
# ==================================

orders = pd.read_csv("data/orders.csv")#order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
products = pd.read_csv("data/products.csv")#product_id,product_name,aisle_id,department_id
aisles = pd.read_csv("data/aisles.csv")#aisle_id,aisle
departments = pd.read_csv("data/departments.csv")#department_id,department

prior = pd.read_csv("data/order_products__prior.csv")
train = pd.read_csv("data/order_products__train.csv") # order_id,product_id,add_to_cart_order,reordered

sample_users = (
    orders["user_id"]
    .drop_duplicates()
    .sample(
        n=50000,
        random_state=42
    )
)

# sample_users = (
#     orders["user_id"]
#     .drop_duplicates()
# )

orders = orders[orders["user_id"].isin(sample_users)]

sample_order_ids = set(orders["order_id"])

prior = prior[prior["order_id"].isin(sample_order_ids)]

print("Data Loaded")


# ==================================
# 2. 构造历史行为表
# ==================================

prior_orders = prior.merge(
    orders,
    on="order_id",
    how="inner"
)

prior_orders = prior_orders.merge(
    products,
    on="product_id",
    how="left"
)

prior_only_orders = orders[
    orders["eval_set"] == "prior"
]

print(prior_orders.head())
print(prior_orders.shape)
print(prior_orders.columns)

# ==================================
# 3. 用户特征
# ==================================

user_features = pd.DataFrame()

user_features["user_total_orders"] = (prior_only_orders.groupby("user_id")["order_number"].max()) # 只包含"prior"订单不含train

user_features["user_avg_days_between_orders"] = (prior_only_orders.groupby("user_id")["days_since_prior_order"].mean())

user_features["user_total_products"] = (prior_orders.groupby("user_id")["product_id"].count())

user_features["user_avg_cart_size"] = (user_features["user_total_products"]/user_features["user_total_orders"])

user_features["user_unique_products"] = (prior_orders.groupby("user_id")["product_id"].nunique())

user_features["user_reorder_ratio"] = (prior_orders.groupby("user_id")["reordered"].mean())

user_features = user_features.reset_index()

print("User Features Created")


# ==================================
# 4. 商品特征
# ==================================

product_features = pd.DataFrame()

product_features["product_total_orders"] = (prior_orders.groupby("product_id").size())

product_features["product_unique_users"] = (prior_orders.groupby("product_id")["user_id"].nunique())

product_features["product_reorder_rate"] = (prior_orders.groupby("product_id")["reordered"].mean())

product_features["product_avg_cart_position"] = (prior_orders.groupby("product_id")["add_to_cart_order"].mean())

product_features = product_features.reset_index()

print("Product Features Created")


# ==================================
# 5. 用户-商品交互特征
# ==================================

up = prior_orders.groupby(
    ["user_id", "product_id"]
)

up_features = pd.DataFrame()

# 用户购买该商品次数
up_features["up_order_count"] = up.size()

# 首次购买订单
up_features["up_first_order"] = (
    up["order_number"].min()
)

# 最近一次购买订单
up_features["up_last_order"] = (
    up["order_number"].max()
)

up_features = up_features.reset_index()

# 合并用户总订单数
up_features = up_features.merge(
    user_features[
        ["user_id", "user_total_orders"]
    ],
    on="user_id",
    how="left"
)

# 商品出现比例
up_features["up_order_rate"] = (
    up_features["up_order_count"]
    /
    up_features["user_total_orders"]
)

# 距离最近一次购买已经过去多少单
up_features["up_orders_since_last"] = (
    up_features["user_total_orders"]
    -
    up_features["up_last_order"]
)

# 商品陪伴用户的订单跨度
up_features["up_order_span"] = (
    up_features["up_last_order"]
    -
    up_features["up_first_order"]
)

# 删除辅助字段
up_features.drop(
    columns=[
        "user_total_orders",
        "up_first_order"
    ],
    inplace=True
)

print("User Product Features Created")

# ==================================
# 6. Department 特征
# ==================================

products = products.merge(
    departments,
    on="department_id",
    how="left"
)

dept_features = (
    prior_orders
    .groupby("department_id")
    .agg({
        "reordered":"mean"
    })
    .rename(
        columns={
            "reordered":
            "department_reorder_rate"
        }
    )
    .reset_index()
)

print("Department Features Created")


# ==================================
# 6. 构造训练标签（Label）
# ==================================

# 研究目标：
# 利用用户历史订单（prior）中的行为，
# 预测用户在下一次订单（train）中
# 是否会再次购买某个商品。
#
# Label定义：
# label = 1 ：该商品出现在用户的train订单中
# label = 0 ：该商品仅出现在历史订单中，未出现在train订单中
#
# 注意：
# 不使用 train 表中的 reordered 字段作为标签。
# reordered 表示：
#   “该商品在当前订单中是否曾经购买过”
#
# 而本项目需要预测的是：
#   “用户下一单是否会购买该商品”
#
# 两者含义不同。


# ----------------------------------
# Step 1
# 仅保留当前采样用户对应的train订单
# ----------------------------------

train = train[
    train["order_id"].isin(sample_order_ids)
]


# ----------------------------------
# Step 2
# 获取所有train订单信息
# ----------------------------------

train_orders = orders[
    orders["eval_set"] == "train"
]


# ----------------------------------
# Step 3
# 构造候选样本集合
# ----------------------------------
#
# 候选样本 = 用户历史买过的所有商品
#
# 例如：
#
# 用户A历史买过：
# 香蕉、牛奶、鸡蛋
#
# 则产生：
#
# (A, 香蕉)
# (A, 牛奶)
# (A, 鸡蛋)
#
# 后续再判断这些商品是否出现在train订单中
#

prior_pairs = (
    prior_orders[
        ["user_id", "product_id"]
    ]
    .drop_duplicates()
)


# ----------------------------------
# Step 4
# 获取train订单中的真实购买商品
# ----------------------------------
#
# train表只有：
# order_id
# product_id
#
# 需要通过orders表补充user_id
#

train_pairs = (
    train
    .merge(
        train_orders[
            ["order_id", "user_id"]
        ],
        on="order_id",
        how="inner"
    )
    [
        ["user_id", "product_id"]
    ]
)


# ----------------------------------
# Step 5
# 初始化标签
# ----------------------------------
#
# 默认认为：
# 历史商品不会被再次购买
#

prior_pairs["label"] = 0


# ----------------------------------
# Step 6
# 标记正样本
# ----------------------------------
#
# 如果：
#
# (user_id, product_id)
#
# 同时出现在train订单中，
# 则说明用户下一单再次购买了该商品。
#
# 标记：
#
# label = 1
#

prior_pairs.loc[
    prior_pairs
    .set_index(
        ["user_id", "product_id"]
    )
    .index
    .isin(
        train_pairs
        .set_index(
            ["user_id", "product_id"]
        )
        .index
    ),
    "label"
] = 1


# ----------------------------------
# 查看标签分布
# ----------------------------------

print(
    prior_pairs["label"]
    .value_counts()
)

print("Labels Created")

# ==================================
# 7. 合并特征
# ==================================

feature_table = prior_pairs.merge(
    user_features,
    on="user_id",
    how="left"
)

feature_table = feature_table.merge(
    product_features,
    on="product_id",
    how="left"
)

feature_table = feature_table.merge(
    up_features,
    on=["user_id", "product_id"],
    how="left"
)

feature_table = (
    feature_table
    .merge(
        products[
            [
                "product_id",
                "department_id"
            ]
        ],
        on="product_id",
        how="left"
    )
)

feature_table = (
    feature_table
    .merge(
        dept_features,
        on="department_id",
        how="left"
    )
)

feature_table.fillna(0, inplace=True)

print(feature_table.head())

# ==================================
# 8. 数据类型优化
# ==================================

int_cols = [
    "user_id",
    "product_id",
    "reordered",
    "user_total_orders",
    "user_total_products",
    "user_unique_products",
    "product_total_orders",
    "product_unique_users",
    "up_order_count",
    "up_first_order",
    "up_last_order",
    "up_orders_since_last"
]

for col in int_cols:
    if col in feature_table.columns:
        feature_table[col] = (
            feature_table[col]
            .fillna(0)
            .astype("int32")
        )

# 比例类

ratio_cols = [
    "user_reorder_ratio",
    "product_reorder_rate",
    "department_reorder_rate",
    "up_order_rate"
]

for col in ratio_cols:
    if col in feature_table.columns:
        feature_table[col] = feature_table[col].round(4)

# 平均值类

mean_cols = [
    "user_avg_days_between_orders",
    "user_avg_cart_size",
    "product_avg_cart_position"
]

for col in mean_cols:
    if col in feature_table.columns:
        feature_table[col] = feature_table[col].round(2)


# ==================================
# 8. 保存结果
# ==================================

# print(feature_table["reordered"].value_counts())

feature_table.to_csv(
    "output/feature_table.csv",
    index=False
)

print("Feature Table Saved")