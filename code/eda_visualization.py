import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

os.makedirs(
    "figure",
    exist_ok=True
)


orders = pd.read_csv("data/orders.csv")#order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
products = pd.read_csv("data/products.csv")#product_id,product_name,aisle_id,department_id
# aisles = pd.read_csv("data/aisles.csv")#aisle_id,aisle
departments = pd.read_csv("data/departments.csv")#department_id,department

prior = pd.read_csv("data/order_products__prior.csv")
# train = pd.read_csv("data/order_products__train.csv") # order_id,product_id,add_to_cart_order,reordered

sns.set_theme(
    style="whitegrid",
    context="talk"
)

plt.style.use("ggplot")

#1 Distribution of Number of Items per Order

order_size = (
    prior.groupby("order_id")
    .size()
)

# sns.histplot(
#     order_size,
#     bins=30
# )

# plt.xlim(0,60)

# plt.xlabel("Items per Order")
# plt.ylabel("Frequency")
# plt.title(
#     "Distribution of Number of Items per Order"
# )

# plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(order_size, bins=50, kde=False)
plt.title("Distribution of Number of Items per Order")
plt.xlabel("Basket size (items per order)")
plt.ylabel("Number of orders")
plt.xlim(0, order_size.quantile(0.99))

plt.savefig(
    "figure/Number_of_Items_per_Order.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


#2 Top 20 Most Frequently Purchased Products

top_products = (
    prior["product_id"]
    .value_counts()
    .head(20)
)

top_products = (
    top_products
    .reset_index()
)

top_products.columns = [
    "product_id",
    "count"
]

top_products = top_products.merge(
    products[
        ["product_id",
         "product_name"]
    ],
    on="product_id"
)

sns.barplot(
    data=top_products,
    x="count",
    y="product_name",
    palette="viridis"
)

plt.title(
    "Top 20 Most Frequently Purchased Products"
)

plt.savefig(
    "figure/Most_Frequently_Purchased_Products.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#3 Department Reorder Rate

prior_orders = prior.merge(
    products,
    on="product_id",
    how="left"
)

prior_orders = prior_orders.merge(
    departments,
    on="department_id",
    how="left"
)

print(prior_orders.columns)

dept_rate = (
    prior_orders
    .groupby("department")
    ["reordered"]
    .mean()
    .sort_values(
        ascending=False
    )
)

dept_rate = (
    dept_rate
    .reset_index()
)

top_dept = (
    dept_rate
    .head(5)
)

bottom_dept = (
    dept_rate
    .tail(5)
)

dept_compare = pd.concat([
    dept_rate.head(5),
    dept_rate.tail(5)
])

sns.barplot(
    data=dept_compare,
    x="reordered",
    y="department",
    palette="viridis"
)

plt.title(
    "Top and Bottom Departments by Reorder Rate"
)

plt.xlabel("Department_Reorder_Rate")

plt.savefig(
    "figure/Reorder Rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#用户订单间隔

# sns.countplot(
#     x="days_since_prior_order",
#     data=orders,
#     palette="viridis"
# )

# plt.show()

days_count = (
    orders["days_since_prior_order"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10,5))

plt.bar(
    days_count.index,
    days_count.values,
    width=1.0,
    color="skyblue"
)

plt.xticks(range(0,31,2))

plt.title(
    "Distribution of Days Since Prior Order"
)

plt.xlabel(
    "Days Since Prior Order"
)

plt.ylabel(
    "Number of Orders"
)

plt.savefig(
    "figure/Days_Since_Prior_Order.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#商品复购率分布

product_features = pd.DataFrame()

product_features["product_reorder_rate"] = (
    prior_orders.groupby("product_id")["reordered"]
    .mean()
)

sns.histplot(
    product_features[
        product_features[
            "product_reorder_rate"
        ] > 0
    ]
)

plt.savefig(
    "figure/product_total_orders.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()