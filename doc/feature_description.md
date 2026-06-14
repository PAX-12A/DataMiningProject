# Feature Description Document

## Label（预测目标）

| 特征名    | 含义                          | 业务解释                                               |
| --------- | ----------------------------- | ------------------------------------------------------ |
| reordered | 用户是否再次购买该商品（0/1） | 预测目标。1表示用户在当前订单中复购该商品，0表示未复购 |

------

## User Features（用户特征）

用于描述用户整体购物习惯和消费行为。

| 特征名                       | 含义                     | 业务解释                                   |
| ---------------------------- | ------------------------ | ------------------------------------------ |
| user_total_orders            | 用户历史订单总数         | 用户活跃程度，订单越多说明平台使用频率越高 |
| user_avg_days_between_orders | 用户平均下单间隔天数     | 衡量购物周期，高频用户通常更容易产生复购   |
| user_total_products          | 用户历史购买商品总数     | 衡量总体消费规模                           |
| user_avg_cart_size           | 用户平均每单购买商品数   | 衡量购物篮大小和消费能力                   |
| user_unique_products         | 用户购买过的不同商品数量 | 衡量消费多样性                             |
| user_reorder_ratio           | 用户历史复购比例         | 衡量用户忠诚度和重复购买倾向               |

------

## Product Features（商品特征）

用于描述商品热度和市场表现。

| 特征名                    | 含义                   | 业务解释                                 |
| ------------------------- | ---------------------- | ---------------------------------------- |
| product_total_orders      | 商品历史被购买次数     | 商品销量指标                             |
| product_unique_users      | 购买该商品的用户数     | 商品覆盖用户范围                         |
| product_reorder_rate      | 商品历史复购率         | 商品粘性，高值商品通常属于日常消耗品     |
| product_avg_cart_position | 商品平均加入购物车顺序 | 数值越小，说明商品越重要、购买优先级越高 |

------

## Department Features（部门特征）

用于描述商品所属大类的整体消费特征。

| 特征名                  | 含义             | 业务解释                                           |
| ----------------------- | ---------------- | -------------------------------------------------- |
| department_id           | 商品所属部门编号 | 商品所属一级分类                                   |
| department_reorder_rate | 部门整体复购率   | 衡量该类商品整体复购倾向，例如牛奶水果通常高于零食 |

------

## User-Product Interaction Features（用户-商品交互特征）

该类特征通常最重要，直接反映用户与商品之间的关系。

| 特征名               | 含义                               | 业务解释                                   |
| -------------------- | ---------------------------------- | ------------------------------------------ |
| up_order_count       | 用户历史购买该商品次数             | 衡量用户对商品的偏好强度                   |
| up_last_order        | 最近一次购买该商品发生在哪个订单   | 衡量购买行为的新鲜度                       |
| up_order_rate        | 用户订单中该商品出现的比例         | 衡量商品在用户购物习惯中的重要程度         |
| up_orders_since_last | 距离最近一次购买已经过去多少单     | 反映用户是否到了再次购买周期               |
| up_order_span        | 从首次购买到最近购买经历的订单跨度 | 衡量商品陪伴用户时间长短，反映长期消费习惯 |

------

## 特征类型说明

| 特征名                       | 类型       |
| ---------------------------- | ---------- |
| reordered                    | Label      |
| user_total_orders            | Count      |
| user_avg_days_between_orders | Continuous |
| user_total_products          | Count      |
| user_avg_cart_size           | Continuous |
| user_unique_products         | Count      |
| user_reorder_ratio           | Ratio      |
| product_total_orders         | Count      |
| product_unique_users         | Count      |
| product_reorder_rate         | Ratio      |
| product_avg_cart_position    | Continuous |
| department_id                | Category   |
| department_reorder_rate      | Ratio      |
| up_order_count               | Count      |
| up_last_order                | Count      |
| up_order_rate                | Ratio      |
| up_orders_since_last         | Count      |
| up_order_span                | Count      |

------

## 总结

| 特征组                | 数量 |
| --------------------- | ---- |
| User Features         | 6    |
| Product Features      | 4    |
| Department Features   | 2    |
| User-Product Features | 5    |
| Label                 | 1    |

总计：

```text
17个输入特征
1个预测标签
```

