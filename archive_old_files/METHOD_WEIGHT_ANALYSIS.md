# 如何科学地确定 Variable → OCEAN 的权重

## 问题陈述
我们有 10 个 categorical variables（grade, purpose, term 等），需要确定：
1. 每个 variable 对每个 OCEAN 维度的**权重系数**
2. 这些权重的**合理性**和**统计显著性**

---

## 🎯 方法 1: 监督学习（需要 Ground Truth OCEAN 标签）

### **Step 1: 获取 Ground Truth**
用 LLM 标注一个子集（500-1000 条）作为训练数据：

```python
# 用 LLM 打分 1000 条样本
scorer = SimplifiedOceanScorer(offline_mode=False)  # 启用 API
sample_df = df.sample(1000, random_state=42)
ground_truth_ocean = scorer.score_batch(sample_df)

# 保存为训练集
train_df = sample_df.copy()
for dim in OCEAN_DIMS:
    train_df[f'{dim}_truth'] = ground_truth_ocean[dim]
```

### **Step 2: One-Hot Encoding Variables**
把 categorical variables 转换为数值特征：

```python
from sklearn.preprocessing import OneHotEncoder

categorical_vars = ['grade', 'purpose', 'term', 'home_ownership', 'emp_length']
encoder = OneHotEncoder(sparse_output=False)
X = encoder.fit_transform(train_df[categorical_vars])

# 得到特征名
feature_names = encoder.get_feature_names_out()
# 例如: ['grade_A', 'grade_B', 'grade_C', ..., 'purpose_car', 'purpose_wedding', ...]
```

### **Step 3: 为每个 OCEAN 维度训练线性回归**
线性回归的系数就是权重！

```python
from sklearn.linear_model import Ridge

for dim in OCEAN_DIMS:
    y = train_df[f'{dim}_truth']

    # 用 Ridge 回归（带正则化，避免过拟合）
    model = Ridge(alpha=0.1)
    model.fit(X, y)

    # 提取系数（权重）
    weights = model.coef_

    # 显示每个特征的权重
    feature_weights = pd.DataFrame({
        'feature': feature_names,
        'weight': weights
    }).sort_values('weight', key=abs, ascending=False)

    print(f"\n=== {dim.upper()} 的权重 ===")
    print(feature_weights.head(20))  # Top 20 最重要的特征
```

**输出示例**：
```
=== CONSCIENTIOUSNESS 的权重 ===
feature                  weight
grade_A                 +0.287   # Grade A 对 Conscientiousness 强正向
grade_G                 -0.312   # Grade G 强负向
purpose_home_improvement +0.195
term_36 months          +0.142
home_ownership_OWN      +0.108
purpose_small_business  -0.056
...
```

### **优点**：
- ✅ 数据驱动，客观
- ✅ 自动发现权重
- ✅ 可以计算统计显著性（p-value）

### **缺点**：
- ⚠️ 需要 LLM 标注（成本 ~$3-5）
- ⚠️ 依赖 LLM 质量

---

## 🎯 方法 2: 无监督学习（无需 Ground Truth）

### **核心思路**：
用 **Variable 与 Default Rate 的关系** 反推 OCEAN 权重

**假设**：
- 如果某个 variable 的某个类别（如 Grade A）与低违约率强相关
- 那么它应该对 **Conscientiousness（负责任）** 和 **Neuroticism（情绪稳定）** 有正向影响

### **Step 1: 计算每个 Variable Category 的 Default Lift**
（Step 1.2 已完成）

```python
# 例如：
Grade A → Lift 0.47x（低风险）
Grade G → Lift 2.57x（高风险）
```

### **Step 2: 建立 Lift → OCEAN 的映射规则**

**心理学假设**：
```python
# Lift < 0.7  → 低风险 → 高 Conscientiousness, 低 Neuroticism
# Lift > 1.3  → 高风险 → 低 Conscientiousness, 高 Neuroticism

def lift_to_ocean_adjustment(lift):
    """
    根据 Lift 值推断 OCEAN 调整
    """
    adjustments = {}

    # Conscientiousness: Lift 低 → 分数高
    if lift < 0.7:
        adjustments['conscientiousness'] = +0.20 * (0.7 - lift)
    elif lift > 1.3:
        adjustments['conscientiousness'] = -0.15 * (lift - 1.3)

    # Neuroticism: Lift 高 → 分数高
    if lift < 0.7:
        adjustments['neuroticism'] = -0.15 * (0.7 - lift)
    elif lift > 1.3:
        adjustments['neuroticism'] = +0.20 * (lift - 1.3)

    return adjustments
```

**应用到每个 Variable**：
```python
# 读取 Step 1.2 的结果
results_df = pd.read_csv('artifacts/results/variable_analysis.csv')

# 为每个 category 生成 OCEAN 权重
for _, row in results_df.iterrows():
    var = row['variable']
    category = row['category']
    lift = row['lift']

    adjustments = lift_to_ocean_adjustment(lift)
    print(f"{var} = {category}: {adjustments}")
```

**输出示例**：
```
grade = A: {'conscientiousness': +0.106, 'neuroticism': -0.0795}
grade = G: {'conscientiousness': -0.2355, 'neuroticism': +0.314}
purpose = home_improvement: {'conscientiousness': +0.066, 'neuroticism': -0.0495}
...
```

### **优点**：
- ✅ 无需 LLM 标注
- ✅ 直接基于违约数据
- ✅ 成本为零

### **缺点**：
- ⚠️ 假设 Lift 与 OCEAN 有线性关系（可能不准确）
- ⚠️ 只能推断 C 和 N，其他维度（O, E, A）需要专家判断

---

## 🎯 方法 3: 混合方法（推荐）

### **Step 1: 用方法 2 自动生成 Baseline 权重**
基于 Lift 自动推断 Conscientiousness 和 Neuroticism

### **Step 2: 用心理学文献补充其他维度**
基于领域知识手动设计 Openness, Extraversion, Agreeableness

**例如**：
```python
# Purpose → Openness (创新性)
"small_business" → openness +0.25  # 创业需要创新
"wedding" → openness +0.10          # 传统事件，开放性中等
"renewable_energy" → openness +0.30 # 环保意识，高开放性

# Application Type → Extraversion (社交性)
"Joint" → extraversion +0.20        # 联合申请，社交导向
"Individual" → extraversion -0.10   # 独立申请，内向

# Purpose → Agreeableness (合作性)
"wedding" → agreeableness +0.20     # 婚礼是社交活动
"home_improvement" → agreeableness +0.10  # 改善家居，顾及他人
```

### **Step 3: 用方法 1 验证和微调**
用 500 条 LLM 标注样本验证权重的准确性，必要时微调

---

## 🎯 方法 4: 迭代优化（Grid Search）

### **思路**：
把权重当作**超参数**，通过 A/B 测试优化

### **步骤**：

**Step 1: 定义权重搜索空间**
```python
# 例如：Grade A 对 Conscientiousness 的权重在 [0.1, 0.2, 0.3, 0.4] 中搜索
weight_search_space = {
    'grade_A_conscientiousness': [0.1, 0.2, 0.3, 0.4],
    'grade_A_neuroticism': [-0.3, -0.2, -0.1, 0],
    'purpose_small_business_openness': [0.1, 0.2, 0.3],
    ...
}
```

**Step 2: 对每组权重，评估模型性能**
```python
from sklearn.model_selection import ParameterGrid

best_roc_auc = 0
best_weights = None

for params in ParameterGrid(weight_search_space):
    # 用这组权重生成 OCEAN 特征
    df_ocean = generate_ocean_with_weights(df, params)

    # 训练模型
    model = train_xgboost(df_ocean)

    # 评估
    roc_auc = evaluate_model(model, X_test, y_test)

    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_weights = params

print(f"最佳权重: {best_weights}")
print(f"最佳 ROC-AUC: {best_roc_auc}")
```

### **优点**：
- ✅ 直接优化最终目标（模型性能）
- ✅ 数据驱动

### **缺点**：
- ⚠️ 计算量大（如果搜索空间大）
- ⚠️ 可能过拟合测试集

---

## 📊 推荐方案（实际可行）

### **阶段 1: 快速验证（1 天）**
用 **方法 2（无监督 + Lift）** 快速生成权重

**代码**：
```python
python3 step2_generate_weights_from_lift.py  # 我会帮你写
```

**预期输出**：
- `ocean_weights_auto.json`（自动生成的权重文件）
- 基于 Lift 的 Conscientiousness 和 Neuroticism 权重

### **阶段 2: 补充其他维度（0.5 天）**
手动设计 O, E, A 的权重（基于常识）

### **阶段 3: 验证（0.5 天）**
用生成的权重跑 A/B 测试：
```python
python3 run_full_pipeline_with_rules.py  # 用规则而非哈希
```

查看 ROC-AUC 是否提升

### **阶段 4: 优化（可选，1-2 天）**
如果效果不好，启用 **方法 1（监督学习）**：
- 用 LLM 标注 500 条
- 训练 Ridge 回归学习权重
- 替换原有权重，重新测试

---

## 🚀 现在立即可以做的

**我现在可以帮你实现**：

### **Option A: 方法 2（无监督 Lift-Based）**
```python
# 自动从 variable_analysis.csv 生成权重
python3 step2_generate_weights_from_lift.py

# 重新跑 A/B 测试（用规则打分）
python3 run_full_pipeline_with_rules.py
```
**时间**：30 分钟实现 + 5 分钟运行

### **Option B: 方法 1（监督学习）**
```python
# 用 LLM 标注 500 条
python3 step2_label_ground_truth.py  # 需要 API key，成本 ~$1

# 训练线性回归学习权重
python3 step2_learn_weights.py

# 导出权重，重新测试
python3 run_full_pipeline_with_learned_weights.py
```
**时间**：1 小时实现 + 10 分钟运行

### **Option C: 方法 4（Grid Search）**
```python
# 定义搜索空间
python3 step2_grid_search_weights.py
```
**时间**：2 小时实现 + 30-60 分钟运行（计算密集）

---

## ❓ 你想要哪个？

**我的建议**：
1. **先做 Option A**（快速验证，无成本）
2. 如果 A 效果不好，再做 **Option B**（高质量，小成本）
3. 如果追求极致，最后做 **Option C**（最优化，计算量大）

**告诉我你的选择，我马上实现！**
