# Supervised Learning 完整指南
## 如何用监督学习计算 Variable → OCEAN 的权重

---

## 📚 核心思路

**问题**: 如何确定 `grade`, `purpose`, `term` 等变量对 OCEAN 各维度的权重？

**解决方案**: 用**线性回归**学习权重！

```
OCEAN_score = w1*feature1 + w2*feature2 + w3*feature3 + ... + bias
               ↑            ↑            ↑
          这些就是权重！(我们要学习的)
```

**为什么有效？**
- 如果 `grade_A` 与高 `Conscientiousness` 相关 → 回归会给它正权重
- 如果 `grade_G` 与低 `Conscientiousness` 相关 → 回归会给它负权重

---

## 🔬 完整流程（3 个步骤）

### **Phase 1: 获取 Ground Truth（真实标签）**

**文件**: `supervised_step1_label_ground_truth.py`

**做什么**:
1. 从数据集随机抽取 **500-1000 条样本**
2. 用 **OpenAI API** (gpt-4o-mini) 给每条样本打 OCEAN 分数
3. 保存为 `ground_truth_ocean.csv`

**示例输出**:
```
grade | purpose           | term      | ... | openness_truth | conscientiousness_truth | ...
------|-------------------|-----------|-----|----------------|------------------------|----
A     | home_improvement  | 36 months | ... | 0.65           | 0.72                   | ...
G     | small_business    | 60 months | ... | 0.75           | 0.28                   | ...
C     | debt_consolidation| 36 months | ... | 0.50           | 0.60                   | ...
```

**运行**:
```bash
# 需要设置 API key
export OPENAI_API_KEY="sk-..."

python3 supervised_step1_label_ground_truth.py
```

**成本**: ~$1-3 (取决于样本数量)
**时间**: 10-20 分钟

---

### **Phase 2: One-Hot Encoding + 训练回归模型**

**文件**: `supervised_step2_learn_weights.py`

**做什么**:

#### **Step 2.1: One-Hot Encoding**
把 categorical variables 转换为数值特征：

```python
原始数据:
  grade = 'A'

One-Hot 编码后:
  grade_A = 1
  grade_B = 0
  grade_C = 0
  ...
  grade_G = 0
```

对所有变量做 One-Hot：
```
grade_A, grade_B, ..., grade_G,
purpose_car, purpose_wedding, ..., purpose_small_business,
term_36 months, term_60 months,
home_ownership_OWN, home_ownership_RENT, ...
```

总共约 **50-80 个特征**

#### **Step 2.2: 训练线性回归**

对**每个 OCEAN 维度**单独训练一个回归模型：

```python
from sklearn.linear_model import Ridge

# 对 Conscientiousness
X = one_hot_encoded_features  # (500, 70) - 500 样本, 70 特征
y = ground_truth['conscientiousness_truth']  # (500,)

model = Ridge(alpha=0.1)  # L2 正则化，防止过拟合
model.fit(X, y)

# 提取权重
weights = model.coef_  # shape: (70,) - 每个特征一个权重
intercept = model.intercept_  # bias term
```

**预测公式**:
```
conscientiousness = intercept +
                   weights[0] * grade_A +
                   weights[1] * grade_B +
                   ... +
                   weights[50] * purpose_wedding +
                   ...
```

#### **Step 2.3: 提取权重**

**示例输出** (Conscientiousness):
```
Top 10 权重:
  grade_A                    : +0.2873 ↑  (A级 → 高责任心)
  grade_G                    : -0.3124 ↓  (G级 → 低责任心)
  purpose_home_improvement   : +0.1950 ↑  (家居改善 → 负责任)
  term_36 months             : +0.1420 ↑  (短期 → 谨慎)
  home_ownership_OWN         : +0.1080 ↑  (房主 → 稳定)
  purpose_small_business     : -0.0560 ↓  (创业 → 可能不谨慎?)
  verification_status_Verified: +0.0720 ↑  (验证 → 透明)
  ...
```

**保存结果**:
- `learned_weights.json` - 所有权重系数
- `onehot_encoder.joblib` - Encoder (用于新数据)

**运行**:
```bash
python3 supervised_step2_learn_weights.py
```

**输出**: `learned_weights.json`
```json
{
  "conscientiousness": {
    "intercept": 0.5234,
    "features": {
      "grade_A": 0.2873,
      "grade_G": -0.3124,
      "purpose_home_improvement": 0.1950,
      ...
    }
  },
  "openness": {
    "intercept": 0.4987,
    "features": {
      "purpose_small_business": 0.2510,
      "purpose_renewable_energy": 0.3021,
      ...
    }
  },
  ...
}
```

---

### **Phase 3: 应用权重 + A/B 测试**

**文件**: `supervised_step3_apply_weights.py`

**做什么**:

#### **Step 3.1: 批量打分**

用学习到的权重为新数据打分：

```python
# 加载权重
with open('learned_weights.json') as f:
    weights = json.load(f)

encoder = joblib.load('onehot_encoder.joblib')

# 新数据
new_borrower = {
    'grade': 'B',
    'purpose': 'debt_consolidation',
    'term': '36 months',
    ...
}

# One-Hot 编码
features_encoded = encoder.transform([new_borrower])

# 计算 OCEAN
ocean_scores = {}
for dim in ['openness', 'conscientiousness', ...]:
    score = weights[dim]['intercept']

    for feat_name, feat_value in zip(feature_names, features_encoded[0]):
        if feat_name in weights[dim]['features']:
            score += weights[dim]['features'][feat_name] * feat_value

    ocean_scores[dim] = max(0.25, min(0.75, score))  # 裁剪范围
```

#### **Step 3.2: A/B 对比**

- **Model A**: Baseline (不含 OCEAN)
- **Model B**: Baseline + Learned OCEAN

**运行**:
```bash
python3 supervised_step3_apply_weights.py
```

**输出**:
```
【A/B 对比结果】
Metric          A (Baseline)         B (Learned OCEAN)    Delta
----------------------------------------------------------------------
ROC-AUC         0.6830               0.6952               +0.0122
PR-AUC          0.2644               0.2711               +0.0067

🎉 成功！ROC-AUC 提升 0.0122 >= +0.010
```

---

## 🎯 为什么 Supervised Learning 有效？

### **优势**:

1. **数据驱动**: 权重不是主观拍脑袋，而是从数据学习
2. **可解释性**: 每个权重有明确含义（某个特征对 OCEAN 的影响）
3. **自动发现模式**: 可能发现你意想不到的关系
4. **统计显著性**: 可以计算 p-value 判断权重是否可靠

### **与 Rule-Based 对比**:

| 方法 | 权重来源 | 准确性 | 可解释性 | 成本 |
|------|---------|--------|---------|------|
| **Rule-Based** | 专家经验/心理学文献 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $0 |
| **Supervised Learning** | 从 LLM 标注数据学习 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $1-3 |

---

## 📊 数学原理（简化版）

### **问题**:
已知 N 个样本的特征和 OCEAN 分数，求权重 w

```
样本 1: [grade_A=1, grade_B=0, ...] → conscientiousness_truth = 0.72
样本 2: [grade_A=0, grade_B=1, ...] → conscientiousness_truth = 0.55
...
样本 N: [grade_A=0, grade_G=1, ...] → conscientiousness_truth = 0.28
```

### **目标**:
找到一组权重 w，使得预测值接近真实值：

```
min Σ (predicted - truth)²
    i

其中 predicted = w1*feature1 + w2*feature2 + ... + bias
```

### **解法**:
用 **Ridge 回归** (带 L2 正则化的线性回归):

```python
model = Ridge(alpha=0.1)
model.fit(X, y)

# 最优权重
w_optimal = model.coef_
```

**正则化的作用**:
- 防止某些权重过大（过拟合）
- 鼓励权重稀疏（只保留重要特征）

---

## 🚀 快速开始（完整流程）

### **需求**:
- Python 3.8+
- OpenAI API Key ($3 预算)
- 时间: 30-40 分钟

### **运行步骤**:

```bash
# Step 1: 设置 API Key
export OPENAI_API_KEY="sk-..."

# Step 2: 标注 Ground Truth (10-20 分钟)
python3 supervised_step1_label_ground_truth.py
# 输出: artifacts/results/ground_truth_ocean.csv

# Step 3: 学习权重 (2-3 分钟)
python3 supervised_step2_learn_weights.py
# 输出: artifacts/results/learned_weights.json

# Step 4: 应用权重 + A/B 测试 (3-5 分钟)
python3 supervised_step3_apply_weights.py
# 输出: artifacts/results/supervised_ab_results.csv
```

### **查看结果**:
```bash
# 权重文件
cat artifacts/results/learned_weights.json | jq '.conscientiousness.features | to_entries | sort_by(.value) | reverse | .[0:10]'

# A/B 对比
cat artifacts/results/supervised_ab_results.csv
```

---

## 🔍 常见问题

### **Q1: 为什么用 Ridge 而不是普通 Lasso 或 Lasso？**

| 方法 | 正则化 | 特点 | 适用场景 |
|------|--------|------|---------|
| OLS (普通最小二乘) | 无 | 可能过拟合 | 样本量 >> 特征数 |
| **Ridge (L2)** | ‖w‖² | 所有权重都有值，但较小 | 特征相关性高 |
| Lasso (L1) | ‖w‖ | 部分权重为 0（特征选择） | 需要稀疏解 |

我们用 Ridge 因为：
- 特征间有相关性（grade 和 sub_grade）
- 希望保留所有特征的信息

### **Q2: 500 条样本够吗？**

**经验法则**: 样本数 ≥ 5-10 倍特征数

```
特征数: ~70 (One-Hot 后)
最少样本: 70 × 5 = 350
推荐样本: 70 × 10 = 700
```

500 条已经足够，1000 条更稳定。

### **Q3: 如果 Ground Truth 质量不好怎么办？**

**诊断方法**:
```python
# 检查 R² (越接近 1 越好)
print(f"R² = {model.score(X, y):.3f}")

# 如果 R² < 0.5，说明 Ground Truth 噪音大
```

**解决方法**:
1. 增加样本量（500 → 1000）
2. 用更好的 LLM（gpt-4o-mini → gpt-4o）
3. 多次标注取平均（每条样本打 3 次分，取 median）

### **Q4: 怎么判断权重是否合理？**

**方法 1: 心理学检查**
```
grade_A → conscientiousness 应该是 +
grade_G → conscientiousness 应该是 -
purpose_small_business → openness 应该是 +
```

**方法 2: 统计检查**
```python
# 计算权重的置信区间（Bootstrap）
from sklearn.utils import resample

bootstrap_weights = []
for i in range(100):
    X_boot, y_boot = resample(X, y)
    model.fit(X_boot, y_boot)
    bootstrap_weights.append(model.coef_)

# 如果 95% CI 包含 0 → 权重不显著
```

---

## 📈 预期效果

基于类似研究 (Yu et al. 2023):
- ROC-AUC 提升: **+0.01 ~ +0.03**
- PR-AUC 提升: **+0.008 ~ +0.02**

你的结果可能更低（因为文本有限），但方向应该是正确的。

---

## 💡 进阶优化

如果初步结果不理想：

### **1. 增加样本量**
```python
SAMPLE_SIZE = 1000  # 从 500 改到 1000
```

### **2. 调整正则化强度**
```python
model = Ridge(alpha=0.01)  # 从 0.1 改到 0.01 (更灵活)
# 或
model = Ridge(alpha=0.5)   # 从 0.1 改到 0.5 (更保守)
```

### **3. 特征工程**
```python
# 添加交互特征
grade_purpose = df['grade'] + '_' + df['purpose']
# 例如: "A_home_improvement", "G_small_business"
```

### **4. 非线性模型**
```python
from sklearn.ensemble import RandomForestRegressor

# 用随机森林代替线性回归
model = RandomForestRegressor(n_estimators=100)
```

---

## ✅ 总结

**核心流程**:
```
LLM 标注 → One-Hot Encoding → Ridge 回归 → 提取权重 → 应用打分
```

**关键输出**:
- `learned_weights.json`: 每个特征对每个 OCEAN 维度的权重

**优势**:
- 数据驱动、可解释、可验证

**成本**:
- $1-3 + 30-40 分钟

**现在就可以运行！** 🚀
