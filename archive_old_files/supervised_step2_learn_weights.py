"""
Supervised Learning - Phase 2: 学习权重系数

核心思路:
1. 把 categorical variables 转换为 One-Hot 特征
2. 用线性回归: OCEAN_score = w1*feature1 + w2*feature2 + ... + bias
3. 回归系数 w1, w2, ... 就是权重！

输入: ground_truth_ocean.csv (Phase 1 的输出)
输出: learned_weights.json (每个 variable 对每个 OCEAN 维度的权重)
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("Supervised Learning - Phase 2: 学习权重系数")
print("=" * 70)

# ========== 加载 Ground Truth ==========
print(f"\n【Step 1】加载 Ground Truth")
print("-" * 70)

ground_truth_path = 'artifacts/results/ground_truth_ocean.csv'

try:
    df = pd.read_csv(ground_truth_path)
    print(f"✅ 加载成功: {len(df)} 条样本")
except FileNotFoundError:
    print(f"❌ 错误: 找不到 {ground_truth_path}")
    print(f"请先运行: python3 supervised_step1_label_ground_truth.py")
    exit(1)

OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# ========== One-Hot Encoding ==========
print(f"\n【Step 2】One-Hot Encoding Categorical Variables")
print("-" * 70)

# 选择要编码的变量
categorical_vars = ['grade', 'purpose', 'term', 'home_ownership', 'emp_length', 'verification_status']
categorical_vars = [v for v in categorical_vars if v in df.columns]

print(f"编码变量: {categorical_vars}")

# 准备数据
X_cat = df[categorical_vars].copy()

# 填充缺失值
for col in categorical_vars:
    X_cat[col] = X_cat[col].fillna('MISSING')

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_cat)

# 获取特征名
feature_names = encoder.get_feature_names_out(categorical_vars)

print(f"✅ 编码完成: {X_encoded.shape[1]} 个特征")
print(f"\n特征示例:")
for i, name in enumerate(feature_names[:10]):
    print(f"  {i+1}. {name}")
print(f"  ...")

# ========== 为每个 OCEAN 维度训练回归模型 ==========
print(f"\n【Step 3】训练线性回归模型学习权重")
print("=" * 70)

learned_weights = {}
models = {}

for dim in OCEAN_DIMS:
    print(f"\n【{dim.upper()}】")
    print("-" * 70)

    # 准备目标变量
    y = df[f'{dim}_truth'].values

    # 训练 Ridge 回归（带 L2 正则化，防止过拟合）
    model = Ridge(alpha=0.1, random_state=42)
    model.fit(X_encoded, y)

    # 5 折交叉验证
    cv_scores = cross_val_score(model, X_encoded, y, cv=5, scoring='r2')

    print(f"交叉验证 R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"训练集 R²: {model.score(X_encoded, y):.3f}")

    # 提取权重
    weights = model.coef_
    intercept = model.intercept_

    # 创建权重 DataFrame
    weight_df = pd.DataFrame({
        'feature': feature_names,
        'weight': weights
    }).sort_values('weight', key=abs, ascending=False)

    print(f"\nTop 10 最重要的特征 (按绝对值):")
    for idx, row in weight_df.head(10).iterrows():
        direction = "↑" if row['weight'] > 0 else "↓"
        print(f"  {row['feature']:40s}: {row['weight']:+.4f} {direction}")

    # 保存权重
    learned_weights[dim] = {
        'intercept': float(intercept),
        'features': {}
    }

    for _, row in weight_df.iterrows():
        if abs(row['weight']) > 0.01:  # 只保存显著的权重
            learned_weights[dim]['features'][row['feature']] = float(row['weight'])

    models[dim] = model

# ========== 保存权重 ==========
print(f"\n【Step 4】保存学习到的权重")
print("=" * 70)

weights_path = 'artifacts/results/learned_weights.json'
with open(weights_path, 'w', encoding='utf-8') as f:
    json.dump(learned_weights, f, indent=2, ensure_ascii=False)

print(f"✅ 权重已保存: {weights_path}")

# 保存 encoder（用于后续预测）
import joblib
encoder_path = 'artifacts/results/onehot_encoder.joblib'
joblib.dump(encoder, encoder_path)
print(f"✅ Encoder 已保存: {encoder_path}")

# ========== 权重分析 ==========
print(f"\n【Step 5】权重分析")
print("=" * 70)

# 统计每个原始变量的总权重贡献
variable_importance = {}

for dim in OCEAN_DIMS:
    variable_importance[dim] = {}

    for var in categorical_vars:
        # 找到所有与该变量相关的特征
        var_features = [f for f in feature_names if f.startswith(var + '_')]

        # 计算该变量的总权重（绝对值之和）
        total_weight = 0
        for feat in var_features:
            if feat in learned_weights[dim]['features']:
                total_weight += abs(learned_weights[dim]['features'][feat])

        variable_importance[dim][var] = total_weight

# 可视化
print("\n各变量对 OCEAN 的重要性（权重绝对值之和）:")
importance_df = pd.DataFrame(variable_importance).T
print(importance_df.to_string())

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
importance_df.T.plot(kind='bar', ax=ax)
ax.set_xlabel('Variable')
ax.set_ylabel('Total Weight (Absolute Sum)')
ax.set_title('Variable Importance for Each OCEAN Dimension')
ax.legend(title='OCEAN', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('artifacts/results/variable_importance.png', dpi=150)
print(f"\n✅ 图表已保存: artifacts/results/variable_importance.png")
plt.close()

# ========== 权重合理性检查 ==========
print(f"\n【Step 6】权重合理性检查")
print("=" * 70)

print("\n心理学合理性验证:")

# 检查 Grade 对 Conscientiousness 的影响
print("\n1. Grade → Conscientiousness (预期: A高, G低)")
for feat in ['grade_A', 'grade_C', 'grade_G']:
    if feat in learned_weights['conscientiousness']['features']:
        weight = learned_weights['conscientiousness']['features'][feat]
        print(f"   {feat}: {weight:+.4f}")

# 检查 Grade 对 Neuroticism 的影响
print("\n2. Grade → Neuroticism (预期: A低, G高)")
for feat in ['grade_A', 'grade_C', 'grade_G']:
    if feat in learned_weights['neuroticism']['features']:
        weight = learned_weights['neuroticism']['features'][feat]
        print(f"   {feat}: {weight:+.4f}")

# 检查 Purpose 对 Openness 的影响
print("\n3. Purpose → Openness (预期: small_business 高)")
for feat in ['purpose_small_business', 'purpose_debt_consolidation', 'purpose_home_improvement']:
    if feat in learned_weights['openness']['features']:
        weight = learned_weights['openness']['features'][feat]
        print(f"   {feat}: {weight:+.4f}")

print("\n" + "=" * 70)
print("✅ Phase 2 完成！权重学习完成")
print("=" * 70)

print("\n关键输出:")
print(f"  1. {weights_path}")
print(f"     → 每个特征对每个 OCEAN 维度的权重系数")
print(f"  2. {encoder_path}")
print(f"     → One-Hot Encoder (用于新数据预测)")
print(f"  3. artifacts/results/variable_importance.png")
print(f"     → 变量重要性可视化")

print("\n下一步: 运行 supervised_step3_apply_weights.py")
print("这将用学习到的权重为所有数据打分")
