"""
Supervised Learning - Phase 3: 应用学习到的权重

用 Phase 2 学习到的权重为完整数据集打分
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
import joblib
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

print("=" * 70)
print("Supervised Learning - Phase 3: 应用学习到的权重")
print("=" * 70)

# ========== 加载权重和 Encoder ==========
print(f"\n【Step 1】加载学习到的权重")
print("-" * 70)

try:
    with open('artifacts/results/learned_weights.json', 'r') as f:
        learned_weights = json.load(f)

    encoder = joblib.load('artifacts/results/onehot_encoder.joblib')

    print(f"✅ 权重加载成功")
    print(f"   OCEAN 维度: {list(learned_weights.keys())}")
except FileNotFoundError as e:
    print(f"❌ 错误: {e}")
    print(f"请先运行: python3 supervised_step2_learn_weights.py")
    exit(1)

OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# ========== 加载完整数据集 ==========
print(f"\n【Step 2】加载完整数据集")
print("-" * 70)

path = kagglehub.dataset_download("ethon0426/lending-club-20072020q1")
file_path = path + "/Loan_status_2007-2020Q3.gzip"

ROW_LIMIT = 5000
df = pd.read_csv(file_path, nrows=ROW_LIMIT, low_memory=False)
df = df[df['loan_status'].isin(["Fully Paid", "Charged Off"])].copy()
df['target'] = (df['loan_status'] == "Charged Off").astype(int)

print(f"✅ 数据加载: {len(df)} 行")
print(f"   违约率: {df['target'].mean():.2%}")

# ========== 批量打分（使用学习到的权重）==========
print(f"\n【Step 3】使用学习到的权重批量打分")
print("-" * 70)

# 准备 categorical variables
categorical_vars = ['grade', 'purpose', 'term', 'home_ownership', 'emp_length', 'verification_status']
categorical_vars = [v for v in categorical_vars if v in df.columns]

X_cat = df[categorical_vars].copy()
for col in categorical_vars:
    X_cat[col] = X_cat[col].fillna('MISSING')

# One-Hot Encoding
X_encoded = encoder.transform(X_cat)
feature_names = encoder.get_feature_names_out(categorical_vars)

print(f"🔄 开始打分 {len(df)} 条样本...")

# 为每个 OCEAN 维度计算分数
ocean_scores = {dim: [] for dim in OCEAN_DIMS}

for i in range(len(df)):
    row_encoded = X_encoded[i]

    for dim in OCEAN_DIMS:
        # 获取该维度的权重
        intercept = learned_weights[dim]['intercept']
        weights = learned_weights[dim]['features']

        # 计算线性组合
        score = intercept

        for j, feat_name in enumerate(feature_names):
            if feat_name in weights:
                score += weights[feat_name] * row_encoded[j]

        # 裁剪到 [0.25, 0.75] 范围
        score = max(0.25, min(0.75, score))

        ocean_scores[dim].append(score)

# 添加到 DataFrame
for dim in OCEAN_DIMS:
    df[dim] = ocean_scores[dim]

print(f"✅ 打分完成")

# 统计
print(f"\n【OCEAN 分数统计】")
ocean_stats = df[OCEAN_DIMS].describe().T[['mean', 'std', 'min', 'max']]
print(ocean_stats.to_string())

# 与违约率相关性
print(f"\n【OCEAN 与违约率的相关性】")
for dim in OCEAN_DIMS:
    corr = df[dim].corr(df['target'])
    print(f"  {dim:20s}: r = {corr:+.3f}")

# ========== A/B 对比测试 ==========
print(f"\n【Step 4】A/B 对比测试")
print("=" * 70)

# Baseline 特征
numeric_features = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc"
]
numeric_features = [c for c in numeric_features if c in df.columns]

categorical_features = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "application_type"
]
categorical_features = [c for c in categorical_features if c in df.columns]

# 清理百分比列
for col in ["int_rate", "revol_util"]:
    if col in df.columns and df[col].dtype == object:
        df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")

# Feature Sets
features_baseline = numeric_features + categorical_features
features_with_ocean = numeric_features + OCEAN_DIMS + categorical_features

print(f"特征集 A (Baseline): {len(features_baseline)} 个特征")
print(f"特征集 B (Baseline + Learned OCEAN): {len(features_with_ocean)} 个特征")

# Train/Test Split
X = df[features_with_ocean].copy()
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

# ========== Model A: Baseline ==========
print(f"\n🔄 训练 Model A (Baseline)...")

X_train_a = X_train[features_baseline]
X_test_a = X_test[features_baseline]

numeric_a = [f for f in numeric_features if f in features_baseline]
categorical_a = [f for f in categorical_features if f in features_baseline]

preprocess_a = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_a),
        ("cat", categorical_transformer, categorical_a),
    ],
    remainder="drop",
    sparse_threshold=0.3
)

pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
scale_pos_weight = neg / max(1, pos)

model_a = Pipeline(steps=[
    ("preprocess", preprocess_a),
    ("clf", XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    ))
])

model_a.fit(X_train_a, y_train)
y_proba_a = model_a.predict_proba(X_test_a)[:, 1]

roc_auc_a = roc_auc_score(y_test, y_proba_a)
pr_auc_a = average_precision_score(y_test, y_proba_a)

print(f"✅ Model A: ROC-AUC={roc_auc_a:.4f}, PR-AUC={pr_auc_a:.4f}")

# ========== Model B: With Learned OCEAN ==========
print(f"\n🔄 训练 Model B (Baseline + Learned OCEAN)...")

X_train_b = X_train[features_with_ocean]
X_test_b = X_test[features_with_ocean]

numeric_b = [f for f in numeric_features + OCEAN_DIMS if f in features_with_ocean]
categorical_b = [f for f in categorical_features if f in features_with_ocean]

preprocess_b = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_b),
        ("cat", categorical_transformer, categorical_b),
    ],
    remainder="drop",
    sparse_threshold=0.3
)

model_b = Pipeline(steps=[
    ("preprocess", preprocess_b),
    ("clf", XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    ))
])

model_b.fit(X_train_b, y_train)
y_proba_b = model_b.predict_proba(X_test_b)[:, 1]

roc_auc_b = roc_auc_score(y_test, y_proba_b)
pr_auc_b = average_precision_score(y_test, y_proba_b)

print(f"✅ Model B: ROC-AUC={roc_auc_b:.4f}, PR-AUC={pr_auc_b:.4f}")

# ========== 结果对比 ==========
print(f"\n【A/B 对比结果】")
print("=" * 70)
print(f"{'Metric':<15} {'A (Baseline)':<20} {'B (Learned OCEAN)':<20} {'Delta':<15}")
print("-" * 70)

delta_roc = roc_auc_b - roc_auc_a
delta_pr = pr_auc_b - pr_auc_a

print(f"{'ROC-AUC':<15} {roc_auc_a:<20.4f} {roc_auc_b:<20.4f} {delta_roc:+.4f}")
print(f"{'PR-AUC':<15} {pr_auc_a:<20.4f} {pr_auc_b:<20.4f} {delta_pr:+.4f}")
print("=" * 70)

# 判断结果
if delta_roc >= 0.010:
    print(f"\n🎉 成功！ROC-AUC 提升 {delta_roc:.4f} >= +0.010")
elif delta_pr >= 0.008:
    print(f"\n🎉 成功！PR-AUC 提升 {delta_pr:.4f} >= +0.008")
elif delta_roc > 0:
    print(f"\n✅ 有提升！ROC-AUC +{delta_roc:.4f}（虽未达最低标准但方向正确）")
else:
    print(f"\n⚠️  未提升。可能原因:")
    print(f"   - 样本量不足（当前 {ROW_LIMIT} 条）")
    print(f"   - Ground Truth 质量（是否用真实 LLM？）")
    print(f"   - 需要调整正则化参数（Ridge alpha）")

# 保存结果
results = {
    'model': ['XGBoost_Baseline', 'XGBoost_LearnedOCEAN'],
    'n_features': [len(features_baseline), len(features_with_ocean)],
    'roc_auc': [roc_auc_a, roc_auc_b],
    'pr_auc': [pr_auc_a, pr_auc_b],
    'roc_delta': [0, delta_roc],
    'pr_delta': [0, delta_pr]
}

results_df = pd.DataFrame(results)
results_df.to_csv('artifacts/results/supervised_ab_results.csv', index=False)

print(f"\n✅ 结果已保存: artifacts/results/supervised_ab_results.csv")
print("\n" + "=" * 70)
print("✅ Supervised Learning 完整流程完成！")
print("=" * 70)
