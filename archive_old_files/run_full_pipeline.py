"""
完整流程: Step 4-10 一次性运行
从数据加载 → OCEAN 打分 → 模型训练 → A/B 对比 → 可视化
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from text_features.personality_simple import SimplifiedOceanScorer, OCEAN_DIMS
from utils.seed import set_seed

# 设置随机种子
set_seed(42)
sns.set_style('whitegrid')

print("=" * 70)
print(" "*20 + "完整 OCEAN 集成流程")
print("=" * 70)

# ========== Step 4: 加载数据 ==========
print("\n【Step 4】 加载数据 (5000 条)")
print("-" * 70)

path = kagglehub.dataset_download("ethon0426/lending-club-20072020q1")
file_path = path + "/Loan_status_2007-2020Q3.gzip"

ROW_LIMIT = 5000
df = pd.read_csv(file_path, nrows=ROW_LIMIT, low_memory=False)

# 准备 target
df = df[df['loan_status'].isin(["Fully Paid", "Charged Off"])].copy()
df['target'] = (df['loan_status'] == "Charged Off").astype(int)

print(f"✅ 数据加载: {len(df)} 行")
print(f"   违约率: {df['target'].mean():.2%}")

# 批量打分
print(f"\n🔄 OCEAN 打分中...")
scorer = SimplifiedOceanScorer(offline_mode=True)
ocean_scores = scorer.score_batch(df, rate_limit_delay=0)
ocean_df = pd.DataFrame(ocean_scores)

# ========== Step 5: 合并 OCEAN 特征 ==========
print(f"\n【Step 5】 合并 OCEAN 特征到 DataFrame")
print("-" * 70)

for dim in OCEAN_DIMS:
    df[dim] = ocean_df[dim]

print(f"✅ 已添加 5 个 OCEAN 特征: {OCEAN_DIMS}")
print(f"\nOCEAN 统计:")
ocean_stats = df[OCEAN_DIMS].describe().T[['mean', 'std', 'min', 'max']]
print(ocean_stats.to_string())

# ========== 准备特征 ==========
print(f"\n【特征工程】 准备 Baseline 和 OCEAN 特征")
print("-" * 70)

# Baseline 数值特征
numeric_features = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc"
]
numeric_features = [c for c in numeric_features if c in df.columns]

# 分类特征
categorical_features = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "application_type"
]
categorical_features = [c for c in categorical_features if c in df.columns]

# 清理百分比列
for col in ["int_rate", "revol_util"]:
    if col in df.columns and df[col].dtype == object:
        df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")

print(f"✅ Baseline 数值特征: {len(numeric_features)} 个")
print(f"✅ Baseline 分类特征: {len(categorical_features)} 个")
print(f"✅ OCEAN 特征: {len(OCEAN_DIMS)} 个")

# 定义两个特征集
features_baseline = numeric_features + categorical_features
features_with_ocean = numeric_features + OCEAN_DIMS + categorical_features

print(f"\n📊 特征集 A (Baseline): {len(features_baseline)} 个特征")
print(f"📊 特征集 B (Baseline+OCEAN): {len(features_with_ocean)} 个特征")

# Train/Test Split
X = df[features_with_ocean].copy()
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ 数据切分: Train={len(X_train)}, Test={len(X_test)}")

# ========== Step 6: 训练 Baseline 模型 (XGBoost) ==========
print(f"\n【Step 6】 训练 Baseline 模型 (不含 OCEAN)")
print("-" * 70)

X_train_a = X_train[features_baseline]
X_test_a = X_test[features_baseline]

numeric_a = [f for f in numeric_features if f in features_baseline]
categorical_a = [f for f in categorical_features if f in features_baseline]

# 构建 pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

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
        tree_method="hist",
        n_estimators=100,  # 减少数量加快训练
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    ))
])

print(f"🔄 训练中...")
model_a.fit(X_train_a, y_train)

y_proba_a = model_a.predict_proba(X_test_a)[:, 1]
roc_auc_a = roc_auc_score(y_test, y_proba_a)
pr_auc_a = average_precision_score(y_test, y_proba_a)

print(f"✅ Baseline 模型训练完成")
print(f"   ROC-AUC: {roc_auc_a:.4f}")
print(f"   PR-AUC:  {pr_auc_a:.4f}")

# ========== Step 7: 训练增强模型 (Baseline + OCEAN) ==========
print(f"\n【Step 7】 训练增强模型 (含 OCEAN)")
print("-" * 70)

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
        tree_method="hist",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    ))
])

print(f"🔄 训练中...")
model_b.fit(X_train_b, y_train)

y_proba_b = model_b.predict_proba(X_test_b)[:, 1]
roc_auc_b = roc_auc_score(y_test, y_proba_b)
pr_auc_b = average_precision_score(y_test, y_proba_b)

print(f"✅ 增强模型训练完成")
print(f"   ROC-AUC: {roc_auc_b:.4f}")
print(f"   PR-AUC:  {pr_auc_b:.4f}")

# ========== Step 8: A/B 对比 ==========
print(f"\n【Step 8】 A/B 对比")
print("=" * 70)
print(f"{'Metric':<15} {'A (Baseline)':<20} {'B (+OCEAN)':<20} {'Delta':<15}")
print("-" * 70)

delta_roc = roc_auc_b - roc_auc_a
delta_pr = pr_auc_b - pr_auc_a

print(f"{'ROC-AUC':<15} {roc_auc_a:<20.4f} {roc_auc_b:<20.4f} {delta_roc:+.4f}")
print(f"{'PR-AUC':<15} {pr_auc_a:<20.4f} {pr_auc_b:<20.4f} {delta_pr:+.4f}")
print("=" * 70)

# 判断是否达到成功标准
success = False
if delta_roc >= 0.010:
    print(f"✅ ROC-AUC 提升 {delta_roc:.4f} >= +0.010 (成功!)")
    success = True
elif delta_pr >= 0.008:
    print(f"✅ PR-AUC 提升 {delta_pr:.4f} >= +0.008 (成功!)")
    success = True
else:
    print(f"⚠️  未达到最低提升标准 (ROC-AUC +0.010 或 PR-AUC +0.008)")
    print(f"   但技术管线已验证可行")

# ========== Step 9: 可视化 ==========
print(f"\n【Step 9】 生成可视化")
print("-" * 70)

from sklearn.metrics import roc_curve, precision_recall_curve

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr_a, tpr_a, _ = roc_curve(y_test, y_proba_a)
fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_b)

axes[0].plot(fpr_a, tpr_a, label=f"Baseline (AUC={roc_auc_a:.3f})", linewidth=2)
axes[0].plot(fpr_b, tpr_b, label=f"+ OCEAN (AUC={roc_auc_b:.3f})", linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Comparison')
axes[0].legend()
axes[0].grid(True)

# Precision-Recall Curve
prec_a, rec_a, _ = precision_recall_curve(y_test, y_proba_a)
prec_b, rec_b, _ = precision_recall_curve(y_test, y_proba_b)

axes[1].plot(rec_a, prec_a, label=f"Baseline (PR-AUC={pr_auc_a:.3f})", linewidth=2)
axes[1].plot(rec_b, prec_b, label=f"+ OCEAN (PR-AUC={pr_auc_b:.3f})", linewidth=2)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve Comparison')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('artifacts/results/ab_comparison.png', dpi=150)
print(f"✅ 图表已保存: artifacts/results/ab_comparison.png")

# ========== Step 10: 导出结果 ==========
print(f"\n【Step 10】 导出结果")
print("-" * 70)

results = {
    'model': ['XGBoost_Baseline', 'XGBoost_OCEAN'],
    'n_features': [len(features_baseline), len(features_with_ocean)],
    'roc_auc': [roc_auc_a, roc_auc_b],
    'pr_auc': [pr_auc_a, pr_auc_b],
    'roc_delta': [0, delta_roc],
    'pr_delta': [0, delta_pr]
}

results_df = pd.DataFrame(results)
results_df.to_csv('artifacts/results/ab_results.csv', index=False)
print(f"✅ 结果已保存: artifacts/results/ab_results.csv")
print(f"\n最终结果表:")
print(results_df.to_string(index=False))

print(f"\n" + "=" * 70)
print(" "*25 + "🎉 流程完成！")
print("=" * 70)
print(f"\n查看结果:")
print(f"  - 图表: artifacts/results/ab_comparison.png")
print(f"  - 数据: artifacts/results/ab_results.csv")
