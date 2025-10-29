"""
Step 1.2: 分析每个 Variable 与违约率(target)的关系
目标: 理解哪些变量对信贷风险有影响，为 OCEAN 映射提供数据支撑
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import kagglehub

print("=" * 70)
print("Step 1.2: Variable vs Default Rate Analysis")
print("=" * 70)

# 加载数据
path = kagglehub.dataset_download("ethon0426/lending-club-20072020q1")
file_path = path + "/Loan_status_2007-2020Q3.gzip"

df = pd.read_csv(file_path, nrows=10000, low_memory=False)

# 准备 target
df = df[df['loan_status'].isin(["Fully Paid", "Charged Off"])].copy()
df['target'] = (df['loan_status'] == "Charged Off").astype(int)

print(f"\n数据加载: {len(df)} 行")
print(f"整体违约率: {df['target'].mean():.2%}\n")

# 要分析的变量
variables_to_analyze = [
    "term", "grade", "sub_grade", "emp_length",
    "home_ownership", "verification_status", "purpose"
]

results = []

print("=" * 70)
print(f"{'Variable':<25} {'Category':<20} {'Count':>8} {'Default%':>10} {'Lift':>8}")
print("=" * 70)

for var in variables_to_analyze:
    if var not in df.columns:
        continue

    # 按类别统计违约率
    grouped = df.groupby(var).agg({
        'target': ['count', 'mean']
    }).reset_index()

    grouped.columns = [var, 'count', 'default_rate']
    grouped['lift'] = grouped['default_rate'] / df['target'].mean()
    grouped = grouped.sort_values('default_rate', ascending=False)

    # 卡方检验
    contingency_table = pd.crosstab(df[var], df['target'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"\n【{var}】 (Chi2 p-value: {p_value:.4f})")
    print("-" * 70)

    for _, row in grouped.iterrows():
        category = str(row[var])[:18]
        count = int(row['count'])
        default_rate = row['default_rate']
        lift = row['lift']

        print(f"  {category:<23} {count:>8,} {default_rate:>9.2%} {lift:>7.2f}x")

        results.append({
            'variable': var,
            'category': row[var],
            'count': count,
            'default_rate': default_rate,
            'lift': lift,
            'chi2_p_value': p_value
        })

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('artifacts/results/variable_analysis.csv', index=False)
print(f"\n✅ 结果已保存: artifacts/results/variable_analysis.csv")

# 生成可视化
print(f"\n🎨 生成可视化...")

# 选择 Top 3 最重要的变量（基于 p-value）
top_vars = results_df.groupby('variable')['chi2_p_value'].first().nsmallest(3).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, var in enumerate(top_vars):
    var_data = results_df[results_df['variable'] == var].sort_values('default_rate', ascending=False)

    axes[i].barh(range(len(var_data)), var_data['default_rate'] * 100, color='steelblue')
    axes[i].set_yticks(range(len(var_data)))
    axes[i].set_yticklabels(var_data['category'])
    axes[i].set_xlabel('Default Rate (%)')
    axes[i].set_title(f'{var}\n(p={var_data.iloc[0]["chi2_p_value"]:.4f})')
    axes[i].axvline(df['target'].mean() * 100, color='red', linestyle='--',
                    linewidth=2, label='Overall Avg')
    axes[i].legend()
    axes[i].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/results/variable_vs_default.png', dpi=150)
print(f"✅ 图表已保存: artifacts/results/variable_vs_default.png")

print("\n" + "=" * 70)
print("分析完成！关键发现:")
print("=" * 70)

# 总结关键发现
print("\n【最显著的变量】（p < 0.001）:")
sig_vars = results_df[results_df['chi2_p_value'] < 0.001]['variable'].unique()
for var in sig_vars:
    p_val = results_df[results_df['variable'] == var]['chi2_p_value'].iloc[0]
    print(f"  - {var:25s} (p = {p_val:.6f})")

print("\n【高风险类别】（Lift > 1.3x）:")
high_risk = results_df[results_df['lift'] > 1.3].sort_values('lift', ascending=False).head(10)
for _, row in high_risk.iterrows():
    print(f"  - {row['variable']:15s} = {str(row['category']):20s} → {row['default_rate']:.1%} (Lift {row['lift']:.2f}x)")

print("\n【低风险类别】（Lift < 0.7x）:")
low_risk = results_df[results_df['lift'] < 0.7].sort_values('lift').head(10)
for _, row in low_risk.iterrows():
    print(f"  - {row['variable']:15s} = {str(row['category']):20s} → {row['default_rate']:.1%} (Lift {row['lift']:.2f}x)")

print("\n下一步: 基于这些发现，设计 Variable → OCEAN 的映射")
print("运行: python3 step1_3_design_mapping.py")
