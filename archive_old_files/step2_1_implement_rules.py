"""
Step 2.1: 实现基于规则的 OCEAN 打分算法
使用 Step 1.3 设计的映射规则
"""
import json
import pandas as pd
import numpy as np
import kagglehub

# 加载映射规则
with open('artifacts/results/ocean_mapping_rules.json', 'r', encoding='utf-8') as f:
    MAPPING_RULES = json.load(f)

OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

def compute_ocean_rule_based(borrower):
    """
    基于规则计算 OCEAN 分数

    Args:
        borrower: dict 或 pandas Series，包含借款人特征

    Returns:
        dict: OCEAN 分数 (0-1 scale)
    """
    # 初始化为中性值
    ocean = {dim: 0.50 for dim in OCEAN_DIMS}

    # 应用 grade 规则
    if 'grade' in borrower and borrower['grade'] in MAPPING_RULES['grade']['OCEAN_mapping']:
        adjustments = MAPPING_RULES['grade']['OCEAN_mapping'][borrower['grade']]
        for dim, delta in adjustments.items():
            ocean[dim] += delta

    # 应用 purpose 规则
    if 'purpose' in borrower and borrower['purpose'] in MAPPING_RULES['purpose']['OCEAN_mapping']:
        adjustments = MAPPING_RULES['purpose']['OCEAN_mapping'][borrower['purpose']]
        for dim, delta in adjustments.items():
            ocean[dim] += delta

    # 应用 term 规则
    if 'term' in borrower:
        term_clean = str(borrower['term']).strip()
        if term_clean in MAPPING_RULES['term']['OCEAN_mapping']:
            adjustments = MAPPING_RULES['term']['OCEAN_mapping'][term_clean]
            for dim, delta in adjustments.items():
                ocean[dim] += delta

    # 应用 home_ownership 规则
    if 'home_ownership' in borrower and borrower['home_ownership'] in MAPPING_RULES['home_ownership']['OCEAN_mapping']:
        adjustments = MAPPING_RULES['home_ownership']['OCEAN_mapping'][borrower['home_ownership']]
        for dim, delta in adjustments.items():
            ocean[dim] += delta

    # 应用 emp_length 规则（简化版）
    if 'emp_length' in borrower:
        emp = str(borrower['emp_length'])
        if '10+' in emp or '10 years' in emp:
            adjustments = MAPPING_RULES['emp_length']['OCEAN_mapping']['10+ years']
            for dim, delta in adjustments.items():
                ocean[dim] += delta
        elif '< 1' in emp or '1 year' in emp:
            adjustments = MAPPING_RULES['emp_length']['OCEAN_mapping']['< 1 year']
            for dim, delta in adjustments.items():
                ocean[dim] += delta

    # 应用 verification_status 规则
    if 'verification_status' in borrower and borrower['verification_status'] in MAPPING_RULES['verification_status']['OCEAN_mapping']:
        adjustments = MAPPING_RULES['verification_status']['OCEAN_mapping'][borrower['verification_status']]
        for dim, delta in adjustments.items():
            ocean[dim] += delta

    # 裁剪到 [0.25, 0.75] 范围
    for dim in ocean:
        ocean[dim] = max(0.25, min(0.75, ocean[dim]))

    return ocean


# ========== 测试函数 ==========
print("=" * 70)
print("Step 2.1: 测试 Rule-Based OCEAN 打分算法")
print("=" * 70)

# 测试案例
test_cases = [
    {
        "name": "优质借款人（A级，房主，长期就业）",
        "borrower": {
            "grade": "A",
            "purpose": "home_improvement",
            "term": "36 months",
            "home_ownership": "OWN",
            "emp_length": "10+ years",
            "verification_status": "Verified"
        }
    },
    {
        "name": "高风险借款人（G级，租房，短期就业）",
        "borrower": {
            "grade": "G",
            "purpose": "small_business",
            "term": "60 months",
            "home_ownership": "RENT",
            "emp_length": "< 1 year",
            "verification_status": "Not Verified"
        }
    },
    {
        "name": "中等借款人（C级，按揭，中期就业）",
        "borrower": {
            "grade": "C",
            "purpose": "debt_consolidation",
            "term": "36 months",
            "home_ownership": "MORTGAGE",
            "emp_length": "5 years",
            "verification_status": "Source Verified"
        }
    }
]

print("\n【测试用例】\n")
for case in test_cases:
    print(f"{case['name']}:")
    print(f"  特征: {case['borrower']}")

    scores = compute_ocean_rule_based(case['borrower'])

    print(f"  OCEAN 分数:")
    for dim, score in scores.items():
        bar = '█' * int(score * 40)
        print(f"    {dim:20s}: {score:.3f}  {bar}")
    print()

# ========== 批量打分真实数据 ==========
print("=" * 70)
print("批量打分真实数据 (1000 条)")
print("=" * 70)

path = kagglehub.dataset_download("ethon0426/lending-club-20072020q1")
file_path = path + "/Loan_status_2007-2020Q3.gzip"

df = pd.read_csv(file_path, nrows=1000, low_memory=False)
df = df[df['loan_status'].isin(["Fully Paid", "Charged Off"])].copy()
df['target'] = (df['loan_status'] == "Charged Off").astype(int)

print(f"\n✅ 数据加载: {len(df)} 行\n")
print("🔄 开始打分...")

# 批量打分
ocean_scores = []
for idx, row in df.iterrows():
    scores = compute_ocean_rule_based(row)
    ocean_scores.append(scores)

ocean_df = pd.DataFrame(ocean_scores)

# 显示统计
print(f"\n【OCEAN 分数统计】")
print(ocean_df.describe()[['mean', 'std', 'min', '50%', 'max']].T.to_string())

# 与违约率的相关性
print(f"\n【OCEAN 与违约率的相关性】")
for dim in OCEAN_DIMS:
    df[dim] = ocean_df[dim]
    corr = df[dim].corr(df['target'])
    print(f"  {dim:20s}: r = {corr:+.3f}")

# 对比违约 vs 非违约的 OCEAN 平均值
print(f"\n【违约 vs 非违约的 OCEAN 平均值】")
ocean_by_target = df.groupby('target')[OCEAN_DIMS].mean()
print(ocean_by_target.T.to_string())

print(f"\n【差异（违约 - 非违约）】")
diff = ocean_by_target.loc[1] - ocean_by_target.loc[0]
for dim in OCEAN_DIMS:
    direction = "↑" if diff[dim] > 0 else "↓"
    print(f"  {dim:20s}: {diff[dim]:+.3f} {direction}")

print("\n" + "=" * 70)
print("✅ Rule-Based 打分算法实现完成！")
print("=" * 70)
print("\n关键发现:")
if diff['conscientiousness'] < -0.02:
    print("  ✅ 违约者的 Conscientiousness 更低（符合预期）")
if diff['neuroticism'] > 0.02:
    print("  ✅ 违约者的 Neuroticism 更高（符合预期）")

print("\n下一步: 用这个算法重新跑 A/B 对比")
print("运行: python3 step2_2_rerun_ab_test.py")
