"""
Supervised Learning - Phase 1: 用 LLM 标注 Ground Truth OCEAN 分数

这一步会：
1. 从数据集中随机抽取 N 条样本
2. 用 LLM (OpenAI API) 给每条样本打 OCEAN 分数
3. 保存为训练集（ground truth）

预计成本: ~$1-3 (取决于样本数量)
预计时间: 10-20 分钟 (取决于 API 速率限制)
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import kagglehub
import os
from text_features.personality_simple import SimplifiedOceanScorer, OCEAN_DIMS

print("=" * 70)
print("Supervised Learning - Phase 1: 标注 Ground Truth")
print("=" * 70)

# ========== 配置 ==========
SAMPLE_SIZE = 500  # 标注样本数量（可调整：500-1000）
USE_API = True      # 是否使用真实 LLM API（需要 OPENAI_API_KEY）

# 检查 API key
if USE_API and not os.getenv('OPENAI_API_KEY'):
    print("\n⚠️  警告: 未检测到 OPENAI_API_KEY 环境变量")
    print("请设置 API key: export OPENAI_API_KEY='sk-...'")
    print("\n或者设置 USE_API=False 使用 offline 模式（仅用于测试）")
    response = input("\n继续使用 offline 模式? (y/n): ")
    if response.lower() != 'y':
        exit(0)
    USE_API = False

# ========== 加载数据 ==========
print(f"\n【Step 1】加载数据")
print("-" * 70)

path = kagglehub.dataset_download("ethon0426/lending-club-20072020q1")
file_path = path + "/Loan_status_2007-2020Q3.gzip"

# 加载更多数据以便抽样
df = pd.read_csv(file_path, nrows=10000, low_memory=False)
df = df[df['loan_status'].isin(["Fully Paid", "Charged Off"])].copy()
df['target'] = (df['loan_status'] == "Charged Off").astype(int)

print(f"✅ 数据加载: {len(df)} 行")
print(f"   违约率: {df['target'].mean():.2%}")

# ========== 分层抽样 ==========
print(f"\n【Step 2】分层抽样（确保违约/非违约比例一致）")
print("-" * 70)

# 按 target 分层抽样
sample_df = df.groupby('target', group_keys=False).apply(
    lambda x: x.sample(n=int(SAMPLE_SIZE * len(x) / len(df)), random_state=42)
).reset_index(drop=True)

print(f"✅ 抽样完成: {len(sample_df)} 条")
print(f"   违约: {(sample_df['target']==1).sum()} 条")
print(f"   非违约: {(sample_df['target']==0).sum()} 条")

# ========== LLM 标注 ==========
print(f"\n【Step 3】LLM 标注 OCEAN 分数")
print("-" * 70)

if USE_API:
    print(f"🔄 使用 OpenAI API 标注...")
    print(f"   模型: gpt-4o-mini")
    print(f"   样本数: {len(sample_df)}")
    print(f"   预计成本: ~${len(sample_df) * 0.002:.2f}")
    print(f"   预计时间: ~{len(sample_df) * 1.2 / 60:.1f} 分钟")
    print()

    scorer = SimplifiedOceanScorer(
        cache_dir="../artifacts/persona_cache_supervised",
        offline_mode=False,  # 启用 API
        model="gpt-4o-mini"
    )
else:
    print(f"⚠️  使用 Offline 模式（仅用于测试管线）")
    scorer = SimplifiedOceanScorer(
        cache_dir="../artifacts/persona_cache_supervised",
        offline_mode=True
    )

# 批量打分
ocean_scores = scorer.score_batch(sample_df, rate_limit_delay=0.5)

# 转换为 DataFrame
ocean_df = pd.DataFrame(ocean_scores)

# 合并到样本数据
for dim in OCEAN_DIMS:
    sample_df[f'{dim}_truth'] = ocean_df[dim]

print(f"\n✅ 标注完成！")
print(f"   统计: {scorer.get_stats()}")

# 显示样例
print(f"\n【标注样例】")
display_cols = ['grade', 'purpose', 'term', 'home_ownership'] + \
               [f'{dim}_truth' for dim in OCEAN_DIMS]
print(sample_df[display_cols].head(10).to_string(index=False))

# ========== 保存 Ground Truth ==========
print(f"\n【Step 4】保存 Ground Truth 训练集")
print("-" * 70)

output_path = 'artifacts/results/ground_truth_ocean.csv'
sample_df.to_csv(output_path, index=False)

print(f"✅ 已保存: {output_path}")
print(f"   包含字段:")
print(f"   - 原始特征: grade, purpose, term, etc.")
print(f"   - Ground Truth: openness_truth, conscientiousness_truth, ...")
print(f"   - Target: target (违约标签)")

# ========== 统计分析 ==========
print(f"\n【Step 5】Ground Truth 统计分析")
print("-" * 70)

truth_cols = [f'{dim}_truth' for dim in OCEAN_DIMS]
print("\nOCEAN Ground Truth 统计:")
stats = sample_df[truth_cols].describe().T[['mean', 'std', 'min', 'max']]
print(stats.to_string())

print("\nOCEAN 与违约率的相关性:")
for dim in OCEAN_DIMS:
    corr = sample_df[f'{dim}_truth'].corr(sample_df['target'])
    print(f"  {dim:20s}: r = {corr:+.3f}")

print("\n" + "=" * 70)
print("✅ Phase 1 完成！Ground Truth 已准备好")
print("=" * 70)

if USE_API:
    print("\n💰 实际成本统计:")
    print(f"   API 调用: {scorer.get_stats()['api_calls']} 次")
    print(f"   缓存命中: {scorer.get_stats()['cache_hits']} 次")
    print(f"   估算成本: ~${scorer.get_stats()['api_calls'] * 0.002:.2f}")

print("\n下一步: 运行 supervised_step2_learn_weights.py")
print("这将训练线性回归模型学习权重")
