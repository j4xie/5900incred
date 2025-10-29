"""
Step 3: 小批量测试 OCEAN 打分（100 条数据）
"""
import sys
sys.path.append('.')

import pandas as pd
import kagglehub
from text_features.personality_simple import SimplifiedOceanScorer, OCEAN_DIMS

print("=" * 60)
print("Step 3: 小批量测试 (100 条数据)")
print("=" * 60)

# 加载数据
path = kagglehub.dataset_download("ethon0426/lending-club-20072020q1")
file_path = path + "/Loan_status_2007-2020Q3.gzip"

# 只读 100 条
df = pd.read_csv(file_path, nrows=100, low_memory=False)

# 准备 target
df = df[df['loan_status'].isin(["Fully Paid", "Charged Off"])].copy()
df['target'] = (df['loan_status'] == "Charged Off").astype(int)

print(f"\n✅ 数据加载完成: {len(df)} 行")
print(f"   违约率: {df['target'].mean():.2%}")

# 初始化打分器
scorer = SimplifiedOceanScorer(offline_mode=True)

# 批量打分
print(f"\n🔄 开始批量打分...")
ocean_scores = scorer.score_batch(df, rate_limit_delay=0)

# 转换为 DataFrame
ocean_df = pd.DataFrame(ocean_scores)

# 显示前 10 条
print(f"\n【前 10 条 OCEAN 分数】")
print(ocean_df.head(10).to_string(index=True))

# 统计信息
print(f"\n【OCEAN 分数统计】")
print(ocean_df.describe().to_string())

# 打分统计
print(f"\n【打分统计】")
stats = scorer.get_stats()
for key, value in stats.items():
    print(f"  {key:20s}: {value}")

print("\n" + "=" * 60)
print("✅ Step 3 完成！100 条数据打分成功")
print("=" * 60)
print("\n下一步: 运行 step4_full_batch.py (5000 条数据)")
