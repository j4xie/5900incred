"""
Step 2: 测试 OCEAN 打分函数（使用现有分类特征）
"""
import sys
sys.path.append('.')

from text_features.personality_simple import SimplifiedOceanScorer, build_borrower_profile

print("=" * 60)
print("Step 2: 测试 OCEAN 打分函数")
print("=" * 60)

# 创建测试样本
test_borrower = {
    "term": " 60 months",
    "grade": "C",
    "sub_grade": "C4",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "pymnt_plan": "n",
    "purpose": "debt_consolidation",
    "initial_list_status": "f",
    "application_type": "Individual"
}

print("\n【测试借款人信息】")
print("-" * 60)
for key, value in test_borrower.items():
    print(f"  {key:25s}: {value}")

# 构建 Profile
profile = build_borrower_profile(test_borrower)
print(f"\n【生成的 Profile 字符串】")
print(f"  {profile}")

# 初始化打分器（offline 模式）
print(f"\n【初始化 OCEAN 打分器】")
scorer = SimplifiedOceanScorer(offline_mode=True)
print(f"  模式: Offline (deterministic)")
print(f"  缓存目录: {scorer.cache_dir}")

# 打分
print(f"\n【OCEAN 打分结果】")
print("-" * 60)
scores = scorer.score_row(test_borrower)

for dim, score in scores.items():
    bar = '█' * int(score * 30)
    print(f"  {dim:20s}: {score:.4f}  {bar}")

# 统计信息
print(f"\n【打分统计】")
stats = scorer.get_stats()
for key, value in stats.items():
    print(f"  {key:20s}: {value}")

print("\n" + "=" * 60)
print("✅ Step 2 完成！打分函数工作正常")
print("=" * 60)
print("\n下一步: 运行 step3_batch_score.py (小批量测试)")
