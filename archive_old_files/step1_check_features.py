"""
Step 1: 检查 LendingClub 数据中可用的 string/categorical features
"""
import pandas as pd
import kagglehub

# 加载数据
path = kagglehub.dataset_download("ethon0426/lending-club-20072020q1")
file_path = path + "/Loan_status_2007-2020Q3.gzip"

# 只读取 1000 行用于快速检查
df = pd.read_csv(file_path, nrows=1000, low_memory=False)

print("=" * 60)
print("Step 1: 检查可用的 String/Categorical Features")
print("=" * 60)

# 定义我们想用的特征
target_features = [
    "term",                 # 贷款期限
    "grade",                # 信用等级
    "sub_grade",            # 信用子等级
    "emp_length",           # 就业年限
    "home_ownership",       # 房屋所有权
    "verification_status",  # 收入验证状态
    "pymnt_plan",           # 还款计划
    "purpose",              # 贷款目的
    "initial_list_status",  # 初始列表状态
    "application_type"      # 申请类型
]

print(f"\n目标特征（我们想用的）: {len(target_features)} 个")
print("-" * 60)

# 检查哪些存在于数据中
available_features = []
for feature in target_features:
    if feature in df.columns:
        available_features.append(feature)

        # 统计非空值
        non_null = df[feature].notna().sum()
        coverage = non_null / len(df) * 100
        n_unique = df[feature].nunique()

        print(f"✅ {feature:25s} | 覆盖率: {coverage:5.1f}% | 唯一值: {n_unique:3d}")

        # 显示前 5 个样例值
        samples = df[feature].dropna().unique()[:5].tolist()
        print(f"   样例: {samples}")
    else:
        print(f"❌ {feature:25s} | 不存在于数据集")

print("\n" + "=" * 60)
print(f"总结: {len(available_features)}/{len(target_features)} 个特征可用")
print("=" * 60)

# 保存可用特征列表到文件
with open('available_features.txt', 'w') as f:
    f.write('\n'.join(available_features))

print(f"\n✅ 可用特征列表已保存到: available_features.txt")
print(f"\n下一步: 运行 step2_create_scorer.py")
