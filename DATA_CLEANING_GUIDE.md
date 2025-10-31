# 数据清洗方案说明文档

**项目**: Credibly Loan Default Prediction with OCEAN Features
**日期**: 2025-10
**目的**: 从原始贷款数据创建干净、无数据泄漏的建模数据集

---

## 📋 目录

1. [概述](#概述)
2. [数据清洗流程](#数据清洗流程)
3. [详细步骤说明](#详细步骤说明)
4. [特征分类与删除原因](#特征分类与删除原因)
5. [最终数据集说明](#最终数据集说明)
6. [质量控制标准](#质量控制标准)
7. [数据泄漏防控](#数据泄漏防控)

---

## 概述

### 原始数据
- **文件**: `data/loan.csv`
- **规模**: 2,260,668 行 × 145 列
- **大小**: ~1.1 GB
- **来源**: LendingClub 历史贷款数据

### 最终建模数据（中间数据）
- **文件**: `data/loan_clean_for_modeling.csv`
- **规模**: ~126,000 行 × 36 列
- **特征数**: 34 个特征 + 1 个文本特征(desc) + 1 个目标变量(target)
- **数据保留率**: 5.6%

### 最终OCEAN建模数据（desc>=50词过滤后）
- **文件**: `data/loan_final_desc50plus.csv`
- **规模**: 34,529 行 × 33 列
- **特征数**: 31 个建模特征 + 1 个文本特征(desc) + 1 个目标变量(target)
- **数据保留率**: 1.5% (相对于原始数据)
- **用途**: OCEAN人格特征提取和最终XGBoost建模

### 清洗目标
1. ✅ **防止数据泄漏** - 删除所有贷款后产生的信息
2. ✅ **质量控制** - 只保留覆盖率≥80%的高质量特征
3. ✅ **保留文本信息** - 保留desc字段用于OCEAN人格特征提取
4. ✅ **文本质量控制** - 筛选desc≥50词的样本（确保文本质量）
5. ✅ **创建明确目标** - 二分类问题（违约 vs 正常还款）

---

## 数据清洗流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                     loan.csv (原始数据)                              │
│                 2,260,668 行 × 145 列 (~1.1 GB)                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: 初步数据清洗 (01_data_cleaning_with_desc.ipynb)            │
│  ────────────────────────────────────────────────────────            │
│  • 筛选有desc字段的数据 (desc长度 > 1)                               │
│  • 清理百分比字符串 (int_rate, revol_util, sec_app_revol_util)      │
│  • 去除类别型变量的空格                                              │
│  • 分析数据质量（覆盖率、缺失值）                                     │
│  • 生成数据质量报告                                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                loan_with_desc.csv (中间数据)                         │
│                  ~126,000 行 × 145 列                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: 特征分类与泄漏检查 (02_feature_selection_and_leakage_check.ipynb) │
│  ────────────────────────────────────────────────────────────────── │
│  • 分类所有145个特征                                                 │
│    - PRE-LOAN: 贷款前已知 (可用) - ~100个                           │
│    - POST-LOAN: 贷款后产生 (删除) - ~35个                           │
│    - METADATA: 无预测价值 (删除) - ~5个                             │
│    - TARGET: 目标变量 (特殊处理) - 1个                              │
│  • 识别数据泄漏风险                                                  │
│  • 生成特征分类清单                                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: 创建建模数据集 (03_create_modeling_dataset.ipynb)           │
│  ──────────────────────────────────────────────────────────          │
│  • 删除POST-LOAN特征 (~35个) - 防止数据泄漏                          │
│  • 删除METADATA特征 (~5个) - 无预测价值                             │
│  • 删除低质量特征 (覆盖率 < 80%) - 质量控制                          │
│  • 保留desc字段 - 用于OCEAN特征提取                                  │
│  • 创建二分类目标变量:                                               │
│    - target = 0: Fully Paid (正常还款)                              │
│    - target = 1: Charged Off (违约)                                 │
│  • 只保留明确状态的样本                                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│       loan_clean_for_modeling.csv (中间清洗数据)                     │
│                   ~126,000 行 × 36 列                                │
│          34 特征 + 1 desc + 1 target                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: 文本质量过滤 (在特征工程阶段)                               │
│  ────────────────────────────────────────────                        │
│  • 筛选 desc >= 50 词的样本                                          │
│  • 原因: 确保文本足够长以进行有效的OCEAN人格分析                     │
│  • 从 126,000 行减少到 34,529 行                                     │
│  • 保留率: 27.4% (相对于有desc的数据)                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│          loan_final_desc50plus.csv (最终OCEAN建模数据)               │
│                   34,529 行 × 33 列                                  │
│          31 特征 + 1 desc + 1 target                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: Baseline建模准备 (04_xgboost_baseline.ipynb)                │
│  ────────────────────────────────────────────                        │
│  • 删除高基数特征 (3个):                                             │
│    - emp_title (78,000+ 唯一值)                                     │
│    - title (36,000+ 唯一值)                                         │
│    - earliest_cr_line (603 唯一值)                                  │
│  • 删除desc (baseline模型不使用OCEAN)                               │
│  • 最终建模特征: 31个                                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│              Baseline特征矩阵 (无OCEAN特征)                         │
│                 31 个特征 + 1 个目标变量                             │
│              数值型: ~27个 | 分类型: ~4个                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 详细步骤说明

### Step 1: 初步数据清洗

**Notebook**: `notebooks/01_data_preparation/01_data_cleaning_with_desc.ipynb`

#### 1.1 筛选有desc的数据
```python
df_with_desc = df[
    df['desc'].notna() &
    (df['desc'].astype(str).str.strip().str.len() > 1)
].copy()
```
- **原因**: 项目需要从desc字段提取OCEAN人格特征
- **效果**: 从2.26M行筛选到~126K行 (保留率: 5.6%)

#### 1.2 清理百分比字符串
```python
percent_cols = ['int_rate', 'revol_util', 'sec_app_revol_util']
for col in percent_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.strip().str.rstrip('%'),
        errors='coerce'
    )
```
- **原因**: 这些字段存储为"10.5%"格式，需转换为数值10.5
- **处理字段**:
  - `int_rate`: 利率 (如 "10.5%" → 10.5)
  - `revol_util`: 循环额度使用率 (如 "80%" → 80)
  - `sec_app_revol_util`: 第二申请人循环使用率

#### 1.3 去除类别型变量的空格
```python
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()
```
- **原因**: 确保类别一致性（避免"RENT"和" RENT "被视为不同类别）

#### 1.4 数据质量分析
- 计算每个特征的覆盖率（非空值比例）
- 分析数值型特征的统计分布
- 检查XGBoost模型需要的关键特征
- 生成质量报告和可视化

**输出文件**:
- `data/loan_with_desc.csv` - 清洗后的数据
- `numeric_features_quality_report.csv` - 数值型特征质量报告
- `categorical_features_report.csv` - 分类特征报告
- `data_quality_report.png` - 可视化报告

---

### Step 2: 特征分类与数据泄漏检查

**Notebook**: `notebooks/01_data_preparation/02_feature_selection_and_leakage_check.ipynb`

#### 2.1 特征分类定义

##### ✅ PRE-LOAN Features (贷款前 - 可以使用) ~100个

这些信息在**贷款申请时就已知**，可以安全使用：

**A. 贷款申请基本信息** (11个)
- `loan_amnt`, `funded_amnt`, `funded_amnt_inv` - 贷款金额相关
- `term`, `int_rate`, `installment` - 贷款条件
- `grade`, `sub_grade` - LendingClub评级
- `purpose`, `title`, `desc` - 贷款目的和描述

**B. 借款人基本信息** (10个)
- `emp_title`, `emp_length` - 雇佣信息
- `home_ownership` - 住房状况
- `annual_inc` - 年收入
- `verification_status` - 收入验证状态
- `issue_d` - 贷款发放日期
- `pymnt_plan`, `initial_list_status` - 贷款计划
- `application_type`, `policy_code` - 申请类型和政策

**C. 信用历史** (约80个)
包括但不限于：
- FICO分数: `fico_range_low`, `fico_range_high`
- 债务情况: `dti`, `revol_bal`, `revol_util`, `total_acc`
- 拖欠记录: `delinq_2yrs`, `mths_since_last_delinq`
- 查询记录: `inq_last_6mths`, `inq_last_12m`
- 账户信息: `open_acc`, `mort_acc`, `num_actv_rev_tl`
- 公共记录: `pub_rec`, `pub_rec_bankruptcies`, `tax_liens`
- 循环信贷: `bc_util`, `percent_bc_gt_75`, `total_rev_hi_lim`
- 分期付款: `total_bal_il`, `il_util`, `num_il_tl`

**D. 联合申请人信息** (约15个)
- `annual_inc_joint`, `dti_joint` - 联合收入和债务
- `sec_app_fico_range_low/high` - 第二申请人FICO
- `sec_app_*` - 其他联合申请人信息

##### ❌ POST-LOAN Features (贷款后 - 必须删除) ~35个

这些信息在**贷款发放后才产生**，使用会造成数据泄漏：

**A. 还款相关** (15个)
- `out_prncp`, `out_prncp_inv` - 未偿还本金
- `total_pymnt`, `total_pymnt_inv` - 总还款金额
- `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` - 收到的本金、利息、滞纳金
- `recoveries`, `collection_recovery_fee` - 回收金额和催收费用
- `last_pymnt_d`, `last_pymnt_amnt` - 最后还款信息
- `next_pymnt_d` - 下次还款日期
- `last_credit_pull_d` - 最后信用查询日期
- `last_fico_range_high/low` - 最近FICO分数

**B. 困难/重组** (15个)
- `hardship_flag`, `hardship_type`, `hardship_reason` - 困难标志和类型
- `hardship_status`, `hardship_start_date`, `hardship_end_date` - 困难状态和时间
- `hardship_loan_status`, `hardship_dpd`, `hardship_length` - 困难详情
- `hardship_amount`, `hardship_payoff_balance_amount` - 困难金额
- `deferral_term`, `payment_plan_start_date` - 延期和还款计划
- `orig_projected_additional_accrued_interest` - 预计额外利息

**C. 债务清偿** (7个)
- `debt_settlement_flag`, `debt_settlement_flag_date` - 清偿标志和日期
- `settlement_status`, `settlement_date` - 清偿状态和日期
- `settlement_amount`, `settlement_percentage` - 清偿金额和比例
- `settlement_term` - 清偿期限

##### 🗑️ METADATA Features (元数据 - 需删除) ~5个

无预测价值的标识符和冗余信息：
- `id`, `member_id` - 贷款和会员ID
- `url` - 贷款URL
- `zip_code` - 邮编（太细粒度，易过拟合）
- `addr_state` - 州（可选删除）
- `funded_amnt_inv` - 投资者资助金额（与funded_amnt重复）

**注意**: `desc`字段保留用于OCEAN特征提取，提取完成后删除

##### 🎯 TARGET Variable (目标变量)
- `loan_status` - 用于创建目标变量

#### 2.2 数据泄漏示例

**为什么POST-LOAN特征会造成泄漏？**

| 特征 | 为什么是泄漏 | 实际应用问题 |
|------|------------|------------|
| `total_pymnt` | 只有贷款结束后才知道总共还了多少钱 | 申请时无法获得此信息 |
| `out_prncp` | 显示当前未偿还本金，如果=0说明已还清 | 申请时未偿还本金等于贷款金额 |
| `recoveries` | 只有违约后催收才产生 | 如果>0，说明已经违约了 |
| `last_pymnt_amnt` | 最后一次还款金额 | 申请时还没有任何还款 |
| `hardship_flag` | 借款人是否申请困难援助 | 这是贷款后的事件 |

**泄漏的危害**:
- 训练时模型看到：`total_pymnt`高 → Fully Paid
- 但实际应用时无法获得`total_pymnt`
- 导致模型性能严重高估，实际部署失败

**输出文件**:
- `feature_classification_complete.csv` - 完整特征分类
- `features_pre_loan.csv` - 可用特征清单
- `features_post_loan.csv` - 需删除特征清单
- `features_metadata.csv` - 元数据特征清单
- `feature_lists.py` - Python特征列表

---

### Step 3: 创建建模数据集

**Notebook**: `notebooks/01_data_preparation/03_create_modeling_dataset.ipynb`

#### 3.1 删除POST-LOAN特征
```python
post_loan_features = [
    'out_prncp', 'total_pymnt', 'recoveries',
    'hardship_flag', 'debt_settlement_flag',
    # ... 约35个特征
]
df_clean = df.drop(columns=post_loan_features, errors='ignore')
```
- **删除数量**: ~35个特征
- **原因**: 防止数据泄漏

#### 3.2 删除METADATA特征
```python
metadata_features = ['id', 'member_id', 'url', 'zip_code', 'funded_amnt_inv']
df_clean = df_clean.drop(columns=metadata_features, errors='ignore')
```
- **删除数量**: ~5个特征
- **原因**: 无预测价值

#### 3.3 删除低质量特征
```python
# 计算覆盖率
for col in df.columns:
    coverage = (df[col].notna().sum() / len(df)) * 100
    if coverage < 80 and col not in ['desc', 'loan_status']:
        # 删除此特征
```
- **标准**: 覆盖率 < 80%
- **原因**: 缺失值过多，影响模型质量
- **保留**: desc和loan_status即使覆盖率低也保留

#### 3.4 创建目标变量
```python
# 只保留明确状态的样本
df_binary = df[
    (df['loan_status'] == 'Fully Paid') |
    (df['loan_status'] == 'Charged Off')
].copy()

# 创建二分类目标
df_binary['target'] = (df_binary['loan_status'] == 'Charged Off').astype(int)
# target = 0: Fully Paid (正常还款)
# target = 1: Charged Off (违约)

# 删除loan_status
df_binary = df_binary.drop(columns=['loan_status'])
```

**为什么只保留Fully Paid和Charged Off？**
- `Current`: 贷款还在进行中，结果未知
- `In Grace Period`: 宽限期内，可能还也可能违约
- `Late (31-120 days)`: 逾期但未确定违约
- `Default`: 违约（数量很少）

只保留**结果明确**的样本，避免标签噪声。

**输出文件**:
- `data/loan_clean_for_modeling.csv` - **最终建模数据集**
- `feature_coverage_report.csv` - 特征覆盖率报告
- `data_cleaning_summary.png` - 清洗摘要可视化

---

### Step 4: 文本质量过滤（desc >= 50词）

**阶段**: 特征工程阶段（在OCEAN提取前）
**文件**: 此步骤在特征工程脚本或05系列notebook中完成

#### 4.1 为什么要过滤desc长度？

OCEAN人格特征提取需要**足够长的文本**才能进行有效分析：

**文本长度与分析质量的关系**:
- **太短的文本** (< 50词): 信息不足，无法准确判断人格特征
- **中等长度文本** (50-200词): 足够进行基本人格分析
- **较长文本** (> 200词): 提供更丰富的人格信息

**示例**:
```python
# 太短 - 无法分析人格
"Need money for bills"  # 4词 - 信息太少

# 足够长 - 可以分析人格
"I am looking to consolidate my credit card debt to simplify
my finances. I have been working in healthcare for 5 years
and have a stable income. I want to pay off my debt faster
with a lower interest rate..." # 50+词 - 足够分析
```

#### 4.2 过滤实现

```python
# 计算词数
def count_words(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

# 过滤
df['word_count'] = df['desc'].apply(count_words)
df_filtered = df[df['word_count'] >= 50].copy()
```

#### 4.3 过滤效果

**过滤前**: 126,244 行 (有desc的数据)
**过滤后**: 34,529 行 (desc >= 50词)
**保留率**: 27.4%
**平均词数**: 从46词 → 92词

**输出文件**:
- `data/loan_final_desc50plus.csv` - **最终用于OCEAN和XGBoost建模的数据**

---

### Step 5: Baseline建模准备（可选的进一步清洗）

**Notebook**: `notebooks/03_modeling/04_xgboost_baseline.ipynb`

#### 4.1 删除高基数特征
```python
high_cardinality_features = ['emp_title', 'title', 'earliest_cr_line']
X = X.drop(columns=high_cardinality_features, errors='ignore')
```

**为什么删除高基数特征？**

| 特征 | 唯一值数量 | 问题 |
|------|----------|------|
| `emp_title` | 78,000+ | One-Hot编码会产生78,000列 |
| `title` | 36,000+ | One-Hot编码会产生36,000列 |
| `earliest_cr_line` | 603 | 日期字段，应转换为数值 |

**影响**:
- 如果不删除，One-Hot编码后会有116,804列
- 内存占用爆炸，训练时间极长
- 删除后，编码后只有~100-150列

**改进建议**:
- `emp_title`和`title`: 可以使用Target Encoding或Frequency Encoding
- `earliest_cr_line`: 应转换为"信用历史年数"（数值特征）

---

## 特征分类与删除原因

### 删除特征汇总表

| 特征类别 | 数量 | 删除原因 | 示例特征 |
|---------|------|---------|---------|
| POST-LOAN | ~35 | 数据泄漏 | total_pymnt, out_prncp, recoveries |
| METADATA | ~5 | 无预测价值 | id, member_id, url |
| 低质量特征 | ~60+ | 覆盖率<80% | 大量联合申请人字段（仅5%样本是联合申请） |
| 高基数特征 | 3 | One-Hot爆炸 | emp_title, title, earliest_cr_line |
| **总删除** | **~103** | - | - |

### 保留特征汇总表

| 特征类别 | 数量 | 说明 |
|---------|------|------|
| 贷款基本信息 | 8 | loan_amnt, term, int_rate, grade等 |
| 借款人信息 | 4 | emp_length, home_ownership, annual_inc等 |
| 信用历史 | 19 | FICO, DTI, 拖欠记录、账户信息等 |
| 文本特征 | 1 | desc (用于OCEAN提取) |
| 目标变量 | 1 | target |
| **总保留** | **~33-36** | **（取决于统计方式）** |

---

## 最终数据集说明

### 文件信息
- **文件名**: `data/loan_clean_for_modeling.csv`
- **大小**: ~50 MB
- **行数**: 126,244 行
- **列数**: 36 列

### 列信息详解

#### 1. 贷款基本信息 (8列)
```
loan_amnt          - 贷款金额
funded_amnt        - 实际资助金额
term               - 期限 (36/60 months)
int_rate           - 利率 (数值型，已清理百分号)
installment        - 每月分期金额
grade              - 等级 (A-G)
sub_grade          - 子等级 (A1-G5)
purpose            - 贷款目的 (credit_card, debt_consolidation等)
```

#### 2. 借款人信息 (6列)
```
emp_length         - 就业年限 (< 1 year, 1 year, 2 years, ..., 10+ years)
home_ownership     - 住房状况 (RENT, OWN, MORTGAGE)
annual_inc         - 年收入
verification_status - 收入验证状态 (Verified, Source Verified, Not Verified)
issue_d            - 贷款发放日期 (YYYY-MM格式)
application_type   - 申请类型 (Individual, Joint App)
```

#### 3. 信用历史 (19列)
```
dti                - 债务收入比
delinq_2yrs        - 过去2年拖欠次数
inq_last_6mths     - 过去6个月查询次数
open_acc           - 开放账户数量
pub_rec            - 公共贬损记录数
revol_bal          - 循环余额
revol_util         - 循环使用率 (数值型，已清理百分号)
total_acc          - 总账户数
collections_12_mths_ex_med - 过去12个月催收次数（不含医疗）
acc_now_delinq     - 当前拖欠账户数
chargeoff_within_12_mths - 过去12个月核销次数
delinq_amnt        - 拖欠金额
pub_rec_bankruptcies - 公开破产记录数
tax_liens          - 税收留置权数
disbursement_method - 支付方式 (Cash, DirectPay)
addr_state         - 州代码 (CA, NY, TX等)
zip_code           - 邮编前3位
earliest_cr_line   - 最早信用额度开立日期 (YYYY-MM格式)
# 注意: earliest_cr_line在建模时会被删除或应转换为数值
```

#### 4. 文本特征 (1列)
```
desc               - 贷款描述 (借款人填写的贷款目的说明)
                   - 用于提取OCEAN人格特征
                   - 平均长度: ~200-300字符
```

#### 5. 目标变量 (1列)
```
target             - 0: Fully Paid (正常还款)
                   - 1: Charged Off (违约)
                   - 违约率: ~18-20%
```

### 数据分布

#### 目标变量分布
```
Fully Paid (0):   ~100,000 (80%)
Charged Off (1):  ~26,000 (20%)
违约率: 20.6%
```

#### 重要特征的分布
- **term**: 36 months (~70%), 60 months (~30%)
- **grade**: B (25%), C (22%), A (18%), D (15%)
- **home_ownership**: MORTGAGE (48%), RENT (40%), OWN (10%)
- **verification_status**: Source Verified (42%), Verified (31%), Not Verified (27%)
- **purpose**: debt_consolidation (58%), credit_card (23%), other (19%)

---

## 质量控制标准

### 1. 数据完整性标准
- ✅ **覆盖率阈值**: ≥80%
- ✅ **必须完整**: target, desc (用于OCEAN)
- ✅ **允许缺失**: 部分信用历史字段（会用imputation处理）

### 2. 数据类型标准
- ✅ **数值型**: int64, float64
- ✅ **分类型**: object (字符串)
- ✅ **日期型**: 统一为"YYYY-MM"格式（字符串存储）
- ✅ **百分比**: 已转换为数值（如10.5代表10.5%）

### 3. 数据清洁标准
- ✅ **无前后空格**: 所有字符串已strip
- ✅ **无重复行**: 每行代表一个独特的贷款
- ✅ **无明显异常值**: 已通过统计分析验证

### 4. 特征质量等级

| 等级 | 覆盖率 | 数量 | 说明 |
|------|-------|------|------|
| 优秀 | ≥95% | ~25 | 可直接使用 |
| 良好 | 80-95% | ~6 | 可使用，注意imputation |
| 一般 | 50-80% | 0 | 已删除 |
| 较差 | <50% | 0 | 已删除 |

---

## 数据泄漏防控

### 1. 时间序列切分原则
```
训练集: 2007-2011 的贷款
测试集: 2012-2014 的贷款

✅ 正确: 使用旧数据预测新数据
❌ 错误: 随机切分（会泄漏未来信息）
```

**实际实现**:
- 使用`train_test_split`的`stratify`参数保持类别分布
- 如果有时间信息，应基于`issue_d`进行时间切分

### 2. 特征工程时的防泄漏

#### OCEAN特征提取
```python
# ❌ 错误做法：对全部数据提取OCEAN
ocean_features = extract_ocean(df['desc'])  # 全部数据
X_train, X_test = train_test_split(df)  # 然后切分

# ✅ 正确做法：先切分，再分别提取
X_train, X_test = train_test_split(df)
ocean_train = extract_ocean(X_train['desc'])  # 训练集
ocean_test = extract_ocean(X_test['desc'])   # 测试集
```

#### Imputation (填充缺失值)
```python
# ✅ 正确做法
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)  # 只在训练集上学习
X_train_filled = imputer.transform(X_train)
X_test_filled = imputer.transform(X_test)  # 使用训练集的统计量

# ❌ 错误做法
imputer.fit(pd.concat([X_train, X_test]))  # 在全部数据上学习
```

#### Scaling (标准化)
```python
# ✅ 正确做法
scaler = StandardScaler()
scaler.fit(X_train)  # 只在训练集上学习均值和标准差
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ❌ 错误做法
scaler.fit(pd.concat([X_train, X_test]))  # 在全部数据上学习
```

### 3. 泄漏检测清单

在建模前，检查以下问题：

- [ ] 是否删除了所有POST-LOAN特征？
- [ ] 目标变量是否基于未来信息？
- [ ] 特征工程是否在切分后进行？
- [ ] Imputation/Scaling是否只在训练集上fit？
- [ ] 是否使用了test set来调参？
- [ ] Cross-validation是否基于时间顺序？
- [ ] 特征选择是否在训练集上进行？

### 4. 实战验证方法

**时间模拟测试**:
1. 假设今天是2010年，只用2007-2009数据训练
2. 在2010年数据上测试
3. 检查是否使用了2010年才能知道的信息

**特征审查**:
- 对每个特征问："在贷款申请时，我能获得这个信息吗？"
- 如果答案是"否"，则删除

---

## 附录

### A. 数据清洗前后对比

#### 完整清洗流程对比

| 阶段 | 文件名 | 行数 | 列数 | 保留率 | 说明 |
|------|--------|------|------|--------|------|
| 原始 | loan.csv | 2,260,668 | 145 | 100% | 原始数据 |
| 有desc | loan_with_desc.csv | 126,244 | 145 | 5.6% | 筛选有描述的贷款 |
| 清洗 | loan_clean_for_modeling.csv | 126,244 | 36 | 5.6% | 删除POST-LOAN等 |
| **最终** | **loan_final_desc50plus.csv** | **34,529** | **33** | **1.5%** | **desc≥50词+删除高基数** |

#### 特征数量对比

| 特征类型 | 原始数据 | 有desc | 清洗后 | 最终OCEAN |
|---------|---------|--------|--------|----------|
| 数值型特征 | 97 | 97 | 30 | 27 |
| 分类特征 | 47 | 47 | 6 | 4 |
| 文本特征 | 1 | 1 | 1 | 1 |
| 目标变量 | 1 | 1 | 1 | 1 |
| **总计** | **145** | **145** | **36** | **33** |

### B. 删除特征的完整列表

**POST-LOAN特征** (35个):
```
out_prncp, out_prncp_inv, total_pymnt, total_pymnt_inv,
total_rec_prncp, total_rec_int, total_rec_late_fee,
recoveries, collection_recovery_fee, last_pymnt_d,
last_pymnt_amnt, next_pymnt_d, last_credit_pull_d,
last_fico_range_high, last_fico_range_low,
hardship_flag, hardship_type, hardship_reason,
hardship_status, hardship_start_date, hardship_end_date,
hardship_loan_status, hardship_dpd, hardship_length,
hardship_amount, hardship_payoff_balance_amount,
deferral_term, payment_plan_start_date,
debt_settlement_flag, debt_settlement_flag_date,
settlement_status, settlement_date, settlement_amount,
settlement_percentage, settlement_term,
orig_projected_additional_accrued_interest,
pymnt_plan, initial_list_status, policy_code
```

**METADATA特征** (5个):
```
id, member_id, url, zip_code (可选), funded_amnt_inv
```

**高基数特征** (3个，在建模时删除):
```
emp_title, title, earliest_cr_line
```

### C. 保留特征的完整列表 (36列)

**贷款信息** (8):
```
loan_amnt, funded_amnt, term, int_rate, installment,
grade, sub_grade, purpose
```

**借款人信息** (6):
```
emp_length, home_ownership, annual_inc, verification_status,
issue_d, application_type
```

**信用历史** (19):
```
dti, delinq_2yrs, earliest_cr_line, inq_last_6mths,
open_acc, pub_rec, revol_bal, revol_util, total_acc,
collections_12_mths_ex_med, acc_now_delinq,
chargeoff_within_12_mths, delinq_amnt, pub_rec_bankruptcies,
tax_liens, disbursement_method, addr_state, zip_code, title
```

**文本特征** (1):
```
desc
```

**目标变量** (1):
```
target
```

**建模时额外删除** (3):
```
emp_title, title, earliest_cr_line
→ 最终建模特征: 31个
```

### D. 相关文件清单

#### 数据文件
- `data/loan.csv` - 原始数据
- `data/loan_with_desc.csv` - 有desc的数据
- `data/loan_clean_for_modeling.csv` - **最终建模数据**

#### 报告文件
- `numeric_features_quality_report.csv` - 数值特征质量
- `categorical_features_report.csv` - 分类特征统计
- `feature_classification_complete.csv` - 特征分类清单
- `features_pre_loan.csv` - PRE-LOAN特征列表
- `features_post_loan.csv` - POST-LOAN特征列表
- `feature_coverage_report.csv` - 特征覆盖率报告

#### 可视化文件
- `data_quality_report.png` - 数据质量可视化
- `data_cleaning_summary.png` - 清洗摘要

#### Notebook文件
1. `notebooks/01_data_preparation/01_data_cleaning_with_desc.ipynb`
2. `notebooks/01_data_preparation/02_feature_selection_and_leakage_check.ipynb`
3. `notebooks/01_data_preparation/03_create_modeling_dataset.ipynb`
4. `notebooks/03_modeling/04_xgboost_baseline.ipynb`

---

## 总结

### 清洗成果
1. ✅ **防止数据泄漏**: 删除所有POST-LOAN特征
2. ✅ **质量控制**: 只保留覆盖率≥80%的特征
3. ✅ **减少噪声**: 只保留明确结果的样本（Fully Paid / Charged Off）
4. ✅ **文本质量控制**: 只保留desc≥50词的样本
5. ✅ **保留关键信息**: 保留desc用于OCEAN特征提取
6. ✅ **优化性能**: 删除高基数特征，避免维度爆炸

### 最终数据集对比

| 数据集 | 文件名 | 行数 | 列数 | 用途 |
|--------|--------|------|------|------|
| 原始数据 | loan.csv | 2,260,668 | 145 | 原始贷款数据 |
| 有desc | loan_with_desc.csv | 126,244 | 145 | 有描述的贷款 |
| 清洗后 | loan_clean_for_modeling.csv | 126,244 | 36 | 中间清洗数据 |
| **最终OCEAN** | **loan_final_desc50plus.csv** | **34,529** | **33** | **OCEAN+XGBoost建模** |

### 数据集特点
- **高质量**: 所有特征覆盖率≥80%
- **无泄漏**: 严格的PRE-LOAN特征筛选
- **文本充足**: 所有样本desc≥50词（平均92词）
- **平衡性**: 违约率14.6%，适度不平衡
- **可解释**: 保留了业务含义清晰的特征
- **可扩展**: 高质量desc用于OCEAN特征提取

### 下一步
1. 提取OCEAN人格特征 (notebooks/02_feature_engineering/)
2. 训练XGBoost基线模型 (notebooks/03_modeling/04_xgboost_baseline.ipynb)
3. 训练包含OCEAN的完整模型 (notebooks/03_modeling/06_xgboost_with_ocean.ipynb)
4. 性能对比分析 (notebooks/04_results_analysis/)

---

**文档版本**: 1.0
**最后更新**: 2025-10
**维护者**: Credibly Team
