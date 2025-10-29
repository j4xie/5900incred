# 技术实现步骤（详细版）
## 从零开始构建 OCEAN 特征提取系统

---

## 📋 Phase 1: 数据探索与变量分析

### **Step 1.1: 识别可用的 Categorical Variables**
**目标**: 找出数据集中所有可以用于人格推断的分类变量

**操作**:
```bash
python3 step1_check_features.py
```

**输出**:
- `available_features.txt`: 10个可用变量列表
- 覆盖率统计
- 唯一值数量

**结果**: ✅ 已完成
- term (2 unique values)
- grade (7), sub_grade (34)
- emp_length (11)
- home_ownership (3)
- verification_status (3)
- purpose (13)
- application_type (1)

---

### **Step 1.2: 分析每个 Variable 与 Target (违约率) 的关系**
**目标**: 理解哪些变量与信贷风险相关（为后续 OCEAN 映射提供依据）

**操作**: 运行以下脚本

**输出**:
- 每个 variable 的违约率分布
- 卡方检验 p-value（是否与 target 显著相关）
- 可视化：柱状图对比

**示例发现**:
```
Variable: grade
  Grade A → 违约率 8%
  Grade G → 违约率 35%
  卡方检验: p < 0.001 (显著相关)

Variable: purpose
  debt_consolidation → 违约率 15%
  small_business → 违约率 22%
  wedding → 违约率 10%
```

---

### **Step 1.3: 建立 Variable → OCEAN 的心理学映射假设**
**目标**: 基于心理学理论，定义每个变量如何影响 OCEAN 维度

**方法**: 文献调研 + 专家判断

**映射表（示例）**:

| Variable | 影响的 OCEAN 维度 | 方向 | 理由 |
|----------|------------------|------|------|
| **purpose** | | | |
| - small_business | Openness ↑ | + | 创业需要创新精神 |
| - debt_consolidation | Conscientiousness ↑ | + | 主动管理债务 = 负责任 |
| - wedding | Agreeableness ↑ | + | 社交导向的消费 |
| **term** | | | |
| - 60 months | Conscientiousness ↓ | - | 长期债务 = 缺乏规划 |
| - 36 months | Conscientiousness ↑ | + | 短期 = 谨慎 |
| **grade** | | | |
| - A-C (好信用) | Conscientiousness ↑ | + | 良好信用 = 负责任 |
| - D-G (差信用) | Neuroticism ↑ | + | 信用差 = 情绪不稳定/焦虑 |
| **home_ownership** | | | |
| - OWN | Conscientiousness ↑ | + | 拥有房产 = 稳定/规划好 |
| - RENT | Neuroticism ↑ | + | 租房 = 不稳定 |
| **emp_length** | | | |
| - 10+ years | Conscientiousness ↑ | + | 长期就业 = 稳定/负责 |
| - < 1 year | Neuroticism ↑ | + | 短期 = 不稳定 |
| **application_type** | | | |
| - Joint | Extraversion ↑ | + | 联合申请 = 社交性强 |
| - Individual | Extraversion ↓ | - | 独自申请 = 内向 |

**输出**: `ocean_mapping_hypothesis.csv`

---

## 📊 Phase 2: 权重设计与验证

### **Step 2.1: 为每个 Variable 设计权重系数**
**目标**: 量化每个变量对 OCEAN 的贡献度

**方法 A: 基于专家判断（简单版）**
```python
# 示例权重矩阵
weights = {
    "purpose": {
        "small_business": {"openness": +0.3, "conscientiousness": +0.2},
        "debt_consolidation": {"conscientiousness": +0.25},
        "wedding": {"agreeableness": +0.2}
    },
    "grade": {
        "A": {"conscientiousness": +0.3, "neuroticism": -0.2},
        "B": {"conscientiousness": +0.2, "neuroticism": -0.1},
        "C": {"conscientiousness": 0, "neuroticism": 0},
        "D": {"conscientiousness": -0.1, "neuroticism": +0.1},
        "E": {"conscientiousness": -0.2, "neuroticism": +0.2},
        "F": {"conscientiousness": -0.3, "neuroticism": +0.25},
        "G": {"conscientiousness": -0.4, "neuroticism": +0.3}
    }
}
```

**方法 B: 基于数据驱动（高级版）**
- 使用逻辑回归学习权重
- 输入: categorical variables (one-hot)
- 输出: OCEAN 分数（通过 LLM 标注的子集训练）

---

### **Step 2.2: 验证权重的合理性**
**目标**: 检查生成的 OCEAN 分数是否符合心理学预期

**测试用例**:
```python
# 测试 1: 信用等级高的人应该有更高的 Conscientiousness
borrower_A = {"grade": "A", "purpose": "debt_consolidation"}
borrower_G = {"grade": "G", "purpose": "credit_card"}

ocean_A = compute_ocean(borrower_A)
ocean_G = compute_ocean(borrower_G)

assert ocean_A["conscientiousness"] > ocean_G["conscientiousness"]
```

**验证方法**:
1. **单变量测试**: 只改变一个变量，观察 OCEAN 变化
2. **极端案例**: 最好/最坏组合，检查 OCEAN 是否合理
3. **与 LLM 对比**: 随机抽 100 条，对比你的算法 vs LLM 打分的相关性

---

### **Step 2.3: 调整权重（迭代优化）**
**目标**: 基于验证结果微调权重

**优化目标**:
- 最大化 OCEAN 与违约率的相关性
- 或：最大化与 LLM baseline 的一致性

**方法**:
```python
# Grid search 权重系数
for purpose_weight in [0.1, 0.2, 0.3]:
    for grade_weight in [0.2, 0.3, 0.4]:
        ocean_scores = compute_ocean_with_weights(df, purpose_weight, grade_weight)
        correlation = compute_correlation(ocean_scores, df["target"])
        # 选择最高相关性的权重组合
```

---

## 🔧 Phase 3: 构建打分算法

### **Step 3.1: 实现基于规则的打分函数**
**目标**: 把权重矩阵转换为可执行代码

**伪代码**:
```python
def compute_ocean_rule_based(borrower):
    # 初始化 OCEAN (baseline = 0.5)
    ocean = {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5
    }

    # 根据 purpose 调整
    if borrower["purpose"] == "small_business":
        ocean["openness"] += 0.15
        ocean["conscientiousness"] += 0.10
    elif borrower["purpose"] == "debt_consolidation":
        ocean["conscientiousness"] += 0.12

    # 根据 grade 调整
    grade_map = {"A": 0.3, "B": 0.2, "C": 0, "D": -0.1, "E": -0.2, "F": -0.3, "G": -0.4}
    if borrower["grade"] in grade_map:
        ocean["conscientiousness"] += grade_map[borrower["grade"]]
        ocean["neuroticism"] -= grade_map[borrower["grade"]]  # 反向

    # 根据 emp_length 调整
    if borrower["emp_length"] == "10+ years":
        ocean["conscientiousness"] += 0.10
    elif borrower["emp_length"] == "< 1 year":
        ocean["neuroticism"] += 0.15

    # 根据 home_ownership 调整
    if borrower["home_ownership"] == "OWN":
        ocean["conscientiousness"] += 0.08
    elif borrower["home_ownership"] == "RENT":
        ocean["neuroticism"] += 0.05

    # 归一化到 [0.25, 0.75] 范围（避免极端值）
    for key in ocean:
        ocean[key] = max(0.25, min(0.75, ocean[key]))

    return ocean
```

---

### **Step 3.2: 对比 Rule-Based vs LLM-Based**
**目标**: 评估自定义算法的质量

**实验设计**:
1. 随机抽 500 条样本
2. 分别用 Rule-Based 和 LLM 打分
3. 计算相关系数（每个 OCEAN 维度）

**期望结果**:
```
Dimension           Correlation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Openness            0.45 - 0.65
Conscientiousness   0.50 - 0.70
Extraversion        0.30 - 0.50
Agreeableness       0.35 - 0.55
Neuroticism         0.40 - 0.60
```

如果相关性 > 0.5，说明 Rule-Based 已经捕捉到主要模式。

---

## 🎯 Phase 4: 集成到模型

### **Step 4.1: 批量生成 OCEAN 特征**
（这一步你已经做过了）
```bash
python3 run_full_pipeline.py
```

### **Step 4.2: 特征重要性分析**
**目标**: 检查 OCEAN 特征在模型中的作用

**XGBoost Feature Importance**:
```python
importances = model.named_steps['clf'].feature_importances_
feature_names = model.named_steps['preprocess'].get_feature_names_out()

# 筛选 OCEAN 特征
ocean_importance = [(name, imp) for name, imp in zip(feature_names, importances)
                    if any(dim in name for dim in OCEAN_DIMS)]

print("OCEAN Features Ranking:")
for name, imp in sorted(ocean_importance, key=lambda x: x[1], reverse=True):
    print(f"  {name:30s}: {imp:.4f}")
```

### **Step 4.3: 如果 OCEAN 重要性很低，调整权重**
**迭代循环**:
1. 增加某个维度的权重（如 Conscientiousness × 1.5）
2. 重新打分
3. 重新训练模型
4. 观察 Feature Importance 是否提升
5. 重复直到 OCEAN 进入 Top 20 特征

---

## 📈 Phase 5: 最终评估

### **Step 5.1: A/B 对比**
（已完成）

### **Step 5.2: 如果性能提升不明显，诊断原因**

**可能原因**:
1. **权重不合理** → 回到 Step 2.3 调整
2. **变量信息不足** → 需要更多文本数据
3. **OCEAN 与违约无关** → 理论假设错误（需要文献支持）
4. **样本量太小** → 扩大到 50k+ 样本

**诊断方法**:
```python
# 检查 OCEAN 与 target 的相关性
for dim in OCEAN_DIMS:
    corr = df[dim].corr(df["target"])
    print(f"{dim:20s}: r = {corr:.3f}")

# 如果所有相关性都 < 0.05，说明 OCEAN 对违约预测无用
```

---

## 🔄 总结：完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: 数据探索                                            │
├─────────────────────────────────────────────────────────────┤
│ 1.1 识别可用变量 (10个)                        ✅ 已完成     │
│ 1.2 分析变量与违约率的关系                     🔄 待实现     │
│ 1.3 建立 Variable → OCEAN 映射假设              🔄 待实现     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: 权重设计                                            │
├─────────────────────────────────────────────────────────────┤
│ 2.1 设计权重系数矩阵                           🔄 待实现     │
│ 2.2 验证权重合理性（单元测试）                 🔄 待实现     │
│ 2.3 迭代优化权重                               🔄 待实现     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: 算法实现                                            │
├─────────────────────────────────────────────────────────────┤
│ 3.1 实现 Rule-Based 打分函数                   🔄 待实现     │
│ 3.2 对比 Rule vs LLM (相关性分析)              🔄 待实现     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: 模型集成                                            │
├─────────────────────────────────────────────────────────────┤
│ 4.1 批量生成 OCEAN 特征                        ✅ 已完成     │
│ 4.2 特征重要性分析                             ✅ 已完成     │
│ 4.3 权重迭代（如果重要性低）                   🔄 待实现     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 5: 评估与诊断                                          │
├─────────────────────────────────────────────────────────────┤
│ 5.1 A/B 对比                                   ✅ 已完成     │
│ 5.2 诊断失败原因                               🔄 待实现     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 当前状态 vs 应该做的

**你已经做的**:
- ✅ Step 1.1: 识别变量
- ✅ Step 3.1: 实现打分函数（但用的是随机哈希，不是规则）
- ✅ Step 4.1: 批量打分
- ✅ Step 4.2: 模型训练
- ✅ Step 5.1: A/B 对比

**你跳过的关键步骤**:
- ⚠️ **Step 1.2**: 分析变量与违约率关系
- ⚠️ **Step 1.3**: 建立心理学映射假设
- ⚠️ **Step 2.1-2.3**: 设计和验证权重
- ⚠️ **Step 3.2**: Rule vs LLM 对比

**为什么性能没提升**:
因为你用的是**随机哈希**（deterministic fallback），而不是**基于心理学理论的规则**。

---

## 🚀 下一步行动建议

### **Option 1: 实现 Rule-Based 打分（推荐）**
我帮你实现 Step 2.1 + 3.1 的完整代码：
- 基于心理学文献设计权重
- 编写规则函数
- 替换当前的哈希方法

### **Option 2: 启用 LLM API**
直接用 OpenAI 打分（跳过权重设计）

### **Option 3: 混合方法**
- Rule-Based 作为 baseline
- LLM 打分 500 条样本
- 调整 Rule 权重使其与 LLM 对齐

**你想要哪个？我现在就帮你实现。**
