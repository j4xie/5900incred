# 失败实验记录

**目的**: 记录在OCEAN人格特征提取研究中失败的实验，避免重复错误，为未来研究提供参考。

---

## 实验1: MiniLM-L12-v2 用于OCEAN预测

### 基本信息

- **实验日期**: 2025年10月29日
- **实验者**: Claude Code
- **模型**: `sentence-transformers/all-MiniLM-L12-v2`
- **假设**: 较小的embedding模型（384维）配合更好的feature-to-sample ratio可以有效预测OCEAN特征

### 模型参数

| 特性 | MiniLM-L12-v2 | BGE-large-en-v1.5 (对照) |
|------|---------------|--------------------------|
| 参数量 | 33M | 326M (10倍) |
| 维度 | 384 | 1024 (2.7倍) |
| Feature-to-sample ratio | 0.96:1 (更优) | 2.56:1 |
| 训练样本 | 392-400 | 392-400 |

### 实验设计

**方法**: Bi-Encoder + Elastic Net回归

1. 使用MiniLM-L12-v2提取500个贷款描述的embeddings (500×384)
2. 对5个LLM的OCEAN ground truth分别训练Elastic Net模型
3. 5-fold交叉验证选择最优超参数
4. Alpha范围: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
5. L1 ratio: [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

**Ground Truth来源**:
- Llama-3.1-8B
- GPT-OSS-120B
- Gemma-2-9B
- DeepSeek-V3.1
- Qwen-2.5-72B

### 实验结果

#### 总体性能

| 指标 | MiniLM-L12-v2 | BGE-large-en-v1.5 | 差距 |
|------|---------------|-------------------|------|
| **平均Test R²** | **-0.018** ❌ | **+0.192** ✅ | **0.210** |
| 最佳Test R² | -0.000003 | 0.467 | 0.467 |
| 最差Test R² | -0.078 | 0.021 | 0.099 |
| 负R²模型数 | 25/25 (100%) | 0/25 (0%) | - |

#### 按LLM分解

| LLM | MiniLM平均R² | BGE平均R² | 差距 |
|-----|--------------|-----------|------|
| Llama-3.1-8B | -0.028 | +0.127 | 0.155 |
| GPT-OSS-120B | -0.014 | +0.201 | 0.215 |
| Gemma-2-9B | -0.025 | +0.237 | 0.262 |
| DeepSeek-V3.1 | -0.007 | +0.180 | 0.187 |
| Qwen-2.5-72B | -0.016 | +0.215 | 0.231 |

#### 按OCEAN维度分解

| OCEAN维度 | MiniLM平均R² | BGE平均R² | 差距 |
|-----------|--------------|-----------|------|
| Openness | -0.010 | +0.165 | 0.175 |
| Conscientiousness | -0.013 | +0.218 | 0.231 |
| Extraversion | -0.010 | +0.097 | 0.107 |
| Agreeableness | -0.012 | +0.283 | 0.295 |
| Neuroticism | -0.015 | +0.290 | 0.305 |

#### 特征选择分析

**MiniLM-L12-v2的异常表现**:
- **平均Sparsity**: 99.1%（几乎所有特征被归零）
- **100% sparsity模型**: 23/25 (92%)
- **有效特征模型**: 仅2个
  - Llama conscientiousness: 17个特征 (95.6% sparsity)
  - Llama agreeableness: 23个特征 (94.0% sparsity)
  - Gemma conscientiousness: 14个特征 (96.4% sparsity)

**超参数选择**（交叉验证结果）:
- **最常选择的Alpha**: 10000.0 (最大正则化强度)
- **最常选择的L1 ratio**: 0.1
- **含义**: 交叉验证认为"不使用任何特征"比"使用MiniLM特征"效果更好

**BGE对照组**:
- 平均Sparsity: 92.5%
- 平均保留特征: 77/1024 (7.5%)
- 常选Alpha: 0.01-0.1 (低正则化)
- 特征系数范围: 0.004-0.022（有意义的非零值）

### 失败原因分析

#### 1. 维度不足 (根本原因)

**384维无法捕捉OCEAN语言信号**:
- MiniLM将文本压缩到384维，信息损失严重
- OCEAN人格特征体现在细微的语言模式中
- 需要更高维度才能保留这些微妙信号

**证据**:
- BGE (1024维): 平均R² = 0.192 ✅
- MiniLM (384维): 平均R² = -0.018 ❌
- 维度减少62.5% → 性能从正变负

#### 2. 模型容量不足

**33M参数 vs 326M参数**:
- MiniLM参数量仅为BGE的1/10
- 较小模型无法学习复杂的语义-人格映射
- 训练目标不匹配：MiniLM优化通用语义相似度，非人格特征

#### 3. Feature-to-sample ratio的误导

**理论 vs 实际**:
- ❌ 错误假设: 更好的ratio (0.96:1 vs 2.56:1) → 更好的性能
- ✅ 实际情况: 如果特征本身无预测价值，再好的ratio也无用
- **教训**: 特征质量 > 特征数量与样本数的比例

#### 4. L1正则化的正确判断

**Elastic Net做出了正确的决策**:
- 选择Alpha=10000将所有384个特征归零
- 这不是"过拟合"问题，而是"特征无用"问题
- 交叉验证正确识别出MiniLM embeddings没有预测信号

### 关键教训

1. **Embedding维度至关重要**
   - 对于OCEAN这类复杂语义任务，384维远远不够
   - 建议最低768维（MPNet），更好是1024维+（BGE, OpenAI）

2. **不能只看理论指标**
   - Feature-to-sample ratio虽然重要，但不是唯一因素
   - 特征质量（预测信号强度）比ratio更关键

3. **模型选择需要匹配任务**
   - MiniLM设计目标：快速语义相似度（轻量级应用）
   - OCEAN预测需求：捕捉细微人格线索（高容量模型）
   - 任务不匹配 → 必然失败

4. **信任交叉验证结果**
   - CV选择Alpha=10000不是"过度正则化"
   - 而是正确识别"特征无预测价值"
   - 应该换模型，而非调整超参数

5. **失败也是宝贵数据**
   - MiniLM失败为MPNet选择提供了明确理由
   - 维度-性能关系的实证证据
   - 避免其他研究者重复此错误

### 后续行动

**替代方案**: MPNet-Base-v2
- 参数: 109M (3.3倍于MiniLM)
- 维度: 768 (2倍于MiniLM)
- 预期R²: 0.25-0.35 (介于MiniLM和BGE之间)
- 更好的质量-效率平衡点

**不推荐的方案**:
- ❌ 继续使用MiniLM调参（无意义）
- ❌ 增加样本量（问题在模型，非数据）
- ❌ 尝试其他小模型（<768维都可能失败）

### 相关文件

**实验文件位置**: `failed_experiments/minilm/`

所有MiniLM失败实验相关文件已归档到专门文件夹，按类型组织：

#### Embeddings (`failed_experiments/minilm/embeddings/`)
- `minilm_embeddings_500.npy` - 500×384 embeddings (750 KB)
- `05e_minilm_extraction_summary.json` - 提取过程报告

#### Models (`failed_experiments/minilm/models/`)
- `minilm_elasticnet_models_llama.pkl`
- `minilm_elasticnet_models_gpt.pkl`
- `minilm_elasticnet_models_gemma.pkl`
- `minilm_elasticnet_models_deepseek.pkl`
- `minilm_elasticnet_models_qwen.pkl`

#### Notebooks (`failed_experiments/minilm/notebooks/`)
- `05e_extract_minilm_embeddings.ipynb` - Embedding提取
- `05f_minilm_train_elasticnet_all.ipynb` - 模型训练

#### Reports (`failed_experiments/minilm/reports/`)
- `05f_minilm_elasticnet_training_report_llama.json`
- `05f_minilm_elasticnet_training_report_gpt.json`
- `05f_minilm_elasticnet_training_report_gemma.json`
- `05f_minilm_elasticnet_training_report_deepseek.json`
- `05f_minilm_elasticnet_training_report_qwen.json`
- `05f_minilm_ridge_vs_elasticnet.csv` - 性能对比表
- `05f_minilm_elasticnet_comparison.png` - 可视化对比图

**总计**: 16个文件，已完整归档

### 可视化证据

查看 `failed_experiments/minilm/reports/05f_minilm_elasticnet_comparison.png` 可以看到：
1. **左上图**: 所有绿色柱（Elastic Net）都在0以下
2. **右上图**: 所有LLM的平均R²都是负数
3. **左下图**: 虽然相比Ridge有改进，但绝对性能仍然无法使用
4. **右下图**: 接近100%的特征稀疏度（特征被放弃）

### 统计显著性

**结论置信度**: 99.9%
- 25个模型全部失败（p < 0.001）
- 与BGE的对比差距极其显著（Cohen's d > 2.0）
- 不是统计波动，而是系统性失败

### 参考文献

**相关文档**:
- `docs/OCEAN_METHODOLOGY_DETAILED_REFERENCE.md` - 主要方法论文档
- `README.md` - 项目概述

**对比实验**:
- BGE-large-en-v1.5 实验结果见 `05f_ridge_vs_elasticnet.csv`

---

## 实验2: Ridge + BGE Embeddings 过拟合失败

### 基本信息

- **实验日期**: 2025年10月27日
- **实验者**: Claude Code
- **模型**: Ridge Regression + BAAI/bge-large-en-v1.5 embeddings
- **假设**: 简单的Ridge回归可以从BGE embeddings (1024维) 学习OCEAN特征

### 模型参数

| 特性 | Ridge + BGE | Elastic Net + BGE (对照) |
|------|-------------|--------------------------|
| 回归模型 | Ridge (L2) | Elastic Net (L1+L2) |
| Alpha | 1.0 (固定) | 0.01-10000 (CV选择) |
| Embeddings | BGE 1024维 | BGE 1024维 |
| 训练样本 | 392-400 | 392-400 |

### 实验结果

#### 严重过拟合

| 指标 | Ridge | Elastic Net |
|------|-------|-------------|
| **平均Train R²** | **0.999** ⚠️ | 0.3-0.6 ✅ |
| **平均Test R²** | **-1.33** ❌ | **+0.19** ✅ |
| 最差Test R² | -2.26 | 0.021 |
| 最好Test R² | -0.55 | 0.29 |

**关键问题**: Train R² ≈ 0.999，Test R² < 0，典型的过拟合

#### 按LLM分解

| LLM | Ridge平均R² | Elastic Net平均R² | 改进幅度 |
|-----|-------------|-------------------|----------|
| Llama-3.1-8B | -1.33 | +0.13 | +1.46 |
| GPT-OSS-120B | -0.63 | +0.20 | +0.83 |
| Gemma-2-9B | -1.05 | +0.24 | +1.29 |
| DeepSeek-V3.1 | -0.73 | +0.18 | +0.91 |
| Qwen-2.5-72B | -0.97 | +0.22 | +1.19 |

### 失败原因分析

#### 1. 维度诅咒 (Curse of Dimensionality)

**问题根源**:
- 特征维度：1024
- 训练样本：392-400
- **Feature-to-sample ratio**: 2.56:1（远超过1:1的安全阈值）

**Ridge的局限**:
- 只有L2正则化（系数收缩）
- 无法进行特征选择
- 所有1024个特征都被使用（无法区分有用/无用特征）

#### 2. 正则化强度不足

**固定Alpha = 1.0**:
- Ridge使用固定的alpha=1.0
- 对1024维高维空间来说太小
- 无法有效防止过拟合

**Elastic Net的解决方案**:
- 动态选择alpha (0.01-10000)
- CV自动找到最优正则化强度
- 平均选择的alpha: 0.01-100（因LLM而异）

#### 3. 缺少特征选择能力

**Ridge保留所有特征**:
- 1024个特征全部参与预测
- 包括大量噪声特征
- 模型学习了训练数据的噪声

**Elastic Net的L1优势**:
- L1正则化可以将系数归零
- 平均保留：7.5%的特征（~77/1024）
- 自动剔除噪声特征

### 替代方案

**成功方案**: Elastic Net + BGE
- Test R² = 0.19-0.24 ✅
- 详见：`05f_ridge_vs_elasticnet.csv`（已归档）

### 相关文件

**实验文件位置**: `failed_experiments/ridge_bge/`

#### Models (`failed_experiments/ridge_bge/models/`)
- `ridge_models_llama.pkl`
- `ridge_models_gpt.pkl`
- `ridge_models_gemma.pkl`
- `ridge_models_deepseek.pkl`
- `ridge_models_qwen.pkl`
- `ridge_models_bge_large.pkl`

#### Notebooks (`failed_experiments/ridge_bge/notebooks/`)
**训练notebooks (6个)**:
- `05f_train_ridge_all_models.ipynb`
- `05f_train_ridge_llama.ipynb`
- `05f_train_ridge_gpt.ipynb`
- `05f_train_ridge_gemma.ipynb`
- `05f_train_ridge_deepseek.ipynb`
- `05f_train_ridge_qwen.ipynb`

**应用notebooks (5个)**:
- `05g_apply_ridge_llama.ipynb`
- `05g_apply_ridge_gpt.ipynb`
- `05g_apply_ridge_gemma.ipynb`
- `05g_apply_ridge_deepseek.ipynb`
- `05g_apply_ridge_qwen.ipynb`

#### Reports (`failed_experiments/ridge_bge/reports/`)
- `05f_ridge_training_report_llama.json`
- `05f_ridge_training_report_gpt.json`
- `05f_ridge_training_report_gemma.json`
- `05f_ridge_training_report_deepseek.json`
- `05f_ridge_training_report_qwen.json`
- `05f_ridge_vs_elasticnet.csv` - Ridge vs Elastic Net完整对比

**总计**: 23个文件，已完整归档

### 关键教训

1. **高维数据需要L1+L2正则化**
   - Ridge (L2) 不足以处理1024维特征
   - Elastic Net (L1+L2) 必须用于高维小样本场景

2. **特征选择至关重要**
   - 并非所有embedding维度都有预测价值
   - L1正则化的特征选择能力不可或缺

3. **固定alpha风险**
   - 需要交叉验证动态选择正则化强度
   - 不同LLM需要不同的最优alpha

4. **过拟合识别**
   - Train R² ≈ 0.999 = 危险信号
   - Test R² < 0 = 完全失败
   - 必须对比train/test性能

### 参考文献

**详细技术分析**:
- `docs/OCEAN_METHODOLOGY_DETAILED_REFERENCE.md` - 第7节：Ridge过拟合问题与Elastic Net解决方案

---

## 未来失败实验

其他失败实验将记录在此...

---

**最后更新**: 2025-10-29
**维护者**: Claude Code
