# Ridge + BGE Embeddings 过拟合失败实验归档

## 实验概要

- **实验日期**: 2025年10月
- **模型**: Ridge Regression + BGE-large-en-v1.5 Embeddings
- **特征维度**: 1024 (BGE embeddings)
- **样本数**: 400 (训练集)
- **实验目的**: 测试Ridge回归能否用于OCEAN人格特征预测

## 结果

❌ **实验失败** - 严重过拟合，无泛化能力

- **Train R²**: 0.999 (完美拟合训练集)
- **Test R²**: -1.33 到 -0.03（全部负数）
- **30个模型**: 100%过拟合
- **交叉验证**: Ridge无法通过正则化防止过拟合

### 典型结果示例

| LLM | 维度 | Train R² | Test R² | 过拟合程度 |
|-----|------|----------|---------|-----------|
| GPT-4o | Openness | 0.999 | -0.034 | 极严重 |
| Llama | Extraversion | 0.999 | -1.329 | 灾难性 |
| DeepSeek | Neuroticism | 0.999 | -0.208 | 极严重 |

## 失败原因

### 1. **维度诅咒 (Curse of Dimensionality)**

- **特征维度**: 1024
- **训练样本**: 400
- **Feature-to-Sample比**: 2.56:1
- **问题**: 特征数远超样本数，模型记忆而非学习

### 2. **Ridge缺乏特征选择能力**

**Ridge (L2正则化)**:
```
Loss = MSE + α × Σ(βᵢ²)
```
- 只缩小系数，不归零
- 无法自动排除无关特征
- 所有1024维都参与预测

**Elastic Net (L1+L2正则化)**:
```
Loss = MSE + α × [ρ × Σ|βᵢ| + (1-ρ) × Σ(βᵢ²)]
```
- L1项可将系数归零
- 自动特征选择
- 典型稀疏性: 99.1%

### 3. **与Elastic Net对比**

| 方法 | Train R² | Test R² | 特征稀疏性 | 泛化能力 |
|------|----------|---------|-----------|----------|
| Ridge | 0.999 | **-1.33 ~ -0.03** | 0% | ❌ 无 |
| Elastic Net | 0.45-0.70 | **0.19-0.24** | 99.1% | ✅ 良好 |

**性能差距**: 平均1.5+ R²（Ridge完全失败）

## 关键教训

> **高维小样本场景必须使用L1正则化！**
>
> Ridge回归不适合高维embedding特征（1024d）
>
> Elastic Net的特征选择能力对OCEAN预测至关重要

### 何时使用Ridge vs Elastic Net

**Ridge适用场景**:
- 特征数 << 样本数
- 所有特征都相关
- 多重共线性问题

**Elastic Net适用场景** ✅:
- 特征数 ≥ 样本数（高维数据）
- 许多特征无关（需特征选择）
- 需要稀疏模型

**本项目场景**: 1024维 vs 400样本 → **必须用Elastic Net**

## 文件结构

```
failed_experiments/ridge_bge/
├── models/                # 6个Ridge模型文件
│   ├── ridge_models_bge_large.pkl
│   ├── ridge_models_deepseek.pkl
│   ├── ridge_models_gemma.pkl
│   ├── ridge_models_gpt.pkl
│   ├── ridge_models_llama.pkl
│   └── ridge_models_qwen.pkl
├── notebooks/             # 11个训练和应用notebooks
│   ├── 05f_ridge_train_bge_deepseek.ipynb
│   ├── 05f_ridge_train_bge_gemma.ipynb
│   ├── 05f_ridge_train_bge_gpt.ipynb
│   ├── 05f_ridge_train_bge_llama.ipynb
│   ├── 05f_ridge_train_bge_qwen.ipynb
│   ├── 05f_ridge_train_bge_weighted.ipynb
│   ├── 05g_apply_ridge_deepseek.ipynb
│   ├── 05g_apply_ridge_gemma.ipynb
│   ├── 05g_apply_ridge_gpt.ipynb
│   ├── 05g_apply_ridge_llama.ipynb
│   └── 05g_apply_ridge_qwen.ipynb
└── reports/               # 6个训练报告
    ├── 05f_ridge_training_report_deepseek.json
    ├── 05f_ridge_training_report_gemma.json
    ├── 05f_ridge_training_report_gpt.json
    ├── 05f_ridge_training_report_llama.json
    ├── 05f_ridge_training_report_qwen.json
    └── 05f_ridge_vs_elasticnet.csv
```

**总计**: 23个文件

## 注意事项

⚠️ **Ridge-Weighted方法未归档**

本次归档仅包含：
- ❌ **Ridge + BGE Embeddings**（路线2A失败版本）

**不包含**（仍在使用）:
- ✅ **Ridge-Weighted方法**（路线1，R²=0.15-0.20）
  - 使用word/phrase/sentence/bigram特征
  - 特征维度较低，Ridge表现良好

## 后续方案

已改用 **Elastic Net + BGE**：
- 特征选择: L1正则化自动筛选
- 稀疏性: 99.1%（仅8-9个特征/模型）
- Test R²: 0.19-0.24 ✅
- 泛化能力: 良好

## 详细分析

完整的失败实验分析请查看：
📄 **[../../docs/FAILED_EXPERIMENTS.md](../../docs/FAILED_EXPERIMENTS.md)**

---

**归档日期**: 2025-10-29
**维护者**: Claude Code
