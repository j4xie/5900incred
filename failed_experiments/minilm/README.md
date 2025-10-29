# MiniLM-L12-v2 失败实验归档

## 实验概要

- **实验日期**: 2025年10月29日
- **模型**: sentence-transformers/all-MiniLM-L12-v2
- **参数量**: 33M
- **Embedding维度**: 384
- **实验目的**: 测试MiniLM能否用于OCEAN人格特征预测

## 结果

❌ **实验失败** - 完全无预测能力

- **Test R²**: -0.078 到 -0.000003（全部负数）
- **25个模型**: 100%失败
- **特征稀疏性**: 99.1%（几乎所有特征被归零）
- **交叉验证结果**: 选择最大正则化（Alpha=10000），说明特征无价值

## 失败原因

1. **维度太低** (384d)
   - 无法捕捉OCEAN相关的细微语言信号
   - 信息压缩过度导致预测能力丧失

2. **模型容量不足** (33M vs BGE 326M)
   - 参数量仅为BGE的1/10
   - 无法学习复杂的语义-人格映射

3. **与BGE对比**
   - MiniLM平均R² = -0.018 ❌
   - BGE平均R² = +0.192 ✅
   - **性能差距**: 0.21 R²（天壤之别）

## 关键教训

> **Embedding维度对OCEAN预测至关重要！**
>
> 不能只看feature-to-sample ratio（MiniLM 0.96:1 vs BGE 2.56:1）
>
> 特征质量（预测信号强度）比ratio更关键

## 文件结构

```
failed_experiments/minilm/
├── embeddings/          # 500×384 embeddings
│   ├── minilm_embeddings_500.npy
│   └── 05e_minilm_extraction_summary.json
├── models/              # 5个LLM的Elastic Net模型
│   ├── minilm_elasticnet_models_llama.pkl
│   ├── minilm_elasticnet_models_gpt.pkl
│   ├── minilm_elasticnet_models_gemma.pkl
│   ├── minilm_elasticnet_models_deepseek.pkl
│   └── minilm_elasticnet_models_qwen.pkl
├── notebooks/           # 提取和训练notebooks
│   ├── 05e_extract_minilm_embeddings.ipynb
│   └── 05f_minilm_train_elasticnet_all.ipynb
└── reports/             # 训练报告和对比
    ├── 05f_minilm_elasticnet_training_report_*.json (5个)
    ├── 05f_minilm_ridge_vs_elasticnet.csv
    └── 05f_minilm_elasticnet_comparison.png
```

**总计**: 16个文件

## 替代方案

已改用 **MPNet-Base-v2**：
- 参数: 109M (3.3倍于MiniLM)
- 维度: 768 (2倍于MiniLM)
- 预期R²: 0.25-0.35

## 详细分析

完整的失败实验分析请查看：
📄 **[../../docs/FAILED_EXPERIMENTS.md](../../docs/FAILED_EXPERIMENTS.md)**

---

**归档日期**: 2025-10-29
**维护者**: Claude Code
