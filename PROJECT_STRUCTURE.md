# Project Structure Guide

## 📁 Directory Organization

```
credibly-info-5900/
│
├── 📂 data/                          # 数据文件
│   ├── loan.csv                      # 原始数据集 (2.26M 行)
│   ├── loan_with_desc.csv            # 含 desc 字段的数据
│   ├── loan_clean_for_modeling.csv   # 清洁建模数据集 (514K 行 × 36 特征)
│   └── loan_clean_with_ocean.csv     # 含 OCEAN 特征的数据
│
├── 📂 notebooks/                     # Jupyter Notebooks（按阶段组织）
│   ├── 01_data_preparation/
│   │   ├── 01_data_cleaning_with_desc.ipynb
│   │   ├── 02_feature_selection_and_leakage_check.ipynb
│   │   ├── 03_create_modeling_dataset.ipynb
│   │   └── view_loan_data.ipynb
│   │
│   ├── 02_feature_engineering/
│   │   ├── 05_ocean_feature_extraction.ipynb
│   │   ├── 05a_llm_ocean_ground_truth.ipynb      [新]
│   │   ├── 05b_train_ocean_ridge_weights.ipynb   [新]
│   │   └── 05c_apply_ocean_to_all.ipynb          [新]
│   │
│   ├── 03_modeling/
│   │   ├── 04_xgboost_baseline.ipynb             ⚡ [已优化]
│   │   ├── 06_xgboost_with_ocean.ipynb           ⚡ [已优化]
│   │   └── baseline_model_evaluation.png
│   │
│   └── 04_results_analysis/
│       └── 07_results_analysis.ipynb
│
├── 📂 docs/                          # 📋 文档和报告（新增）
│   ├── README.md                     # 项目总述
│   ├── PROJECT_EXECUTION_ROADMAP.md  # 执行路线图
│   ├── PREPROCESSING_OPTIMIZATION.md # 预处理优化总结
│   ├── OPTIMIZATION_EXPLANATION.md   # 如何向客户解释优化
│   ├── OPTIMIZATION_DETAILED_BREAKDOWN.md  # 优化细节分解
│   ├── GROUND_TRUTH_STATUS.md        # Ground Truth 状态报告
│   ├── OCEAN_METHODOLOGY_FINAL.md    # OCEAN 方法论
│   ├── OCEAN_IMPROVEMENT_ANALYSIS.md # OCEAN 改进分析
│   ├── FINAL_PROJECT_REPORT.md       # 最终项目报告
│   ├── requirements.txt              # Python 依赖
│   ├── GENAI_OCEAN_COMPLETE_REPORT_EN.txt
│   └── ... 其他文档
│
├── 📂 results/                       # 🎯 所有输出结果（新增）
│   ├── 📊 CSV Files (数据和指标)
│   │   ├── baseline_metrics.json     # 基线模型性能
│   │   ├── full_model_metrics.json   # 完整模型性能
│   │   ├── model_comparison.csv      # 模型对比结果
│   │   ├── baseline_feature_importance.csv
│   │   ├── full_model_feature_importance.csv
│   │   ├── ocean_ground_truth_500.csv
│   │   ├── ocean_features.csv
│   │   ├── ocean_weights_coefficients.csv
│   │   ├── feature_coverage_report.csv
│   │   ├── retained_features_examples.csv
│   │   └── ... 其他结果文件
│   │
│   ├── 📈 PNG Files (图表)
│   │   ├── baseline_model_evaluation.png
│   │   ├── full_model_evaluation.png
│   │   ├── data_cleaning_summary.png
│   │   ├── ocean_weights_visualization.png
│   │   ├── ocean_features_distribution.png
│   │   └── ... 其他图表
│   │
│   └── 📝 Config Files (配置)
│       ├── baseline_feature_config.json
│       ├── full_model_feature_config.json
│       ├── feature_lists_clean.json
│       └── ocean_weights_formula.json
│
├── 📂 models/                        # 🤖 模型文件（新增）
│   ├── saved_models/
│   │   ├── xgboost_baseline_model.pkl
│   │   ├── xgboost_full_model.pkl
│   │   ├── xgboost_ocean_model.pkl
│   │   └── ocean_ridge_models.pkl
│   │
│   └── preprocessors/
│       ├── preprocessor_baseline.pkl
│       ├── preprocessor_full.pkl
│       └── preprocessor_*.pkl
│
├── 📂 scripts/                       # 🔧 Python 执行脚本（新增）
│   ├── run_ocean_pipeline.py         # 运行 OCEAN 管线
│   ├── run_xgboost_comparison.py     # 运行 XGBoost 对比
│   ├── train_xgboost_models.py       # 训练 XGBoost 模型
│   ├── execute_pipeline.py           # 执行完整管线
│   ├── execute_05b_with_new_ground_truth.py
│   ├── execute_05c_with_new_weights.py
│   └── ... 其他脚本
│
├── 📂 logs/                          # 📝 日志文件（新增）
│   ├── xgboost_execution.log
│   ├── ocean_execution.log
│   ├── xgboost_training.log
│   ├── pipeline_execution.log
│   ├── regenerate_ground_truth.log
│   └── regenerate_ground_truth_final.log
│
├── 📂 artifacts/                     # 缓存和临时文件
│   ├── persona_cache_simple/         # LLM 缓存
│   ├── results/                      # 旧结果（可删除）
│   └── ...
│
├── 📂 archive_old_files/             # 旧版本存档
│   ├── supervised_step1_label_ground_truth.py
│   ├── ground_truth_llama_old.csv
│   ├── You are a psychologist specialized in th.ini
│   └── ... 其他旧文件
│
├── 📂 text_features/                 # 文本特征提取工具
│   ├── personality_simple.py         # 简化版 OCEAN 评分器
│   └── ...
│
├── 📂 utils/                         # 实用函数
│   └── ...
│
├── .env                              # 环境变量（配置）
├── .env.example                      # 环境变量示例
├── .gitignore                        # Git 忽略规则
├── README.md                         # 项目主 README [已迁移到 docs/]
└── requirements.txt                  # Python 依赖 [已迁移到 docs/]
```

---

## 📋 文件用途说明

### 🔴 Data Layer (`data/`)
| 文件 | 大小 | 说明 |
|------|------|------|
| `loan.csv` | 2.26G | 原始完整数据集 |
| `loan_with_desc.csv` | 筛选 | 仅含 desc 字段的记录 |
| `loan_clean_for_modeling.csv` | 56M | 干净建模数据 (514K × 36 特征) ⭐ |
| `loan_clean_with_ocean.csv` | 58M | 含 OCEAN 特征的数据 |

**用途：** 模型训练和特征工程的输入数据

---

### 📊 Results Layer (`results/`)

#### Metrics & Config
- `baseline_metrics.json` - 基线模型 AUC/准确率等
- `full_model_metrics.json` - 完整模型指标
- `model_comparison.csv` - 两个模型的对比
- `*_feature_config.json` - 特征列表和配置

#### Feature Reports
- `baseline_feature_importance.csv` - 特征重要性排名
- `full_model_feature_importance.csv` - 含 OCEAN 的特征重要性
- `feature_coverage_report.csv` - 特征覆盖率分析
- `retained_features_examples.csv` - 保留特征示例

#### OCEAN Results
- `ocean_ground_truth_500.csv` - Ground truth 标签 (500 样本)
- `ocean_features.csv` - 提取的 OCEAN 特征 (123K 行 × 5)
- `ocean_weights_coefficients.csv` - Ridge 回归权重
- `ocean_weights_formula.json` - 权重计算公式

#### Visualizations
- `baseline_model_evaluation.png` - 混淆矩阵、ROC 曲线等
- `full_model_evaluation.png` - 完整模型评估图表
- `ocean_weights_visualization.png` - OCEAN 权重可视化
- `data_cleaning_summary.png` - 数据清洗过程总结

---

### 🤖 Models Layer (`models/`)

#### Saved Models
```
models/saved_models/
├── xgboost_baseline_model.pkl      # 基线 XGBoost 模型
├── xgboost_full_model.pkl          # 含 OCEAN 特征的 XGBoost
├── ocean_ridge_models.pkl          # OCEAN Ridge 回归权重
└── ... 其他模型
```

#### Preprocessors
```
models/preprocessors/
├── preprocessor_baseline.pkl       # 基线预处理管道
├── preprocessor_full.pkl           # 完整模型预处理管道
└── ... 其他预处理器
```

**用途：** 生产环境中加载已训练的模型进行推理

---

### 📚 Documentation (`docs/`)

#### 执行指南
- `PROJECT_EXECUTION_ROADMAP.md` ⭐ - **完整执行步骤（客户展示用）**
- `README.md` - 项目总述

#### 优化报告
- `PREPROCESSING_OPTIMIZATION.md` - 优化总结 (成本、时间、维度)
- `OPTIMIZATION_EXPLANATION.md` - 如何向不同受众解释优化
- `OPTIMIZATION_DETAILED_BREAKDOWN.md` - 数学细节分解

#### 方法论
- `GROUND_TRUTH_STATUS.md` ⭐ - **Ground Truth 生成指南**
- `OCEAN_METHODOLOGY_FINAL.md` - OCEAN 特征提取方法
- `OCEAN_IMPROVEMENT_ANALYSIS.md` - OCEAN 性能分析

#### 最终报告
- `FINAL_PROJECT_REPORT.md` - 完整项目报告

---

### 🔧 Scripts (`scripts/`)

#### 管线执行
- `run_ocean_pipeline.py` - 运行完整 OCEAN 管线
- `execute_pipeline.py` - 执行预处理→建模管线
- `run_xgboost_comparison.py` - 比较基线与完整模型

#### 模型训练
- `train_xgboost_models.py` - 直接训练 XGBoost 模型

#### Ground Truth 生成
- `execute_05b_with_new_ground_truth.py` - 用新 ground truth 训练
- `regenerate_ground_truth_proper_llm.py` - 重新生成 ground truth

**用途：** 快速运行完整管线，无需逐个打开 notebook

---

### 📝 Logs (`logs/`)

所有执行日志：
- `xgboost_execution.log` - XGBoost 训练日志
- `ocean_execution.log` - OCEAN 特征提取日志
- `regenerate_ground_truth_final.log` - Ground truth 生成日志
- ... 其他执行日志

**用途：** 调试和追踪管线执行

---

## 🎯 典型工作流

### 场景 1：首次运行完整管线
```
1. 检查数据：data/loan_clean_for_modeling.csv ✅
2. 运行脚本：python scripts/run_ocean_pipeline.py
3. 检查结果：results/ (CSV、JSON、PNG)
4. 查看报告：docs/PROJECT_EXECUTION_ROADMAP.md
```

### 场景 2：只运行建模
```
1. 使用预处理数据：data/loan_clean_with_ocean.csv
2. 运行 notebook：notebooks/03_modeling/04_xgboost_baseline.ipynb
3. 模型保存到：models/saved_models/
4. 结果保存到：results/
```

### 场景 3：生成 Ground Truth
```
1. 参考指南：docs/GROUND_TRUTH_STATUS.md
2. 运行脚本：python scripts/regenerate_ground_truth_proper_llm.py
3. 查看结果：results/ocean_ground_truth_500.csv
4. 训练权重：notebooks/02_feature_engineering/05b_*.ipynb
```

### 场景 4：给客户展示
```
1. 执行路线图：docs/PROJECT_EXECUTION_ROADMAP.md
2. 优化说明：docs/OPTIMIZATION_EXPLANATION.md
3. 优化细节：docs/OPTIMIZATION_DETAILED_BREAKDOWN.md
4. 最终报告：docs/FINAL_PROJECT_REPORT.md
```

---

## 🧹 清理和维护

### 定期清理
- `logs/` - 保留最新的日志，删除旧日志
- `artifacts/` - 清理过期的缓存文件

### 版本管理
- 重要结果保存到 `results/` 或 `models/`
- 临时文件放在 `artifacts/`
- 旧版本归档到 `archive_old_files/`

### 备份关键文件
```
关键文件：
- data/loan_clean_for_modeling.csv    (核心数据)
- models/saved_models/*.pkl            (训练好的模型)
- results/ocean_weights_coefficients.csv (OCEAN 权重)
- results/*_metrics.json                (性能指标)
```

---

## 📊 文件大小参考

```
data/
  loan.csv                    2.26 GB  (原始完整数据)
  loan_clean_for_modeling.csv   56 MB  (清洁数据)

models/
  xgboost_baseline_model.pkl   ~10 MB
  xgboost_full_model.pkl       ~12 MB
  ocean_ridge_models.pkl       ~5 MB

results/
  ocean_features.csv           50 MB   (123K × 5 特征)
  baseline_feature_importance.csv 2 MB
```

---

## ✅ 整理结果

✅ **完成的整理：**
1. 📁 创建 `docs/` - 所有文档统一管理
2. 📁 创建 `results/` - 所有输出结果集中存放
3. 📁 创建 `scripts/` - 所有 Python 脚本便于查找
4. 📁 创建 `models/` - 模型和预处理器分类存储
5. 📁 创建 `logs/` - 执行日志集中管理
6. 📂 整理 `notebooks/` - Notebook 按阶段组织
7. 📋 创建本文档 - 项目结构清晰说明

---

## 📖 快速导航

| 需求 | 位置 |
|------|------|
| 📚 查看执行步骤 | `docs/PROJECT_EXECUTION_ROADMAP.md` |
| 📊 查看优化效果 | `docs/OPTIMIZATION_EXPLANATION.md` |
| 🤖 加载训练好的模型 | `models/saved_models/` |
| 📈 查看性能指标 | `results/*_metrics.json` |
| 🔍 查看特征重要性 | `results/*_feature_importance.csv` |
| 🧮 运行完整管线 | `scripts/run_ocean_pipeline.py` |
| 📝 查看所有日志 | `logs/` |
| 💾 检查原始数据 | `data/` |

---

**Last Updated:** October 2024
**Status:** ✅ Fully Organized
