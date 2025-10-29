# 项目路径更新总结

## 更新时间
2025-10-22

## 更新原因
项目重组：将CSV数据文件移至 `data/` 目录，将notebook文件按工作流阶段分组到 `notebooks/` 子目录。

## 目录结构变化

### 之前（根目录混乱）
```
├── loan.csv
├── loan_with_desc.csv  
├── LC_loans_granting_model_dataset.csv
├── 01_data_cleaning_with_desc.ipynb
├── 02_feature_selection_and_leakage_check.ipynb
├── ...其他notebook文件...
└── ...其他文件...
```

### 之后（清晰的分层结构）
```
├── data/                                    # 所有CSV数据文件
│   ├── loan.csv
│   ├── loan_with_desc.csv
│   └── LC_loans_granting_model_dataset.csv
│
├── notebooks/                               # 按阶段组织的notebooks
│   ├── 01_data_preparation/
│   ├── 02_feature_engineering/
│   ├── 03_modeling/
│   └── 04_results_analysis/
│
└── (根目录的输出/报告文件)
    ├── *.csv (报告)
    ├── *.png (可视化)
    └── *.pkl (模型)
```

## 路径更新规则

所有notebooks现在使用相对路径 `../../` 来访问根目录文件：

### 数据输入路径
- `loan.csv` → `../../data/loan.csv`
- `loan_with_desc.csv` → `../../data/loan_with_desc.csv`
- `LC_loans_granting_model_dataset.csv` → `../../data/LC_loans_granting_model_dataset.csv`
- `loan_clean_for_modeling.csv` → `../../data/loan_clean_for_modeling.csv`

### 输出文件路径（保存在根目录）
- 所有 `.csv` 报告 → `../../filename.csv`
- 所有 `.png` 图表 → `../../filename.png`
- 所有 `.pkl` 模型 → `../../filename.pkl`
- 所有 `.json` 配置 → `../../filename.json`

## 已更新的Notebooks

### ✅ 01_data_preparation/ (4个)
1. `01_data_cleaning_with_desc.ipynb` - 4处路径更新
2. `02_feature_selection_and_leakage_check.ipynb` - 3处路径更新
3. `03_create_modeling_dataset.ipynb` - 5处路径更新
4. `view_loan_data.ipynb` - 1处路径更新

### ✅ 02_feature_engineering/ (1个)
5. `05_ocean_feature_extraction.ipynb` - 5处路径更新

### ✅ 03_modeling/ (2个)
6. `04_xgboost_baseline.ipynb` - 5处路径更新
7. `06_xgboost_with_ocean.ipynb` - 7处路径更新

### ✅ 04_results_analysis/ (1个)
8. `07_results_analysis.ipynb` - 4处路径更新

## 验证检查

所有notebooks的路径已验证：
- ✅ 所有 `pd.read_csv()` 使用正确的相对路径
- ✅ 所有 `.to_csv()` 输出到正确位置
- ✅ 所有图表保存路径正确
- ✅ 所有模型保存路径正确

## 使用说明

1. **运行notebooks**: 直接在Jupyter中打开并运行即可，路径会自动解析
2. **数据文件**: 全部位于 `data/` 目录
3. **输出文件**: 自动保存到项目根目录，便于查看和共享
4. **模型文件**: 保存在根目录，便于加载和部署

## 注意事项

⚠️ 如果移动notebook文件到其他位置，需要相应调整相对路径深度
⚠️ 新增notebook时，请按照相同的路径规范配置
