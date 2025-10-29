"""
Llama OCEAN 特征生成完整流程
一键运行脚本
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import kagglehub
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# 项目模块
from utils.io import load_lending_club_data, prepare_binary_target
from utils.seed import set_seed
from utils.metrics import compute_all_metrics, delong_test
from text_features.ocean_llama_labeler import OceanLlamaLabeler, OCEAN_DIMS
from utils.ocean_weight_learner import OceanWeightLearner
from utils.ocean_feature_generator import OceanFeatureGenerator
from utils.ocean_evaluator import OceanEvaluator

# 设置随机种子
set_seed(42)


def main():
    print("=" * 80)
    print("Llama OCEAN 特征生成完整流程")
    print("=" * 80)
    print()

    # ========== 0. 加载数据 ==========
    print("\n[Step 0] 加载数据...")
    print("-" * 80)

    # 直接使用本地已下载的数据文件（避免 kagglehub 下载卡住）
    file_path = os.path.expanduser("~/.cache/kagglehub/datasets/ethon0426/lending-club-20072020q1/versions/3/Loan_status_2007-2020Q3.gzip")

    if not os.path.exists(file_path):
        print("本地数据不存在，使用 kagglehub 下载...")
        path = kagglehub.dataset_download("ethon0426/lending-club-20072020q1")
        file_path = path + "/Loan_status_2007-2020Q3.gzip"

    ROW_LIMIT = 10000  # 测试用，可以改为 None 加载全部数据

    df = load_lending_club_data(file_path, row_limit=ROW_LIMIT)
    df = prepare_binary_target(df, target_col="loan_status")

    # 清理百分比列
    percent_cols = ["int_rate", "revol_util"]
    for col in percent_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")

    print(f"\n数据形状: {df.shape}")
    print(f"违约率: {df['target'].mean():.2%}")

    # ========== 1. Llama 打标签 ==========
    print("\n[Step 1] Llama 打标签（生成 Ground Truth）...")
    print("-" * 80)
    print("⏱️  预计时间: 10-15分钟")
    print("💰 成本: $0 (免费)")
    print()

    # 检查是否已有 ground truth（避免重复标注）
    ground_truth_path = 'artifacts/ground_truth_llama.csv'
    if os.path.exists(ground_truth_path):
        print(f"✅ 发现已存在的 ground truth: {ground_truth_path}")
        print("   自动使用已有文件（节省时间）")
        df_truth = pd.read_csv(ground_truth_path)
        print(f"   加载了 {len(df_truth)} 个已标注样本")
    else:
        print("未找到已有 ground truth，开始新标注...")
        os.makedirs('artifacts', exist_ok=True)
        labeler = OceanLlamaLabeler()
        df_truth = labeler.label_batch(
            df,
            sample_size=500,
            stratified=True,
            rate_limit_delay=0.5
        )
        df_truth.to_csv(ground_truth_path, index=False)
        print(f"\n✅ Ground Truth 已保存到: {ground_truth_path}")

    # 评估 Ground Truth 质量
    evaluator = OceanEvaluator()
    evaluator.evaluate_ground_truth_quality(df_truth)

    # ========== 2. 学习权重 ==========
    print("\n[Step 2] 学习权重（Ridge Regression）...")
    print("-" * 80)

    CATEGORICAL_VARS = [
        'grade', 'purpose', 'term', 'home_ownership',
        'emp_length', 'verification_status', 'application_type'
    ]
    CATEGORICAL_VARS = [c for c in CATEGORICAL_VARS if c in df_truth.columns]

    learner = OceanWeightLearner(method='ridge', alpha=0.1)

    # 准备 OCEAN ground truth（重命名列以匹配学习器期望）
    y_ocean_truth = df_truth[[f'{d}_truth' for d in OCEAN_DIMS]].copy()
    y_ocean_truth.columns = OCEAN_DIMS  # 重命名：openness_truth → openness

    weights, encoder = learner.fit(
        X_categorical=df_truth[CATEGORICAL_VARS],
        y_ocean_truth=y_ocean_truth,
        cv=5
    )

    # 保存权重
    os.makedirs('artifacts', exist_ok=True)
    weights_path = 'artifacts/ocean_weights_llama.pkl'
    joblib.dump({'weights': weights, 'encoder': encoder}, weights_path)
    print(f"\n✅ 权重已保存到: {weights_path}")

    # 显示学习摘要
    print("\n学习结果摘要:")
    print(learner.get_summary())

    # ========== 3. 生成全量 OCEAN 特征 ==========
    print("\n[Step 3] 生成全量 OCEAN 特征...")
    print("-" * 80)

    generator = OceanFeatureGenerator(weights, encoder)
    df_full = generator.generate_features(df)

    print("\n✅ OCEAN 特征已添加到数据集")

    # 评估预测能力
    evaluator.evaluate_predictive_power(df_full, target_col='target')
    summary = evaluator.generate_summary_report()
    print("\nOCEAN 特征汇总:")
    print(summary)

    # ========== 4. XGBoost A/B 对比测试 ==========
    print("\n[Step 4] XGBoost A/B 对比测试...")
    print("-" * 80)

    # 定义特征
    numeric_features = [
        "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
        "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
        "total_acc"
    ]
    numeric_features = [c for c in numeric_features if c in df_full.columns]
    categorical_features_model = [c for c in CATEGORICAL_VARS if c in df_full.columns]

    baseline_features = numeric_features + categorical_features_model
    ocean_features = OCEAN_DIMS

    print(f"\nBaseline 特征数: {len(baseline_features)}")
    print(f"OCEAN 特征数: {len(ocean_features)}")

    # 预处理器
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    # 方案 A: Baseline
    preprocessor_A = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features_model),
        ],
        remainder="drop"
    )

    # 方案 B: Baseline + OCEAN
    preprocessor_B = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("ocean", "passthrough", ocean_features),
            ("cat", categorical_transformer, categorical_features_model),
        ],
        remainder="drop"
    )

    # Train-Test Split
    y = df_full['target'].values
    X_train, X_test, y_train, y_test = train_test_split(
        df_full, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n训练集: {len(X_train)} 样本 | 违约率: {y_train.mean():.2%}")
    print(f"测试集: {len(X_test)} 样本 | 违约率: {y_test.mean():.2%}")

    # 训练方案 A: Baseline
    print("\n训练方案 A: Baseline (无 OCEAN)...")
    X_train_A = preprocessor_A.fit_transform(X_train)
    X_test_A = preprocessor_A.transform(X_test)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(1, pos)

    model_A = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        verbosity=0
    )
    model_A.fit(X_train_A, y_train)

    y_proba_A = model_A.predict_proba(X_test_A)[:, 1]
    metrics_A = compute_all_metrics(y_test, y_proba_A)

    print("\n方案 A 结果:")
    for k, v in metrics_A.items():
        print(f"  {k}: {v:.4f}")

    # 训练方案 B: Baseline + OCEAN
    print("\n训练方案 B: Baseline + OCEAN...")
    X_train_B = preprocessor_B.fit_transform(X_train)
    X_test_B = preprocessor_B.transform(X_test)

    model_B = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        verbosity=0
    )
    model_B.fit(X_train_B, y_train)

    y_proba_B = model_B.predict_proba(X_test_B)[:, 1]
    metrics_B = compute_all_metrics(y_test, y_proba_B)

    print("\n方案 B 结果:")
    for k, v in metrics_B.items():
        print(f"  {k}: {v:.4f}")

    # 对比结果
    comparison = evaluator.compare_models(y_test, y_proba_A, y_proba_B)

    # ========== 5. 保存结果 ==========
    print("\n[Step 5] 保存结果...")
    print("-" * 80)

    os.makedirs('artifacts/results', exist_ok=True)

    # 保存模型
    joblib.dump(model_B, 'artifacts/xgb_ocean_llama.pkl')
    print("✅ 模型已保存: artifacts/xgb_ocean_llama.pkl")

    # 保存对比结果
    results_df = pd.DataFrame([
        {'model': 'Baseline', **metrics_A},
        {'model': 'Baseline+OCEAN', **metrics_B}
    ])
    results_df.to_csv('artifacts/results/llama_ocean_results.csv', index=False)
    print("✅ 结果已保存: artifacts/results/llama_ocean_results.csv")

    print("\n" + "=" * 80)
    print("🎉 完成！")
    print("=" * 80)
    print("\n📊 最终结果:")
    print(results_df.to_string(index=False))

    print(f"\n💰 总成本: $0")
    print(f"⏱️  总耗时: ~20分钟")

    # 评估是否达标
    delta_auc = metrics_B['roc_auc'] - metrics_A['roc_auc']
    delta_pr = metrics_B['pr_auc'] - metrics_A['pr_auc']
    delta_ks = metrics_B['ks'] - metrics_A['ks']

    print("\n🎯 评估标准（至少满足一项）:")
    print(f"  ROC-AUC 提升 ≥ +0.010: {delta_auc:+.4f} {'✅' if delta_auc >= 0.010 else '❌'}")
    print(f"  PR-AUC 提升 ≥ +0.008:  {delta_pr:+.4f} {'✅' if delta_pr >= 0.008 else '❌'}")
    print(f"  KS 提升 ≥ +1.0:        {delta_ks:+.2f} {'✅' if delta_ks >= 1.0 else '❌'}")

    if comparison['significant']:
        print(f"\n✅ 统计显著性: p={comparison['delong_p']:.4f} < 0.05")
    else:
        print(f"\n⚠️  统计显著性: p={comparison['delong_p']:.4f} ≥ 0.05 (不显著)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
