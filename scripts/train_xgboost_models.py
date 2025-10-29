#!/usr/bin/env python3
"""
XGBoost Credit Risk Prediction Models
1. Baseline model (without OCEAN features)
2. Model with OCEAN features
3. Performance comparison
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/Users/jietaoxie/Documents/GitHub/Credibly-INFO-5900')

print("\n" + "="*100)
print("XGBOOST CREDIT RISK PREDICTION - BASELINE VS OCEAN FEATURES")
print("="*100 + "\n")

# Feature definitions
numeric_features = [
    'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'collections_12_mths_ex_med'
]

categorical_features = [
    'term', 'grade', 'sub_grade', 'purpose', 'home_ownership', 'emp_length',
    'verification_status', 'application_type'
]

ocean_features = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("Step 1: Loading data...\n")

df_baseline = pd.read_csv('data/loan_clean_for_modeling.csv', low_memory=False)
df_ocean = pd.read_csv('loan_clean_with_ocean_llm.csv', low_memory=False)

print(f"✅ Baseline data: {df_baseline.shape[0]:,} rows × {df_baseline.shape[1]} columns")
print(f"✅ OCEAN data: {df_ocean.shape[0]:,} rows × {df_ocean.shape[1]} columns")

# Extract target
y = df_baseline['target'].values
print(f"   Target distribution: {(y==0).sum():,} Fully Paid | {(y==1).sum():,} Charged Off\n")

# ============================================================================
# STEP 2: Prepare Feature Matrix
# ============================================================================
print("Step 2: Preparing features...\n")

# Handle missing values for baseline
X_baseline = df_baseline[numeric_features + categorical_features].copy()
for col in numeric_features:
    if X_baseline[col].isnull().sum() > 0:
        X_baseline[col].fillna(X_baseline[col].median(), inplace=True)
for col in categorical_features:
    if X_baseline[col].isnull().sum() > 0:
        X_baseline[col].fillna('unknown', inplace=True)

# Handle missing values for OCEAN
X_ocean = df_ocean[numeric_features + categorical_features + ocean_features].copy()
for col in numeric_features:
    if X_ocean[col].isnull().sum() > 0:
        X_ocean[col].fillna(X_ocean[col].median(), inplace=True)
for col in categorical_features:
    if X_ocean[col].isnull().sum() > 0:
        X_ocean[col].fillna('unknown', inplace=True)

print(f"✅ Baseline features: {X_baseline.shape}")
print(f"✅ OCEAN features: {X_ocean.shape}\n")

# ============================================================================
# STEP 3: Train/Test Split
# ============================================================================
print("Step 3: Train/Test split (80/20)...\n")

X_train_base, X_test_base, y_train, y_test = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42, stratify=y
)

X_train_ocean, X_test_ocean, _, _ = train_test_split(
    X_ocean, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {len(X_train_base):,} samples")
print(f"✅ Test set: {len(X_test_base):,} samples\n")

# ============================================================================
# STEP 4: Feature Preprocessing
# ============================================================================
print("Step 4: Feature preprocessing...\n")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

X_train_base_processed = preprocessor.fit_transform(X_train_base)
X_test_base_processed = preprocessor.transform(X_test_base)

X_train_ocean_processed = preprocessor.fit_transform(X_train_ocean[numeric_features + categorical_features])
X_test_ocean_processed = preprocessor.transform(X_test_ocean[numeric_features + categorical_features])

# Add OCEAN features
X_train_ocean_full = np.column_stack([
    X_train_ocean_processed,
    X_train_ocean[ocean_features].values
])
X_test_ocean_full = np.column_stack([
    X_test_ocean_processed,
    X_test_ocean[ocean_features].values
])

print(f"✅ Baseline features processed: {X_train_base_processed.shape}")
print(f"✅ OCEAN features processed: {X_train_ocean_full.shape}\n")

# ============================================================================
# STEP 5: Train XGBoost Models
# ============================================================================
print("Step 5: Training XGBoost models...\n")

xgb_params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0
}

print("Training BASELINE model (without OCEAN)...")
model_baseline = xgb.XGBClassifier(**xgb_params)
model_baseline.fit(X_train_base_processed, y_train, eval_set=[(X_test_base_processed, y_test)], verbose=False)
print("✅ Baseline model trained\n")

print("Training OCEAN model (with OCEAN features)...")
model_ocean = xgb.XGBClassifier(**xgb_params)
model_ocean.fit(X_train_ocean_full, y_train, eval_set=[(X_test_ocean_full, y_test)], verbose=False)
print("✅ OCEAN model trained\n")

# ============================================================================
# STEP 6: Evaluate Models
# ============================================================================
print("Step 6: Model evaluation...\n")

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance"""
    print(f"\n{model_name}")
    print("-" * 70)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"Train Accuracy:  {train_acc:.4f}")
    print(f"Test Accuracy:   {test_acc:.4f}")
    print(f"Precision:       {test_prec:.4f}")
    print(f"Recall:          {test_rec:.4f}")
    print(f"F1-Score:        {test_f1:.4f}")
    print(f"ROC-AUC:         {test_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print(f"Specificity:     {specificity:.4f}")

    return {
        'model_name': model_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'auc': test_auc,
        'specificity': specificity,
        'cm': cm
    }

results_baseline = evaluate_model(
    model_baseline, X_train_base_processed, X_test_base_processed, y_train, y_test,
    "BASELINE MODEL (without OCEAN)"
)

results_ocean = evaluate_model(
    model_ocean, X_train_ocean_full, X_test_ocean_full, y_train, y_test,
    "OCEAN MODEL (with OCEAN features)"
)

# ============================================================================
# STEP 7: Performance Comparison
# ============================================================================
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70 + "\n")

comparison_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity'],
    'Baseline': [
        results_baseline['test_acc'],
        results_baseline['precision'],
        results_baseline['recall'],
        results_baseline['f1'],
        results_baseline['auc'],
        results_baseline['specificity']
    ],
    'With OCEAN': [
        results_ocean['test_acc'],
        results_ocean['precision'],
        results_ocean['recall'],
        results_ocean['f1'],
        results_ocean['auc'],
        results_ocean['specificity']
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Improvement'] = comparison_df['With OCEAN'] - comparison_df['Baseline']
comparison_df['% Change'] = (comparison_df['Improvement'] / comparison_df['Baseline'] * 100).round(2)

print(comparison_df.to_string(index=False))

print("\n" + "="*70)
improvement_auc = results_ocean['auc'] - results_baseline['auc']
improvement_f1 = results_ocean['f1'] - results_baseline['f1']

if improvement_auc > 0:
    print(f"✅ OCEAN features IMPROVE ROC-AUC by {improvement_auc:+.4f} ({improvement_auc/results_baseline['auc']*100:+.2f}%)")
else:
    print(f"⚠️ OCEAN features slight decrease in ROC-AUC by {improvement_auc:.4f}")

if improvement_f1 > 0:
    print(f"✅ OCEAN features IMPROVE F1-Score by {improvement_f1:+.4f} ({improvement_f1/results_baseline['f1']*100:+.2f}%)")
else:
    print(f"⚠️ OCEAN features slight decrease in F1-Score by {improvement_f1:.4f}")

# ============================================================================
# STEP 8: Feature Importance
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70 + "\n")

# Get feature names
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(cat_feature_names) + ocean_features

# Baseline feature importance
baseline_importance = model_baseline.feature_importances_[:len(all_feature_names) - len(ocean_features)]
ocean_importance = model_ocean.feature_importances_

print("Top 10 Features - BASELINE Model:")
baseline_feature_names = list(numeric_features) + list(cat_feature_names)
baseline_sorted = sorted(zip(baseline_feature_names, baseline_importance), key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(baseline_sorted[:10], 1):
    print(f"  {i:2d}. {feat[:40]:<40s} {imp:.4f}")

print("\nTop 10 Features - OCEAN Model:")
ocean_sorted = sorted(zip(all_feature_names, ocean_importance), key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(ocean_sorted[:10], 1):
    marker = "🌊" if feat in ocean_features else "  "
    print(f"  {i:2d}. {marker} {feat[:40]:<40s} {imp:.4f}")

# Check OCEAN feature importance in OCEAN model
print("\nOCEAN Features Importance in OCEAN Model:")
ocean_only = [(feat, imp) for feat, imp in zip(all_feature_names, ocean_importance) if feat in ocean_features]
ocean_only_sorted = sorted(ocean_only, key=lambda x: x[1], reverse=True)
for feat, imp in ocean_only_sorted:
    rank = [f for f, _ in ocean_sorted].index(feat) + 1
    print(f"  {feat:20s}: {imp:.4f} (Rank: #{rank})")

# ============================================================================
# STEP 9: Save Results
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70 + "\n")

# Save models
with open('xgboost_baseline_model.pkl', 'wb') as f:
    pickle.dump(model_baseline, f)
print("✅ Saved: xgboost_baseline_model.pkl")

with open('xgboost_ocean_model.pkl', 'wb') as f:
    pickle.dump(model_ocean, f)
print("✅ Saved: xgboost_ocean_model.pkl")

# Save comparison results
results_dict = {
    'baseline': results_baseline,
    'ocean': results_ocean,
    'comparison': comparison_df.to_dict(),
    'improvement_auc': float(improvement_auc),
    'improvement_f1': float(improvement_f1)
}

with open('xgboost_comparison_results.json', 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    results_for_json = {
        'baseline': {
            'test_acc': float(results_baseline['test_acc']),
            'precision': float(results_baseline['precision']),
            'recall': float(results_baseline['recall']),
            'f1': float(results_baseline['f1']),
            'auc': float(results_baseline['auc']),
            'specificity': float(results_baseline['specificity'])
        },
        'ocean': {
            'test_acc': float(results_ocean['test_acc']),
            'precision': float(results_ocean['precision']),
            'recall': float(results_ocean['recall']),
            'f1': float(results_ocean['f1']),
            'auc': float(results_ocean['auc']),
            'specificity': float(results_ocean['specificity'])
        },
        'improvement_auc': float(improvement_auc),
        'improvement_f1': float(improvement_f1)
    }
    json.dump(results_for_json, f, indent=2)
print("✅ Saved: xgboost_comparison_results.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*100)
print("✅ XGBOOST TRAINING COMPLETE!")
print("="*100)

print("\n📊 FINAL RESULTS SUMMARY:\n")
print(f"Baseline Model ROC-AUC:  {results_baseline['auc']:.4f}")
print(f"OCEAN Model ROC-AUC:     {results_ocean['auc']:.4f}")
print(f"Improvement:             {improvement_auc:+.4f} ({improvement_auc/results_baseline['auc']*100:+.2f}%)")

print(f"\nBaseline Model F1-Score: {results_baseline['f1']:.4f}")
print(f"OCEAN Model F1-Score:    {results_ocean['f1']:.4f}")
print(f"Improvement:             {improvement_f1:+.4f} ({improvement_f1/results_baseline['f1']*100:+.2f}%)")

print("\n✅ Models saved for deployment:")
print("   - xgboost_baseline_model.pkl")
print("   - xgboost_ocean_model.pkl")
print("   - xgboost_comparison_results.json")

print("\n" + "="*100 + "\n")
