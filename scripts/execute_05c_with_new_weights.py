#!/usr/bin/env python3
"""
05c - Apply learned OCEAN weights to all customers using LLM-based Ground Truth
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

os.chdir('/Users/jietaoxie/Documents/GitHub/Credibly-INFO-5900')

OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

print("\n" + "="*100)
print("05c - APPLY OCEAN WEIGHTS TO ALL CUSTOMERS (LLM-BASED)")
print("="*100 + "\n")

print("Step 1: Loading trained models and preprocessor...")
try:
    with open('ocean_ridge_models_llm.pkl', 'rb') as f:
        model_package = pickle.load(f)

    ocean_models = model_package['models']
    preprocessor = model_package['preprocessor']
    feature_names = model_package['feature_names']
    numeric_features = model_package['numeric_features']
    categorical_features = model_package['categorical_features']
    alpha = model_package['alpha']

    print("✅ Models loaded successfully!")
    print(f"   OCEAN dimensions: {len(ocean_models)}")
    print(f"   Total features: {len(feature_names)}")

except FileNotFoundError:
    print("❌ Error: ocean_ridge_models_llm.pkl not found!")
    sys.exit(1)

print("\nStep 2: Loading complete modeling dataset...")
try:
    df = pd.read_csv('data/loan_clean_for_modeling.csv', low_memory=False)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    sys.exit(1)

print("\nStep 3: Preparing feature matrix...")
X = df[numeric_features + categorical_features].copy()

# Handle missing values
for col in numeric_features:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

for col in categorical_features:
    if X[col].isnull().sum() > 0:
        X[col].fillna('unknown', inplace=True)

print(f"✅ Feature matrix shape: {X.shape}")

print("\nStep 4: Encoding features using learned preprocessor...")
X_processed = preprocessor.transform(X)
print(f"✅ Encoded shape: {X_processed.shape}")

print("\nStep 5: Generating OCEAN scores for all customers...")
ocean_cols = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
ocean_features_dict = {col: [] for col in ocean_cols}

for ocean_trait, model in ocean_models.items():
    print(f"   Generating {ocean_trait} scores...", end='', flush=True)
    scores = model.predict(X_processed)
    scores = np.clip(scores, 0, 1)
    ocean_features_dict[ocean_trait] = scores
    mean_val = scores.mean()
    print(f" ✅ (mean={mean_val:.4f})")

ocean_df = pd.DataFrame(ocean_features_dict)

print(f"\n✅ OCEAN scores generated for {len(ocean_df):,} customers")

print("\nOCEAN Scores Summary:")
print(ocean_df.describe())

print("\nStep 6: Creating complete dataset with OCEAN features...")
df_with_ocean = df.copy()
for col in ocean_cols:
    df_with_ocean[col] = ocean_df[col]

print(f"✅ Complete dataset shape: {df_with_ocean.shape}")

print("\nStep 7: Saving datasets...")

# Save OCEAN-only features
ocean_only_file = 'ocean_features_llm.csv'
ocean_df.to_csv(ocean_only_file, index=False)
print(f"✅ {ocean_only_file} ({len(ocean_df):,} rows × {len(ocean_cols)} columns)")

# Save complete dataset with OCEAN
full_file = 'loan_clean_with_ocean_llm.csv'
df_with_ocean.to_csv(full_file, index=False)
print(f"✅ {full_file} ({df_with_ocean.shape[0]:,} rows × {df_with_ocean.shape[1]} columns)")

print("\nStep 8: Analysis by target variable...")
print("\nFully Paid (target=0) OCEAN scores:")
fully_paid_ocean = ocean_df[df['target'] == 0]
print(fully_paid_ocean.describe())

print("\nCharged Off (target=1) OCEAN scores:")
charged_off_ocean = ocean_df[df['target'] == 1]
print(charged_off_ocean.describe())

print("\nMean difference (Charged Off - Fully Paid):")
diff_means = charged_off_ocean.mean() - fully_paid_ocean.mean()
for col in ocean_cols:
    print(f"  {col:20s}: {diff_means[col]:+.6f}")

print("\n" + "="*100)
print("✅ 05c COMPLETE - OCEAN FEATURES APPLIED TO ALL CUSTOMERS!")
print("="*100)

print("\nGenerated files (LLM-based):")
print("  ✅ ocean_ground_truth_500_llm.csv     - 500 labeled samples with LLM-generated OCEAN scores")
print("  ✅ ocean_ridge_models_llm.pkl         - Trained Ridge regression models")
print("  ✅ ocean_weights_formula_llm.json     - Weight coefficients in JSON format")
print("  ✅ ocean_features_llm.csv             - OCEAN scores for all 123K+ customers")
print("  ✅ loan_clean_with_ocean_llm.csv      - Complete dataset with OCEAN features")

print("\nNext steps:")
print("  1. Run XGBoost baseline (without OCEAN)")
print("  2. Run XGBoost with OCEAN features")
print("  3. Compare performance")
