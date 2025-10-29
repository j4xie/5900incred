#!/usr/bin/env python3
"""
05b - Train OCEAN Ridge Regression Weights using new LLM-based Ground Truth
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hashlib import md5

os.chdir('/Users/jietaoxie/Documents/GitHub/Credibly-INFO-5900')

print("\n" + "="*100)
print("05b - TRAIN OCEAN RIDGE REGRESSION WEIGHTS (WITH NEW LLM GROUND TRUTH)")
print("="*100 + "\n")

# OCEAN dimensions
OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

print("Step 1: Loading new Ground Truth data...")
try:
    df_ground_truth = pd.read_csv('ocean_ground_truth_500_llm.csv', low_memory=False)
    print(f"✅ Loaded Ground Truth: {df_ground_truth.shape[0]} samples")
except Exception as e:
    print(f"❌ Failed to load Ground Truth: {e}")
    sys.exit(1)

print("\nStep 2: Loading full modeling dataset...")
try:
    df_modeling = pd.read_csv('data/loan_clean_for_modeling.csv', low_memory=False)
    print(f"✅ Loaded modeling data: {df_modeling.shape[0]:,} samples")
except Exception as e:
    print(f"❌ Failed to load modeling data: {e}")
    sys.exit(1)

print("\nStep 3: Matching Ground Truth samples with modeling data...")
def get_desc_hash(desc):
    if pd.isna(desc):
        return None
    return md5(str(desc).encode()).hexdigest()

df_ground_truth['desc_hash'] = df_ground_truth['desc'].apply(get_desc_hash)
df_modeling['desc_hash'] = df_modeling['desc'].apply(get_desc_hash)

matching_indices = []
for hash_val in df_ground_truth['desc_hash']:
    matches = df_modeling[df_modeling['desc_hash'] == hash_val]
    if len(matches) > 0:
        matching_indices.append(matches.index[0])

print(f"✅ Matched {len(matching_indices)} / {len(df_ground_truth)} samples")

df_features = df_modeling.loc[matching_indices].reset_index(drop=True)
if len(matching_indices) < len(df_ground_truth):
    df_ground_truth = df_ground_truth.iloc[:len(df_features)].reset_index(drop=True)

print(f"   Feature matrix: {df_features.shape}")
print(f"   Ground Truth: {df_ground_truth.shape}")

# Define features
print("\nStep 4: Preparing feature matrix...")
numeric_features = [
    'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
    'delinq_2yrs', 'fico_range_low', 'fico_range_high', 'inq_last_6mths',
    'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc'
]

categorical_features = [
    'term', 'grade', 'sub_grade', 'purpose', 'home_ownership', 'emp_length',
    'verification_status', 'application_type'
]

available_numeric = [f for f in numeric_features if f in df_features.columns]
available_categorical = [f for f in categorical_features if f in df_features.columns]

print(f"✅ Numeric features: {len(available_numeric)}")
print(f"✅ Categorical features: {len(available_categorical)}")

# Prepare features
X = df_features[available_numeric + available_categorical].copy()

for col in available_numeric:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

for col in available_categorical:
    if X[col].isnull().sum() > 0:
        X[col].fillna('unknown', inplace=True)

# Create preprocessor
print("\nStep 5: Fitting preprocessor...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), available_numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_categorical)
    ])

X_processed = preprocessor.fit_transform(X)

try:
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(available_categorical)
    feature_names = list(available_numeric) + list(cat_feature_names)
except:
    feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]

print(f"✅ Total features (encoded): {X_processed.shape[1]}")

# Train Ridge models
print("\nStep 6: Training Ridge Regression models...")
ocean_models = {}
ocean_scores_dict = {}
alpha = 0.17

for ocean_trait in OCEAN_DIMS:
    print(f"  Training {ocean_trait}...", end='', flush=True)

    # Get target column with proper naming
    truth_col = f'{ocean_trait}_truth'
    if truth_col not in df_ground_truth.columns:
        print(f" ⚠️ Column {truth_col} not found!")
        continue

    y = df_ground_truth[truth_col].values

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_processed, y)

    y_pred = model.predict(X_processed)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    ocean_models[ocean_trait] = model
    ocean_scores_dict[ocean_trait] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

    print(f" ✅ (R²={r2:.4f}, RMSE={rmse:.4f})")

# Save models
print("\nStep 7: Saving trained models...")
with open('ocean_ridge_models_llm.pkl', 'wb') as f:
    pickle.dump({
        'models': ocean_models,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'numeric_features': available_numeric,
        'categorical_features': available_categorical,
        'alpha': alpha
    }, f)

print("✅ Models saved to ocean_ridge_models_llm.pkl")

# Save weights formula
weights_summary = {}
for ocean_trait, model in ocean_models.items():
    weights_summary[ocean_trait] = {
        'intercept': float(model.intercept_),
        'coefficients': {feature_names[i]: float(model.coef_[i]) for i in range(len(feature_names))},
        'alpha': alpha
    }

with open('ocean_weights_formula_llm.json', 'w') as f:
    json.dump(weights_summary, f, indent=2)

print("✅ Weights formula saved to ocean_weights_formula_llm.json")

print("\nStep 8: Model Performance Summary")
print("-" * 60)
for ocean_trait in OCEAN_DIMS:
    metrics = ocean_scores_dict[ocean_trait]
    print(f"{ocean_trait:20s}: R²={metrics['r2']:6.4f} | RMSE={metrics['rmse']:6.4f} | MAE={metrics['mae']:6.4f}")

print("\n" + "="*100)
print("✅ 05b COMPLETE - RIDGE WEIGHTS TRAINED WITH LLM GROUND TRUTH!")
print("="*100)
print("\nNext step: Run 05c to apply weights to all 123K+ customers")
