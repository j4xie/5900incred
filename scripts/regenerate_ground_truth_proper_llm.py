#!/usr/bin/env python3
"""
Regenerate OCEAN Ground Truth using proper Llama LLM via HF Router API
This uses the working LlamaClient from utils.llama_client
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load HF_TOKEN from environment variable
if 'HF_TOKEN' not in os.environ:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running this script.")
os.chdir('/Users/jietaoxie/Documents/GitHub/Credibly-INFO-5900')

print("\n" + "="*100)
print("REGENERATING OCEAN GROUND TRUTH WITH PROPER LLAMA LLM")
print("Using HF Router API: https://router.huggingface.co/v1/chat/completions")
print("="*100 + "\n")

# Import the working components
from text_features.ocean_llama_labeler import OceanLlamaLabeler, OCEAN_DIMS
from utils.llama_client import LlamaClient

print("Step 1: Verifying Llama Client configuration...")
try:
    hf_token = os.getenv('HF_TOKEN')
    client = LlamaClient(hf_token)
    print(f"✅ LlamaClient initialized")
    print(f"   API URL: {client.api_url}")
    print(f"   Model: {client.model}")
except Exception as e:
    print(f"❌ Failed to initialize LlamaClient: {e}")
    sys.exit(1)

print("\nStep 2: Loading modeling dataset...")
try:
    df = pd.read_csv('data/loan_clean_for_modeling.csv', low_memory=False)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    sys.exit(1)

print("\nStep 3: Initializing OceanLlamaLabeler...")
try:
    labeler = OceanLlamaLabeler(hf_token)
    print(f"✅ Labeler initialized")
except Exception as e:
    print(f"❌ Failed to initialize labeler: {e}")
    sys.exit(1)

print("\nStep 4: Generating Ground Truth with Llama LLM...")
print("(This will take 15-30 minutes with 500 samples)\n")

try:
    # Use the labeler's batch method to generate ground truth
    df_ground_truth = labeler.label_batch(
        df,
        sample_size=500,
        stratified=True,
        rate_limit_delay=0.5
    )

    print(f"\n✅ Ground Truth generated successfully!")
    print(f"   Rows: {len(df_ground_truth):,}")
    print(f"   Columns: {len(df_ground_truth.columns)}")

except Exception as e:
    print(f"\n❌ Failed during label generation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Saving Ground Truth dataset...")
try:
    output_file = 'ocean_ground_truth_500_llm.csv'
    df_ground_truth.to_csv(output_file, index=False)

    import os as os_module
    file_size = os_module.path.getsize(output_file) / 1024

    print(f"✅ Saved: {output_file}")
    print(f"   Size: {file_size:.2f} KB")
    print(f"   Rows: {len(df_ground_truth):,}")
    print(f"   Columns: {len(df_ground_truth.columns)}")

except Exception as e:
    print(f"❌ Failed to save file: {e}")
    sys.exit(1)

print("\nStep 6: Verifying Ground Truth quality...")
try:
    truth_cols = [f'{dim}_truth' for dim in OCEAN_DIMS]
    truth_data = df_ground_truth[truth_cols]

    print(f"\nOCEAN Ground Truth Statistics:")
    print(truth_data.describe())

    # Check for proper variation
    for col in truth_cols:
        min_val = truth_data[col].min()
        max_val = truth_data[col].max()
        mean_val = truth_data[col].mean()
        std_val = truth_data[col].std()
        print(f"\n{col}:")
        print(f"  Range: [{min_val:.2f}, {max_val:.2f}]")
        print(f"  Mean: {mean_val:.4f} | Std: {std_val:.4f}")

except Exception as e:
    print(f"❌ Failed to verify quality: {e}")

print("\n" + "="*100)
print("✅ OCEAN GROUND TRUTH REGENERATION COMPLETE!")
print("="*100)
print("\nNext steps:")
print("1. Review ocean_ground_truth_500_llm.csv to verify quality")
print("2. Run 05b_train_ocean_ridge_weights.ipynb with new Ground Truth")
print("3. Run 05c_apply_ocean_to_all.ipynb to generate OCEAN features")
print("4. Run XGBoost models for comparison")
