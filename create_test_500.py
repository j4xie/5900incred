#!/usr/bin/env python3
"""
快速创建 test_samples_500.csv
从 loan_final_desc50plus.csv 提取前500行
"""
import pandas as pd
import os

print("="*80)
print("Creating test_samples_500.csv")
print("="*80)

# Load original dataset
original_file = 'data/loan_final_desc50plus.csv'

if not os.path.exists(original_file):
    print(f"❌ Error: {original_file} not found!")
    print(f"   Current directory: {os.getcwd()}")
    exit(1)

print(f"\n[1/3] Loading: {original_file}")
df = pd.read_csv(original_file, low_memory=False)
print(f"  ✓ Loaded {len(df):,} samples, {len(df.columns)} columns")

# Take first 500 samples
print(f"\n[2/3] Extracting first 500 samples...")
df_500 = df.head(500).copy()
print(f"  ✓ Selected {len(df_500)} samples")

# Verify desc column
if 'desc' not in df_500.columns:
    print(f"  ❌ Error: 'desc' column not found!")
    print(f"  Available columns: {df_500.columns.tolist()[:10]}...")
    exit(1)

desc_count = df_500['desc'].notna().sum()
print(f"  ✓ Valid descriptions: {desc_count}/{len(df_500)}")

# Save
output_file = 'test_samples_500.csv'
print(f"\n[3/3] Saving: {output_file}")
df_500.to_csv(output_file, index=False)

file_size = os.path.getsize(output_file) / 1024 / 1024
print(f"  ✓ Saved: {output_file}")
print(f"  File size: {file_size:.2f} MB")

print("\n" + "="*80)
print("✓ Complete! You can now run 05e_extract_bge_embeddings.ipynb")
print("="*80)
