"""
Create test_samples_500.csv from the main dataset
"""
import pandas as pd

print("Loading main dataset...")
df = pd.read_csv('loan_final_desc50plus_with_ocean_bge.csv', low_memory=False)
print(f"Loaded: {len(df):,} rows")

# Take first 500 samples
df_500 = df.head(500).copy()

# Save to test_samples_500.csv
output_file = 'test_samples_500.csv'
df_500.to_csv(output_file, index=False)

print(f"\n✓ Created: {output_file}")
print(f"  Rows: {len(df_500)}")
print(f"  Columns: {len(df_500.columns)}")
print(f"\nColumns included:")
print(f"  - desc: {df_500['desc'].notna().sum()} samples with descriptions")
print(f"  - All other loan features")

# Show sample
print(f"\nSample preview:")
print(df_500[['desc']].head(3))
