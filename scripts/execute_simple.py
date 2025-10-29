#!/usr/bin/env python3
"""
Simple direct execution of OCEAN pipeline
"""

import os
import sys

# Set environment
# Load HF_TOKEN from environment variable
if 'HF_TOKEN' not in os.environ:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running this script.")
os.chdir('/Users/jietaoxie/Documents/GitHub/Credibly-INFO-5900')

print("\n" + "="*100)
print("STARTING OCEAN PIPELINE - 05a (Ground Truth Generation)")
print("="*100 + "\n")

# Execute 05a - Load and setup
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from huggingface_hub import InferenceClient
import time
import hashlib

pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

print("Step 1: Loading cleaned modeling data...")
df = pd.read_csv('data/loan_clean_for_modeling.csv', low_memory=False)
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

print("\nStep 2: Selecting 500 balanced samples...")
np.random.seed(42)
df_charged_off = df[df['target'] == 1].copy()
df_fully_paid = df[df['target'] == 0].copy()

sample_charged_off = df_charged_off.sample(n=min(250, len(df_charged_off)), random_state=42)
sample_fully_paid = df_fully_paid.sample(n=min(250, len(df_fully_paid)), random_state=42)

df_sample_500 = pd.concat([sample_charged_off, sample_fully_paid], ignore_index=False)
df_sample_500 = df_sample_500.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Selected {len(df_sample_500):,} samples")
print(f"   - Charged Off: {(df_sample_500['target']==1).sum():,}")
print(f"   - Fully Paid: {(df_sample_500['target']==0).sum():,}")

print("\nStep 3: Initializing Hugging Face Inference Client...")
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    print("❌ ERROR: HF_TOKEN not set!")
    sys.exit(1)

try:
    client = InferenceClient(token=hf_token)
    print("✅ Hugging Face client initialized")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

print("\nStep 4: Defining OCEAN extraction function...")

def extract_ocean_from_llm(desc_text, client, max_retries=3):
    """Extract OCEAN scores from loan description"""
    if pd.isna(desc_text) or str(desc_text).strip() == '':
        return {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5,
            'status': 'empty_text'
        }

    prompt = f"""Analyze the following loan application description and rate the borrower's personality on the OCEAN traits (Big Five Personality Model) on a scale of 0 to 1.

Description:
{desc_text}

---

Output ONLY a JSON object (no markdown, no extra text) with the following format:
{{
  "openness": 0.X,
  "conscientiousness": 0.X,
  "extraversion": 0.X,
  "agreeableness": 0.X,
  "neuroticism": 0.X
}}

Where:
- Openness (0-1): Imagination, curiosity, creativity
- Conscientiousness (0-1): Responsibility, discipline, organization
- Extraversion (0-1): Sociability, energy, assertiveness
- Agreeableness (0-1): Cooperation, trust, altruism
- Neuroticism (0-1): Emotional instability, anxiety

Respond with ONLY the JSON object."""

    for attempt in range(max_retries):
        try:
            response = client.text_generation(
                prompt,
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                max_new_tokens=200,
                temperature=0.3,
                top_p=0.9
            )

            response_text = str(response).strip()

            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]

            ocean_dict = json.loads(response_text)

            required_keys = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
            valid = True

            for key in required_keys:
                if key not in ocean_dict:
                    valid = False
                    break
                value = float(ocean_dict[key])
                ocean_dict[key] = np.clip(value, 0, 1)

            if valid:
                ocean_dict['status'] = 'success'
                return ocean_dict

        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                return {
                    'openness': 0.5,
                    'conscientiousness': 0.5,
                    'extraversion': 0.5,
                    'agreeableness': 0.5,
                    'neuroticism': 0.5,
                    'status': 'json_error'
                }
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    'openness': 0.5,
                    'conscientiousness': 0.5,
                    'extraversion': 0.5,
                    'agreeableness': 0.5,
                    'neuroticism': 0.5,
                    'status': 'api_error'
                }
            time.sleep(2 ** attempt)

    return {
        'openness': 0.5,
        'conscientiousness': 0.5,
        'extraversion': 0.5,
        'agreeableness': 0.5,
        'neuroticism': 0.5,
        'status': 'max_retries'
    }

print("✅ Function defined")

print("\nStep 5: Generating OCEAN labels for 500 samples...")
print("(This will take 10-30 minutes due to API rate limiting)\n")

ocean_labels = []
status_counts = {}

for idx, row in df_sample_500.iterrows():
    desc = row['desc']
    ocean_scores = extract_ocean_from_llm(desc, client)
    status = ocean_scores.pop('status', 'unknown')
    status_counts[status] = status_counts.get(status, 0) + 1
    ocean_labels.append(ocean_scores)

    if (idx + 1) % 50 == 0:
        print(f"Progress: {idx + 1} / {len(df_sample_500)} samples processed")

    if (idx + 1) % 10 == 0:
        time.sleep(1)

ocean_labels_df = pd.DataFrame(ocean_labels)

print(f"\n✅ Label generation complete!")
print(f"\nStatus distribution:")
for status, count in sorted(status_counts.items()):
    print(f"  {status}: {count}")

print("\nStep 6: Creating and saving Ground Truth dataset...")
df_ground_truth = df_sample_500[['desc', 'target']].reset_index(drop=True)
df_ground_truth = pd.concat([df_ground_truth, ocean_labels_df], axis=1)

output_file = 'ocean_ground_truth_500.csv'
df_ground_truth.to_csv(output_file, index=False)

import os
file_size = os.path.getsize(output_file) / 1024
print(f"✅ Saved: {output_file}")
print(f"   Size: {file_size:.2f} KB")
print(f"   Rows: {len(df_ground_truth):,}")
print(f"   Columns: {len(df_ground_truth.columns)}")

print("\n" + "="*100)
print("✅ 05a COMPLETE - Ground Truth OCEAN labels generated!")
print("="*100)
print("\nNext steps:")
print("1. Run 05b_train_ocean_ridge_weights.ipynb")
print("2. Run 05c_apply_ocean_to_all.ipynb")
print("3. Run XGBoost models")

