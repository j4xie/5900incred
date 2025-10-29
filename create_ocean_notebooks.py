#!/usr/bin/env python3
"""
Generate OCEAN Notebooks with Correct Prompt
Creates 4 notebooks for the new model configurations
"""

import json

# Correct OCEAN prompt
OCEAN_PROMPT = '''You are a psychologist specialized in the Big Five (OCEAN) for credit behavior research.
Infer traits ONLY from the applicant's own text. Ignore any structured fields
(credit grade, term, home ownership, income, etc.). If evidence is insufficient, return 0.50 with low confidence.

Use the following meanings to guide your analysis:
- Openness: curiosity, imagination, preference for novelty and new ideas.
  (Look for words like "learn," "try new," "explore," "creative," "open-minded.")
- Conscientiousness: organization, discipline, reliability, planning, and self-control.
  (Look for phrases about "planning," "saving," "on time," "responsibility.")
- Extraversion: sociability, assertiveness, energy, and enthusiasm in communication.
  (Look for interpersonal or energetic tone such as "team," "connect," "talk," "outgoing.")
- Agreeableness: cooperation, empathy, kindness, and trust.
  (Look for "help," "care," "family," "support," "honest.")
- Neuroticism: emotional instability, anxiety, sensitivity to stress or uncertainty.
  (Look for "worry," "stress," "pressure," "concern," "can't sleep.")

Loan description:
{description_text}

Return ONLY valid JSON in this exact format:
{{
  "openness": 0.5,
  "conscientiousness": 0.5,
  "extraversion": 0.5,
  "agreeableness": 0.5,
  "neuroticism": 0.5
}}'''

# Model configurations
MODELS = [
    {
        'model_id': 'openai/gpt-oss-120b',
        'provider': 'novita',
        'display_name': 'GPT-OSS-120B',
        'short_name': 'gpt_oss_120b',
        'output_file': '../ocean_ground_truth/gpt_oss_120b_ocean_500.csv',
        'checkpoint_file': '../ocean_ground_truth/.checkpoint_gpt_oss_120b.json'
    },
    {
        'model_id': 'Qwen/Qwen2.5-72B-Instruct',
        'provider': 'nebius',
        'display_name': 'Qwen-2.5-72B',
        'short_name': 'qwen_2.5_72b',
        'output_file': '../ocean_ground_truth/qwen_2.5_72b_ocean_500.csv',
        'checkpoint_file': '../ocean_ground_truth/.checkpoint_qwen_2.5_72b.json'
    },
    {
        'model_id': 'google/gemma-2-9b-it',
        'provider': 'nebius',
        'display_name': 'Gemma-2-9B',
        'short_name': 'gemma_2_9b',
        'output_file': '../ocean_ground_truth/gemma_2_9b_ocean_500.csv',
        'checkpoint_file': '../ocean_ground_truth/.checkpoint_gemma_2_9b.json'
    },
    {
        'model_id': 'deepseek-ai/DeepSeek-V3.1',
        'provider': 'novita',
        'display_name': 'DeepSeek-V3.1',
        'short_name': 'deepseek_v3.1',
        'output_file': '../ocean_ground_truth/deepseek_v3.1_ocean_500.csv',
        'checkpoint_file': '../ocean_ground_truth/.checkpoint_deepseek_v3.1.json'
    }
]

def create_notebook(config):
    """Create a notebook for a specific model"""
    notebook = {
        "cells": [
            # Header
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# 05d - OCEAN Generation: {config['display_name']}\\n",
                    "\\n",
                    f"**Model**: {config['model_id']}  \\n",
                    f"**Provider**: {config['provider']}  \\n",
                    "**Samples**: 500  \\n",
                    f"**Output**: {config['output_file']}  \\n",
                    "**Estimated Time**: 1.5-2 hours\\n",
                    "\\n",
                    f"This notebook generates OCEAN personality scores for 500 loan application samples using the {config['display_name']} model."
                ]
            },
            # Step 1: Import
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 1: Import Libraries and Load Data"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\\n",
                    "import numpy as np\\n",
                    "import json\\n",
                    "import requests\\n",
                    "import time\\n",
                    "import os\\n",
                    "from datetime import datetime\\n",
                    "\\n",
                    "print('✓ Libraries imported')"
                ]
            },
            # Load token
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def load_env():\\n",
                    "    env_dict = {}\\n",
                    "    try:\\n",
                    "        with open('../.env', 'r') as f:\\n",
                    "            for line in f:\\n",
                    "                if line.strip() and not line.startswith('#'):\\n",
                    "                    key, value = line.strip().split('=', 1)\\n",
                    "                    env_dict[key] = value\\n",
                    "    except:\\n",
                    "        print('Warning: Unable to read .env file')\\n",
                    "    return env_dict\\n",
                    "\\n",
                    "env_vars = load_env()\\n",
                    "hf_token = env_vars.get('HF_TOKEN', '')\\n",
                    "print('✓ HF token loaded' if hf_token else '❌ HF_TOKEN not found')"
                ]
            },
            # Load data
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_samples = pd.read_csv('../test_samples_500.csv')\\n",
                    "print(f'✓ Loaded {len(df_samples)} samples')"
                ]
            },
            # Step 2: Configuration
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 2: Define Model Configuration"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"MODEL_NAME = '{config['model_id']}'\\n",
                    f"PROVIDER = '{config['provider']}'\\n",
                    f"DISPLAY_NAME = '{config['display_name']}'\\n",
                    f"OUTPUT_FILE = '{config['output_file']}'\\n",
                    f"CHECKPOINT_FILE = '{config['checkpoint_file']}'\\n",
                    "\\n",
                    "print(f'Model: {DISPLAY_NAME}')\\n",
                    "print(f'Provider: {PROVIDER}')"
                ]
            },
            # Prompt
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"ocean_prompt_template = '''{OCEAN_PROMPT}'''\\n",
                    "\\n",
                    "print('✓ OCEAN prompt template defined')"
                ]
            },
            # Step 3: API Function
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 3: Define API Function"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def call_llm_for_ocean_scores(description_text, model_name, provider, api_token, max_retries=3):\\n",
                    "    prompt = ocean_prompt_template.format(description_text=description_text)\\n",
                    "    api_url = 'https://router.huggingface.co/v1/chat/completions'\\n",
                    "    headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}\\n",
                    "    payload = {\\n",
                    "        'messages': [{'role': 'user', 'content': prompt}],\\n",
                    "        'model': f'{model_name}:{provider}',\\n",
                    "        'stream': False,\\n",
                    "        'max_tokens': 200,\\n",
                    "        'temperature': 0.7\\n",
                    "    }\\n",
                    "    \\n",
                    "    for attempt in range(max_retries):\\n",
                    "        try:\\n",
                    "            response = requests.post(api_url, headers=headers, json=payload, timeout=30)\\n",
                    "            if response.status_code == 200:\\n",
                    "                result = response.json()\\n",
                    "                if 'choices' in result and len(result['choices']) > 0:\\n",
                    "                    text_output = result['choices'][0].get('message', {}).get('content', '')\\n",
                    "                    try:\\n",
                    "                        json_start = text_output.find('{')\\n",
                    "                        if json_start != -1:\\n",
                    "                            json_string = text_output[json_start:]\\n",
                    "                            json_end = json_string.find('}') + 1\\n",
                    "                            json_string = json_string[:json_end]\\n",
                    "                            score_dict = json.loads(json_string)\\n",
                    "                            return_value = {}\\n",
                    "                            for key in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:\\n",
                    "                                if key in score_dict:\\n",
                    "                                    return_value[key] = float(score_dict[key])\\n",
                    "                            if len(return_value) == 5:\\n",
                    "                                return return_value\\n",
                    "                    except:\\n",
                    "                        pass\\n",
                    "            elif response.status_code == 429 and attempt < max_retries - 1:\\n",
                    "                time.sleep(2 * (attempt + 1))\\n",
                    "                continue\\n",
                    "        except Exception as e:\\n",
                    "            if attempt < max_retries - 1:\\n",
                    "                time.sleep(2)\\n",
                    "    return None\\n",
                    "\\n",
                    "print('✓ API function defined')"
                ]
            },
            # Step 4: Checkpoint
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 4: Load Checkpoint (if exists)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "os.makedirs('../ocean_ground_truth', exist_ok=True)\\n",
                    "\\n",
                    "if os.path.exists(CHECKPOINT_FILE):\\n",
                    "    with open(CHECKPOINT_FILE, 'r') as f:\\n",
                    "        checkpoint = json.load(f)\\n",
                    "    print(f'✓ Checkpoint loaded: {checkpoint[\"processed_count\"]}/{checkpoint[\"total_count\"]}')\\n",
                    "    ocean_scores = checkpoint['ocean_scores']\\n",
                    "    start_idx = checkpoint['processed_count']\\n",
                    "else:\\n",
                    "    print('No checkpoint found, starting from scratch')\\n",
                    "    ocean_scores = []\\n",
                    "    start_idx = 0\\n",
                    "\\n",
                    "success_count = sum(1 for s in ocean_scores if s is not None)\\n",
                    "failure_count = len(ocean_scores) - success_count"
                ]
            },
            # Step 5: Process
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 5: Process All Samples"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('=' * 80)\\n",
                    "print(f'Processing {DISPLAY_NAME}')\\n",
                    "print('=' * 80)\\n",
                    "print(f'Total samples: {len(df_samples)}')\\n",
                    "print(f'Starting from: {start_idx}')\\n",
                    "print('=' * 80)\\n",
                    "\\n",
                    "start_time = time.time()\\n",
                    "\\n",
                    "for idx in range(start_idx, len(df_samples)):\\n",
                    "    row = df_samples.iloc[idx]\\n",
                    "    description = row.get('desc', '')\\n",
                    "    \\n",
                    "    if len(description) < 10:\\n",
                    "        ocean_scores.append(None)\\n",
                    "        failure_count += 1\\n",
                    "        continue\\n",
                    "    \\n",
                    "    ocean_score = call_llm_for_ocean_scores(description, MODEL_NAME, PROVIDER, hf_token, max_retries=2)\\n",
                    "    \\n",
                    "    if ocean_score:\\n",
                    "        ocean_scores.append(ocean_score)\\n",
                    "        success_count += 1\\n",
                    "    else:\\n",
                    "        ocean_scores.append(None)\\n",
                    "        failure_count += 1\\n",
                    "    \\n",
                    "    if (idx + 1) % 50 == 0 or (idx + 1) == len(df_samples):\\n",
                    "        elapsed = time.time() - start_time\\n",
                    "        rate = (idx + 1 - start_idx) / elapsed if elapsed > 0 else 0\\n",
                    "        eta = (len(df_samples) - (idx + 1)) / rate / 60 if rate > 0 else 0\\n",
                    "        print(f'{idx + 1}/{len(df_samples)} ({(idx+1)/len(df_samples)*100:.1f}%) | Success: {success_count} ({success_count/(idx+1)*100:.1f}%) | ETA: {eta:.1f}min')\\n",
                    "        checkpoint = {'model_name': MODEL_NAME, 'provider': PROVIDER, 'display_name': DISPLAY_NAME, 'total_count': len(df_samples), 'processed_count': idx+1, 'success_count': success_count, 'failure_count': failure_count, 'ocean_scores': ocean_scores, 'last_update': datetime.now().isoformat()}\\n",
                    "        with open(CHECKPOINT_FILE, 'w') as f:\\n",
                    "            json.dump(checkpoint, f, indent=2)\\n",
                    "    time.sleep(1)\\n",
                    "\\n",
                    "print(f'\\\\n✅ COMPLETE: {(time.time()-start_time)/60:.1f}min | Success: {success_count}/{len(df_samples)} ({success_count/len(df_samples)*100:.1f}%)')"
                ]
            },
            # Step 6: Save
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 6: Save Results"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data_list = [{'sample_id': idx, **score} if score else {'sample_id': idx, 'openness': None, 'conscientiousness': None, 'extraversion': None, 'agreeableness': None, 'neuroticism': None} for idx, score in enumerate(ocean_scores)]\\n",
                    "df_ocean = pd.DataFrame(data_list)\\n",
                    "df_ocean.to_csv(OUTPUT_FILE, index=False)\\n",
                    "print(f'✓ Results saved: {OUTPUT_FILE} ({len(df_ocean)} rows, {df_ocean[\"openness\"].notna().sum()} valid)')\\n",
                    "if os.path.exists(CHECKPOINT_FILE):\\n",
                    "    os.remove(CHECKPOINT_FILE)\\n",
                    "    print('✓ Checkpoint removed')"
                ]
            },
            # Step 7: Statistics
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Step 7: Display Statistics"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('=' * 80)\\n",
                    "print('OCEAN Statistics')\\n",
                    "print('=' * 80)\\n",
                    "print(df_ocean[['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']].describe())\\n",
                    "print('\\\\n✅ ALL DONE!')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook


def main():
    """Generate all 4 notebooks"""
    print('Generating OCEAN Notebooks with Correct Prompt')
    print('=' * 80)

    for config in MODELS:
        notebook = create_notebook(config)
        filename = f"notebooks/05d_{config['short_name']}.ipynb"

        with open(filename, 'w') as f:
            json.dump(notebook, f, indent=2)

        print(f'✓ Created: {filename}')

    print('\n✅ All notebooks created successfully!')
    print('\nNext steps:')
    print('1. Run: venv/bin/python3 run_ocean_pipeline.py')
    print('   OR run each notebook individually in Jupyter')
    print('2. Verify all CSV files have 500 valid OCEAN scores')


if __name__ == '__main__':
    main()
