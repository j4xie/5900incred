#!/usr/bin/env python3
"""
Run OCEAN Ground Truth Generation Pipeline
Executes all 4 model notebooks sequentially
"""

import subprocess
import sys
import time
from datetime import datetime

# Model configurations
MODELS = [
    {
        'name': 'GPT-OSS-120B',
        'notebook': 'notebooks/05d_gpt_oss_120b.ipynb',
        'output': 'ocean_ground_truth/gpt_oss_120b_ocean_500.csv'
    },
    {
        'name': 'Qwen-2.5-72B',
        'notebook': 'notebooks/05d_qwen_2.5_72b.ipynb',
        'output': 'ocean_ground_truth/qwen_2.5_72b_ocean_500.csv'
    },
    {
        'name': 'Gemma-2-9B',
        'notebook': 'notebooks/05d_gemma_2_9b.ipynb',
        'output': 'ocean_ground_truth/gemma_2_9b_ocean_500.csv'
    },
    {
        'name': 'DeepSeek-V3.1',
        'notebook': 'notebooks/05d_deepseek_v3.1.ipynb',
        'output': 'ocean_ground_truth/deepseek_v3.1_ocean_500.csv'
    }
]

def run_notebook(notebook_path):
    """Execute a Jupyter notebook using papermill or nbconvert"""
    print(f'\n{"=" * 100}')
    print(f'Running: {notebook_path}')
    print(f'{"=" * 100}')
    print(f'Start time: {datetime.now().isoformat()}')

    try:
        # Try using jupyter nbconvert
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',
            notebook_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f'\n✅ SUCCESS: {notebook_path}')
            return True
        else:
            print(f'\n❌ FAILED: {notebook_path}')
            print(f'Error: {result.stderr}')
            return False

    except Exception as e:
        print(f'\n❌ EXCEPTION: {notebook_path}')
        print(f'Error: {str(e)}')
        return False


def main():
    print('=' * 100)
    print('OCEAN Ground Truth Generation Pipeline')
    print('=' * 100)
    print(f'Total models: {len(MODELS)}')
    print(f'Pipeline start: {datetime.now().isoformat()}\n')

    start_time = time.time()
    results = []

    for idx, model_config in enumerate(MODELS, 1):
        print(f'\n\n{"#" * 100}')
        print(f'MODEL {idx}/{len(MODELS)}: {model_config["name"]}')
        print(f'{"#" * 100}')

        model_start = time.time()
        success = run_notebook(model_config['notebook'])
        model_time = time.time() - model_start

        results.append({
            'model': model_config['name'],
            'notebook': model_config['notebook'],
            'output': model_config['output'],
            'success': success,
            'time_minutes': model_time / 60
        })

        print(f'\nModel processing time: {model_time / 60:.1f} minutes')

        if not success:
            print(f'\n⚠️ WARNING: {model_config["name"]} failed. Continuing with next model...')

    # Print summary
    total_time = time.time() - start_time
    print(f'\n\n{"=" * 100}')
    print('PIPELINE COMPLETE')
    print(f'{"=" * 100}')
    print(f'Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)')
    print(f'\nResults:')

    success_count = sum(1 for r in results if r['success'])
    for r in results:
        status = '✅' if r['success'] else '❌'
        print(f'  {status} {r["model"]}: {r["time_minutes"]:.1f} min')

    print(f'\nSuccess rate: {success_count}/{len(MODELS)} ({success_count/len(MODELS)*100:.1f}%)')

    if success_count == len(MODELS):
        print('\n✅ ALL MODELS COMPLETED SUCCESSFULLY!')
        print('\nNext steps:')
        print('1. Verify all CSV files have 500 valid OCEAN scores')
        print('2. Run notebook 06 (XGBoost with OCEAN features)')
    else:
        print(f'\n⚠️ {len(MODELS) - success_count} model(s) failed')
        print('Please check the error messages above')


if __name__ == '__main__':
    main()
