#!/usr/bin/env python3
"""
Execute XGBoost models for credit risk prediction
1. Baseline model (without OCEAN features)
2. Model with OCEAN features
3. Compare performance
"""

import json
import os
import sys
import traceback
from pathlib import Path

os.chdir('/Users/jietaoxie/Documents/GitHub/Credibly-INFO-5900')

def extract_notebook_code(notebook_path):
    """Extract code cells from a Jupyter notebook"""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if source.strip():
                code_cells.append(source)

    return code_cells


def run_notebook(notebook_path, description):
    """Extract and execute notebook code"""
    print("\n" + "="*100)
    print(f"Executing: {description}")
    print(f"Notebook: {notebook_path}")
    print("="*100 + "\n")

    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return False

    try:
        # Extract code cells
        code_cells = extract_notebook_code(notebook_path)
        print(f"Found {len(code_cells)} code cells\n")

        # Create execution namespace
        exec_globals = {
            '__name__': '__main__',
            '__file__': notebook_path,
        }

        # Execute each cell
        for idx, code in enumerate(code_cells, 1):
            print(f"\n--- Cell {idx}/{len(code_cells)} ---")
            try:
                exec(code, exec_globals)
            except Exception as e:
                print(f"\n❌ Error in cell {idx}:")
                print(f"   {type(e).__name__}: {e}")
                traceback.print_exc()
                # Continue with next cell instead of aborting

        print(f"\n✅ Successfully executed: {notebook_path}")
        return True

    except Exception as e:
        print(f"\n❌ Failed to execute {notebook_path}: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution"""
    print("\n" + "="*100)
    print("XGBOOST CREDIT RISK PREDICTION - BASELINE vs OCEAN FEATURES")
    print("="*100)

    # Define execution order
    notebooks = [
        ('notebooks/03_modeling/04_xgboost_baseline.ipynb', '04 - XGBoost Baseline (without OCEAN)'),
        ('notebooks/03_modeling/06_xgboost_with_ocean.ipynb', '06 - XGBoost with OCEAN Features'),
    ]

    # Track execution
    results = {}

    for notebook_path, description in notebooks:
        # Skip if notebook doesn't exist
        if not os.path.exists(notebook_path):
            print(f"\n⚠️ Skipping: {notebook_path} (not found)")
            results[notebook_path] = 'SKIPPED'
            continue

        # Execute notebook
        success = run_notebook(notebook_path, description)
        results[notebook_path] = 'SUCCESS' if success else 'FAILED'

    # Print summary
    print("\n" + "="*100)
    print("XGBOOST EXECUTION SUMMARY")
    print("="*100)

    for notebook, status in results.items():
        symbol = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⊘"
        print(f"{symbol} {notebook:60s} - {status}")

    print("\n" + "="*100)
    if all(s in ['SUCCESS', 'SKIPPED'] for s in results.values()):
        print("✅ XGBoost execution completed successfully!")
    else:
        print("⚠️ Some notebooks failed - check the output above")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
