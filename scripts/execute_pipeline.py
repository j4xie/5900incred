#!/usr/bin/env python3
"""
Complete pipeline execution script
Runs all notebooks in proper order with proper data dependencies
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Set up environment
# Load HF_TOKEN from environment variable - set it before running this script
if 'HF_TOKEN' not in os.environ:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running this script.")

def run_notebook(notebook_path, description):
    """Execute a Jupyter notebook"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Notebook: {notebook_path}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--ExecutePreprocessor.timeout=1800',
            '--output', notebook_path,
            notebook_path
        ], timeout=2400)
        
        if result.returncode == 0:
            print(f"\n✅ Successfully executed: {notebook_path}")
            return True
        else:
            print(f"\n❌ Failed to execute: {notebook_path}")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n❌ Timeout executing: {notebook_path}")
        return False
    except Exception as e:
        print(f"\n❌ Error executing {notebook_path}: {e}")
        return False

def main():
    """Main pipeline execution"""
    os.chdir('/Users/jietaoxie/Documents/GitHub/Credibly-INFO-5900')
    
    # Check if prerequisites exist
    if not os.path.exists('loan_clean_for_modeling.csv'):
        print("⚠️ loan_clean_for_modeling.csv not found!")
        print("Running prerequisite notebooks...")
        
        # Run feature selection
        if not run_notebook(
            'notebooks/01_data_preparation/02_feature_selection_and_leakage_check.ipynb',
            'Feature Selection & Data Leakage Check'
        ):
            print("Failed to run feature selection")
            return False
        
        # Run data preparation
        if not run_notebook(
            'notebooks/01_data_preparation/03_create_modeling_dataset.ipynb',
            'Create Modeling Dataset'
        ):
            print("Failed to run data preparation")
            return False
    else:
        print("✅ loan_clean_for_modeling.csv found - skipping prerequisites")
    
    print("\n" + "="*80)
    print("STARTING OCEAN PIPELINE")
    print("="*80)
    
    # Run OCEAN pipeline
    notebooks = [
        ('05a_llm_ocean_ground_truth.ipynb', 'LLM Ground Truth Generation (500 samples)'),
        ('05b_train_ocean_ridge_weights.ipynb', 'Train Ridge Regression Weights'),
        ('05c_apply_ocean_to_all.ipynb', 'Apply OCEAN Formula to All Customers'),
    ]
    
    for notebook, description in notebooks:
        if not os.path.exists(notebook):
            print(f"⚠️ {notebook} not found")
            continue
            
        if not run_notebook(notebook, description):
            print(f"Failed to run {notebook}")
            # Continue with next notebook instead of breaking
    
    # Run XGBoost models
    print("\n" + "="*80)
    print("STARTING XGBOOST MODELS")
    print("="*80)
    
    xgboost_notebooks = [
        ('notebooks/03_modeling/04_xgboost_baseline.ipynb', 'XGBoost Baseline (without OCEAN)'),
        ('notebooks/03_modeling/06_xgboost_with_ocean.ipynb', 'XGBoost with OCEAN Features'),
        ('notebooks/04_results_analysis/07_results_analysis.ipynb', 'Results Analysis & Comparison'),
    ]
    
    for notebook, description in xgboost_notebooks:
        if not os.path.exists(notebook):
            print(f"⚠️ {notebook} not found")
            continue
            
        if not run_notebook(notebook, description):
            print(f"Failed to run {notebook}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
