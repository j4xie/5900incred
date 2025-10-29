# DeBERTa Training Guide

## Quick Setup

DeBERTa Ridge/Elastic Net notebooks can be created by copying BGE versions and changing 3 lines:

### For Ridge:

```python
# Line to change #1: Embedding file
embedding_file = '../deberta_embeddings_500.npy'  # Changed from bge_embeddings_500.npy

# Line to change #2: Model file output
model_file = f'../ridge_models_deberta_{llm_key}.pkl'  # Added _deberta

# Line to change #3: Report metadata
'embedding_model': 'microsoft/deberta-v3-large',  # Changed from BAAI/bge-large-en-v1.5
```

### For Elastic Net:

Same 3 changes, plus update title to mention DeBERTa.

## Step-by-Step Instructions

1. **Copy BGE notebook**:
   ```bash
   cp 05f_train_ridge_all_models.ipynb 05f_deberta_train_ridge_all.ipynb
   ```

2. **Edit cell-0** (title):
   ```markdown
   # 05f - Train Ridge Regression Models (All LLMs, DeBERTa Embeddings)
   
   **Purpose**: Train Ridge regression models using DeBERTa-v3-large embeddings
   
   **Input Files**:
   - deberta_embeddings_500.npy - DeBERTa embeddings (500x1024)
   ```

3. **Edit cell-6** (load embeddings):
   ```python
   embedding_file = '../deberta_embeddings_500.npy'
   ```

4. **Edit cell-8** (model file):
   ```python
   model_file = f'../ridge_models_deberta_{llm_key}.pkl'
   ```

5. **Edit cell-8** (report metadata):
   ```python
   'embedding_model': 'microsoft/deberta-v3-large',
   ```

6. **Edit cell-10** (comparison file):
   ```python
   comparison_file = '../05f_ridge_deberta_comparison.csv'
   ```

7. **Edit cell-12** (visualization):
   ```python
   viz_file = '../05f_ridge_deberta_visualization.png'
   fig.suptitle('Ridge Regression (DeBERTa Embeddings)', ...)
   ```

## Expected Results

Based on model sizes:
- **BGE**: 326M parameters → R² ≈ -1.0 (Ridge) / 0.19 (Elastic Net)
- **DeBERTa**: 1.5B parameters → Expected R² ≈ -0.5~0.0 (Ridge) / 0.25-0.35 (Elastic Net)

DeBERTa should perform better due to:
1. Larger model (4.6x more parameters)
2. Better language understanding
3. Disentangled attention mechanism

## Files to Create

1. `05f_deberta_train_ridge_all.ipynb`
2. `05f_deberta_train_elasticnet_all.ipynb`

Simply follow the 7 edits above for each file!
