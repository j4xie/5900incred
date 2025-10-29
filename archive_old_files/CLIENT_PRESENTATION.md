# Big Five (OCEAN) Personality Features for Credit Risk Modeling
## Client Presentation - Project Plan & Progress

**Date**: October 1, 2025
**Team**: Credibly INFO-5900 Gamma
**Project**: Integrating Personality Psychology into Credit Default Prediction

---

## 📊 Executive Summary

We are implementing **Big Five personality traits** (OCEAN: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) as additional features for LendingClub credit risk models, inspired by Yu et al. (2023).

**Goal**: Improve baseline model performance by extracting psychological signals from borrower text data.

**Success Criteria**: Achieve at least ONE of:
- ROC-AUC improvement ≥ **+0.010**
- PR-AUC improvement ≥ **+0.008**
- KS statistic improvement ≥ **+1.0**

---

## 🎯 Current State: What We Have

### Baseline Models (Completed ✅)

#### 1. **Logistic Regression Baseline**
- **File**: [`LogRegBaseLineModel.ipynb`](LogRegBaseLineModel.ipynb)
- **Features**: 11 numeric + 8 categorical
- **Performance**:
  - ROC-AUC: **0.7196**
  - PR-AUC: **0.3165**
- **Preprocessing**: Median imputation + StandardScaler + OHE
- **Random seed**: 42, Train/Test: 80/20

#### 2. **XGBoost Baseline**
- **File**: [`XGBoostBaseLineModel.ipynb`](XGBoostBaseLineModel.ipynb)
- **Features**: Same as LogReg
- **Performance** (10k sample):
  - ROC-AUC: **0.6765**
  - PR-AUC: **0.2587**
- **Experiments**: Full features, reduced features, PCA, grid search, threshold tuning

### Key Observations
✅ Strong preprocessing pipeline established
✅ Reproducible experiments (fixed seeds)
✅ Clear baseline metrics to beat
⚠️ **Text features not yet utilized** (desc, title, purpose columns available but unused)

---

## 🚀 Proposed Plan: OCEAN Feature Integration

### Architecture Overview

```
LendingClub Data (loan_status, title, emp_title, financial features)
           ↓
    Text Extraction & Merging
    (title + emp_title → combined_text)
           ↓
    OCEAN Scoring Module
    (LLM-based personality assessment)
           ↓
    [O] [C] [E] [A] [N] → 5 numeric features (0-1 scale)
           ↓
    Feature Integration
    (Baseline features + OCEAN → ColumnTransformer)
           ↓
    Model Training (LogReg / XGBoost)
           ↓
    A/B Comparison & Statistical Testing
```

---

## 📋 Implementation Roadmap

### Phase 1: Infrastructure Setup (✅ **COMPLETED**)

**What We Built**:

1. **Project Structure**
   ```
   Credibly-INFO-5900/
   ├── data/                          # Raw data cache
   ├── artifacts/
   │   ├── persona_cache/             # LLM response caching
   │   └── results/                   # Metrics & visualizations
   ├── text_features/
   │   └── personality.py             # OCEAN scoring engine
   ├── utils/
   │   ├── seed.py                    # Reproducibility
   │   ├── metrics.py                 # Evaluation (ROC, PR, KS, Brier, ECE)
   │   └── io.py                      # Data loading helpers
   └── notebooks/
       ├── 03_ocean_features.ipynb    # Main integration notebook
       └── 04_explain_shap.ipynb      # Interpretability analysis
   ```

2. **Core Modules** ([text_features/personality.py](text_features/personality.py))
   - ✅ LLM-based OCEAN scoring (OpenAI API integration)
   - ✅ SHA256-based caching (avoid redundant API calls)
   - ✅ **Offline deterministic fallback** (hash-based pseudo-scores for development)
   - ✅ Rate limiting & exponential backoff
   - ✅ Strict JSON validation (0-1 scale enforcement)

3. **Utilities**
   - ✅ [`utils/metrics.py`](utils/metrics.py): ROC-AUC, PR-AUC, KS, Brier, ECE, DeLong test, Bootstrap CI, Lift curves
   - ✅ [`utils/seed.py`](utils/seed.py): Global seed management (42)
   - ✅ [`utils/io.py`](utils/io.py): Data loading, text merging, feature splitting

---

### Phase 2: Data Analysis & OCEAN Scoring (🔄 **READY TO RUN**)

**Status**: Notebook created, awaiting execution

**Notebook**: [`notebooks/03_ocean_features.ipynb`](notebooks/03_ocean_features.ipynb)

**What It Does**:

1. **Text Coverage Analysis**
   - Available fields: `title` (100% coverage), `emp_title` (95% coverage)
   - ⚠️ **Data Limitation**: No `desc` (borrower self-description) field
   - **Adaptation**: Use `title + emp_title` as weak personality proxy
   - Visualizations: Length distribution, coverage by loan grade

2. **OCEAN Scoring Pipeline**
   - Initialize `OceanScorer` (offline mode for demo)
   - Batch score 10k samples with caching
   - Generate 5 features: `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`
   - Validate distributions (should be roughly 0.3-0.7 range given limited context)

3. **Feature Integration**
   - Merge OCEAN into numeric feature list
   - Maintain existing ColumnTransformer (median imputation + StandardScaler)
   - Split: Feature Set A (baseline) vs Feature Set B (baseline + OCEAN)

---

### Phase 3: A/B Model Comparison (🔄 **READY TO RUN**)

**Experiments**:

| Model | Feature Set | Metrics Tracked |
|-------|-------------|-----------------|
| **LogReg A** | Baseline (19 features) | ROC-AUC, PR-AUC, KS, Brier, ECE |
| **LogReg B** | Baseline + OCEAN (24 features) | ↑ Same metrics |
| **XGBoost A** | Baseline | ↑ Same metrics |
| **XGBoost B** | Baseline + OCEAN | ↑ Same metrics + Feature Importance |

**Statistical Tests**:
- **DeLong test**: Compare ROC-AUC between A and B (p < 0.05 for significance)
- **Bootstrap CI**: 95% confidence intervals for PR-AUC and KS
- **5-Fold CV**: Cross-validated performance (mean ± std)

**Visualizations** (auto-generated):
- ROC curves (A vs B overlay)
- Precision-Recall curves
- Lift curves (decile analysis)
- Metric comparison bar charts

---

### Phase 4: Interpretability & Validation (🔄 **READY TO RUN**)

**Notebook**: [`notebooks/04_explain_shap.ipynb`](notebooks/04_explain_shap.ipynb)

**Analyses**:

1. **XGBoost Feature Importance**
   - Rank all features by gain
   - Check where OCEAN dimensions appear in Top-N
   - Visualize Top 30 features

2. **SHAP Analysis**
   - Global: Beeswarm plot (feature contributions across all samples)
   - Local: Force plots for high-risk vs low-risk borrowers
   - OCEAN-specific: Isolated SHAP values for personality dimensions
   - **Key Question**: Does higher conscientiousness reduce default risk? (expected: yes)

3. **Correlation Analysis**
   - Heatmap: OCEAN vs financial features
   - Check for multicollinearity (correlation > 0.7 indicates redundancy)

4. **Robustness Checks**
   - Text truncation sensitivity (400/800/1200 chars)
   - Scoring consistency (repeated evaluations)

---

## 🔧 What's Missing & Next Steps

### Missing Components

1. **Text Data Limitation** ⚠️
   - **Issue**: Dataset lacks borrower descriptions (original Yu et al. paper used self-written loan applications)
   - **Current Workaround**: Using `title + emp_title` (short, formulaic text)
   - **Impact**: OCEAN scores will be **weak signals** (not rich personality assessments)
   - **Mitigation**:
     - Document limitation clearly in report
     - Frame as proof-of-concept for technical pipeline
     - Suggest data enrichment for production (e.g., application essays, social media)

2. **LLM API Integration** (Optional)
   - **Current State**: Offline mode (deterministic fallback) ready for demo
   - **To Enable Real Scoring**:
     ```python
     # In notebook 03, change:
     scorer = OceanScorer(offline_mode=False)  # Requires OPENAI_API_KEY
     ```
   - **Cost Estimate**: ~$2-5 for 10k samples with gpt-4o-mini
   - **Decision**: Use offline mode for initial demo, enable API if results promising

3. **Execution & Results** 🎯
   - **Action Required**: Run [`notebooks/03_ocean_features.ipynb`](notebooks/03_ocean_features.ipynb)
   - **Time**: ~10-15 minutes (offline mode), ~30-60 min (API mode with rate limits)
   - **Output**:
     - Metrics CSV with A/B comparison
     - 5+ visualization PNG files
     - Trained model (`xgb_ocean_model.joblib`)

### Immediate Next Steps (For Client Demo)

1. ✅ **Preparation Done**:
   - Code infrastructure complete
   - Notebooks documented and ready
   - Utility functions tested

2. 🔄 **Ready to Execute**:
   ```bash
   # Open Jupyter
   jupyter notebook notebooks/03_ocean_features.ipynb

   # Run all cells (Kernel → Restart & Run All)
   # Expected runtime: 10-15 minutes
   ```

3. 📊 **Demo Materials**:
   - Show `CLIENT_PRESENTATION.md` (this document)
   - Walk through notebook 03 structure
   - Display sample OCEAN scores (from Section 2)
   - Present A/B comparison table (Section 4)
   - Show SHAP interpretability (notebook 04)

4. 📝 **Report Deliverables**:
   - One-page summary: Baseline → +OCEAN → Results
   - Visualization deck: Coverage, distributions, A/B metrics, SHAP
   - Model card: Limitations, data quality, ethical considerations

---

## 🎬 Demo Script (15-20 min)

### Slide 1: Problem Statement (2 min)
- Credit risk modeling traditionally uses financial features only
- **Insight from Yu et al. (2023)**: Personality traits (Big Five) add predictive power
- **Our Approach**: Extract OCEAN from loan text → integrate into XGBoost/LogReg

### Slide 2: Current Baseline (3 min)
- Show [`LogRegBaseLineModel.ipynb`](LogRegBaseLineModel.ipynb) & [`XGBoostBaseLineModel.ipynb`](XGBoostBaseLineModel.ipynb)
- Metrics: ROC-AUC 0.72 (LogReg), 0.68 (XGB on 10k sample)
- **Gap**: Text features unused

### Slide 3: OCEAN Pipeline Architecture (4 min)
- Show diagram (from "Architecture Overview" above)
- Live code walkthrough: [`text_features/personality.py`](text_features/personality.py)
  - Demonstrate caching
  - Show offline fallback logic
- Quick test: Score 3 sample loans

### Slide 4: Integration Notebook Walkthrough (6 min)
- Open [`notebooks/03_ocean_features.ipynb`](notebooks/03_ocean_features.ipynb)
- **Section 1**: Text coverage analysis (show plots)
- **Section 2**: OCEAN scoring (show distribution histograms)
- **Section 4**: A/B comparison table
  - Highlight ROC-AUC delta
  - Show DeLong test p-value
- **Section 6**: Visualizations (ROC curves, Lift charts)

### Slide 5: Interpretability (3 min)
- Open [`notebooks/04_explain_shap.ipynb`](notebooks/04_explain_shap.ipynb)
- Show XGBoost feature importance ranking
- Display SHAP beeswarm plot
- **Key Message**: "Where do OCEAN features rank? Are they interpretable?"

### Slide 6: Results & Limitations (2 min)
- **If metrics improve**: "OCEAN adds signal, next steps: scale to full data, enable API"
- **Data Limitation**: Acknowledge lack of rich borrower text
- **Frame as**: Technical validation of pipeline, not production-ready features

### Q&A (5 min)

---

## 📈 Success Metrics Checklist

**Acceptance Criteria** (at least 1 must be met):
- [ ] ROC-AUC improvement ≥ +0.010
- [ ] PR-AUC improvement ≥ +0.008
- [ ] KS improvement ≥ +1.0
- [ ] SHAP analysis shows reasonable OCEAN feature importance
- [ ] OCEAN features have interpretable directions (e.g., high conscientiousness → lower default)

**Process Validation**:
- [x] Code is reproducible (seed=42)
- [x] Caching works (SHA256-based)
- [x] A/B comparison follows best practices
- [x] Statistical tests implemented (DeLong, Bootstrap)
- [ ] Results documented with visualizations

---

## 📚 References

1. **Yu et al. (2023)**: "Chatbot or Human? Using Chatgpt to Extract Personality Traits and Credit Scoring"
   - Method: ChatGPT → Big Five → LightGBM
   - Result: Improved accuracy + interpretability

2. **LendingClub Dataset**:
   - Source: Kaggle `ethon0426/lending-club-20072020q1`
   - Limitation: No borrower descriptions (adapted approach using title + emp_title)

3. **SHAP**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"

---

## 🔗 Quick Links

- **Main Integration Notebook**: [notebooks/03_ocean_features.ipynb](notebooks/03_ocean_features.ipynb)
- **Interpretability Notebook**: [notebooks/04_explain_shap.ipynb](notebooks/04_explain_shap.ipynb)
- **OCEAN Scoring Module**: [text_features/personality.py](text_features/personality.py)
- **Metrics Utilities**: [utils/metrics.py](utils/metrics.py)
- **Baseline Models**: [LogRegBaseLineModel.ipynb](LogRegBaseLineModel.ipynb) | [XGBoostBaseLineModel.ipynb](XGBoostBaseLineModel.ipynb)

---

## 💡 For Next Meeting

**Prepare**:
1. Run notebook 03 (generate results)
2. Export key visualizations to `artifacts/results/`
3. Create 1-page executive summary with filled-in metrics
4. Prepare to discuss: "Should we invest in richer text data collection?"

**Discussion Points**:
- Is +0.01 AUC improvement worth the added complexity?
- How to handle data limitation in production?
- Alternative text sources (external enrichment, NLP on existing fields)
- Timeline for full-scale validation (100k+ samples)

---

**Document Status**: 🟢 Ready for Client Review
**Last Updated**: October 1, 2025
**Contact**: Credibly Team Gamma
