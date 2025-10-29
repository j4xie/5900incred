# Big Five (OCEAN) Personality Features Integration
## Complete Methodology

> **Note**: This is the primary OCEAN methodology document in English. For a more detailed Chinese reference with comprehensive analysis, see [OCEAN_METHODOLOGY_DETAILED_REFERENCE.md](./OCEAN_METHODOLOGY_DETAILED_REFERENCE.md).

---

## 📋 Overview

This document outlines the complete methodology for integrating Big Five (OCEAN) personality traits into credit risk models using supervised learning.

---

## 🔬 Step-by-Step Methodology

---

### **Step 1: Find Useful String Variables in Dataset**

**Objective**: Identify all categorical variables that can be used to infer personality traits

**Available Variables**:
- `grade` - Credit grade (A-G)
- `sub_grade` - Credit sub-grade (A1-G5)
- `purpose` - Loan purpose (debt_consolidation, small_business, wedding, etc.)
- `term` - Loan term (36 months, 60 months)
- `emp_length` - Employment length (< 1 year, 10+ years, etc.)
- `home_ownership` - Home ownership status (OWN, RENT, MORTGAGE)
- `verification_status` - Income verification status (Verified, Not Verified)
- `application_type` - Application type (Individual, Joint)

**Rationale**:
These categorical variables capture behavioral choices and life circumstances that may reflect underlying personality traits.

**Output**:
- List of 8-10 usable categorical variables
- Coverage statistics (% non-null)
- Number of unique categories per variable

---

### **Step 2: Create Combined Context**

**Objective**: Combine multiple variables to create richer context for personality inference

**Approach**:
Since individual variables provide limited information (e.g., `purpose = "car"` alone doesn't reveal much), we combine them into a **borrower profile**:

```
Borrower Profile =
  "Loan Purpose: debt_consolidation |
   Loan Term: 60 months |
   Credit Grade: C (C4) |
   Employment Length: 5 years |
   Home Ownership: RENT |
   Income Verification: Verified |
   Application Type: Individual"
```

**Benefit**:
Combined context provides a more complete picture of the borrower's financial behavior and life situation.

---

### **Step 3: Analyze Relationship Between Variables and Target**

**Objective**: Understand which variables are predictive of default risk (to inform personality mapping)

**Method**:
1. **Calculate default rate by category**
   ```
   grade_A → 7.3% default rate
   grade_G → 40.3% default rate
   ```

2. **Compute Lift**
   ```
   Lift = Category Default Rate / Overall Default Rate

   grade_A → Lift 0.47x (low risk)
   grade_G → Lift 2.57x (high risk)
   ```

3. **Statistical significance test**
   ```
   Chi-square test: p-value < 0.001 → Significant relationship
   ```

**Output**:
- `variable_analysis.csv`: Default rates and lifts for all categories
- Visualizations: Bar charts showing default rates by category
- Ranked list of most predictive variables

**Key Findings Example**:
```
High Risk Categories (Lift > 1.3):
  - grade_G: 40.3% default (Lift 2.57x)
  - purpose_small_business: 26.0% default (Lift 1.65x)

Low Risk Categories (Lift < 0.7):
  - grade_A: 7.3% default (Lift 0.47x)
  - purpose_home_improvement: 10.5% default (Lift 0.67x)
```

---

### **Step 4: Establish Psychological Hypothesis**

**Objective**: Map variables to OCEAN dimensions based on psychological theory

**Mapping Examples**:

| Variable | Category | OCEAN Impact | Rationale |
|----------|----------|--------------|-----------|
| **grade** | A | C ↑, N ↓ | Good credit = Responsible, Emotionally stable |
| | G | C ↓, N ↑ | Poor credit = Less conscientious, Anxious |
| **purpose** | small_business | O ↑, E ↑ | Entrepreneurship = Openness, Extraversion |
| | home_improvement | C ↑, A ↑ | Home care = Conscientious, Agreeable |
| | wedding | A ↑, E ↑ | Social event = Agreeable, Extraverted |
| **term** | 36 months | C ↑ | Short-term = Cautious planning |
| | 60 months | C ↓ | Long-term = Less cautious |
| **home_ownership** | OWN | C ↑, N ↓ | Homeowner = Stable, Responsible |
| | RENT | N ↑ | Renting = Less stable |

**Legend**:
- O = Openness
- C = Conscientiousness
- E = Extraversion
- A = Agreeableness
- N = Neuroticism
- ↑ = Positive influence
- ↓ = Negative influence

---

### **Step 5: Obtain Ground Truth Labels (500+ Samples)**

**Objective**: Generate "correct answers" (OCEAN scores) for a subset of data to train the model

**Method A: Using GenAI (Recommended)**

```python
from text_features.personality_simple import SimplifiedOceanScorer

# Initialize scorer with OpenAI API
scorer = SimplifiedOceanScorer(
    offline_mode=False,  # Enable API
    model="gpt-4o-mini"
)

# Sample 500 borrowers (stratified by default status)
sample_df = df.sample(500, random_state=42)

# Batch score using LLM
ocean_scores = scorer.score_batch(sample_df)

# Save ground truth
for dim in OCEAN_DIMS:
    sample_df[f'{dim}_truth'] = ocean_scores[dim]

sample_df.to_csv('ground_truth_ocean.csv')
```

**Cost**: ~$1-3 (500 samples × $0.002/sample)
**Time**: 10-20 minutes

**Method B: Manual Annotation (Alternative)**

If no API access:
- Recruit psychology experts
- Provide annotation guidelines based on Big Five theory
- Have each expert rate 500 samples
- Average ratings across experts

**Output Format**:
```
borrower_id | grade | purpose | ... | openness_truth | conscientiousness_truth | ...
    1       |   A   |  car    | ... |     0.65       |        0.72             | ...
    2       |   G   | business| ... |     0.75       |        0.28             | ...
   ...
```

---

### **Step 6: Encode Variables (One-Hot Encoding)**

**Objective**: Convert categorical variables into numerical features for regression

**Process**:

**Before Encoding**:
```
grade = 'A'
purpose = 'car'
term = '36 months'
```

**After Encoding**:
```
grade_A = 1, grade_B = 0, grade_C = 0, ..., grade_G = 0
purpose_car = 1, purpose_wedding = 0, purpose_business = 0, ...
term_36months = 1, term_60months = 0
```

**Implementation**:
```python
from sklearn.preprocessing import OneHotEncoder

# Select categorical variables
categorical_vars = ['grade', 'purpose', 'term', 'home_ownership',
                    'emp_length', 'verification_status']

# Fit encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(sample_df[categorical_vars])

# Get feature names
feature_names = encoder.get_feature_names_out(categorical_vars)
# Example: ['grade_A', 'grade_B', ..., 'purpose_car', ...]
```

**Result**:
- Original: 8 categorical variables
- Encoded: ~70 binary features
- Shape: (500 samples, 70 features)

---

### **Step 7: Calculate Weights Using Ridge Regression**

**Objective**: Learn the weight coefficient for each feature's contribution to each OCEAN dimension

**Mathematical Formulation**:

```
minimize: L(w) = Σ(ŷᵢ - yᵢ)² + α·Σwⱼ²
                 i=1 to n      j=1 to p

where:
  ŷᵢ = predicted OCEAN score for sample i
  yᵢ = ground truth OCEAN score (from LLM)
  wⱼ = weight coefficient for feature j
  α  = regularization parameter (prevents overfitting)
  n  = number of samples (500)
  p  = number of features (70)
```

**English Interpretation**:
- **First term** (RSS): Minimize prediction error
- **Second term** (L2 penalty): Prevent weights from becoming too large
- **α (alpha)**: Controls trade-off between fit and regularization

**Implementation**:

```python
from sklearn.linear_model import Ridge

# Train separate model for each OCEAN dimension
learned_weights = {}

for dim in ['openness', 'conscientiousness', 'extraversion',
            'agreeableness', 'neuroticism']:

    # Prepare data
    X = X_encoded  # (500, 70)
    y = sample_df[f'{dim}_truth']  # (500,)

    # Train Ridge regression
    model = Ridge(alpha=0.1, random_state=42)
    model.fit(X, y)

    # Extract weights
    weights = model.coef_  # shape: (70,)
    intercept = model.intercept_

    # Store weights
    learned_weights[dim] = {
        'intercept': intercept,
        'features': dict(zip(feature_names, weights))
    }
```

**Output Example** (Conscientiousness):
```json
{
  "conscientiousness": {
    "intercept": 0.523,
    "features": {
      "grade_A": 0.2873,      // A-grade increases by 0.287
      "grade_G": -0.3124,     // G-grade decreases by 0.312
      "purpose_home_improvement": 0.1950,
      "term_36months": 0.1420,
      "home_ownership_OWN": 0.1080,
      ...
    }
  }
}
```

**Interpretation**:
- **Positive weight**: Feature increases OCEAN score
- **Negative weight**: Feature decreases OCEAN score
- **Near-zero weight**: Feature has minimal impact

**Validation**:
```python
# Cross-validation R² score
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
# Good model: R² > 0.5
```

---

### **Step 8: Create Scoring Algorithm Based on Learned Weights**

**Objective**: Use learned weights to score new borrowers

**Algorithm**:

```python
def compute_ocean_from_weights(borrower, learned_weights, encoder):
    """
    Calculate OCEAN scores using learned weights

    Args:
        borrower: Dict with categorical features
        learned_weights: Dict from Ridge regression
        encoder: Fitted OneHotEncoder

    Returns:
        Dict with OCEAN scores (0-1 scale)
    """
    # Step 1: One-Hot encode borrower
    features_encoded = encoder.transform([borrower])
    feature_names = encoder.get_feature_names_out()

    # Step 2: Calculate score for each OCEAN dimension
    ocean_scores = {}

    for dim in OCEAN_DIMS:
        # Start with intercept
        score = learned_weights[dim]['intercept']

        # Add weighted sum of features
        for i, feat_name in enumerate(feature_names):
            if feat_name in learned_weights[dim]['features']:
                weight = learned_weights[dim]['features'][feat_name]
                score += weight * features_encoded[0, i]

        # Clip to reasonable range [0.25, 0.75]
        score = max(0.25, min(0.75, score))

        ocean_scores[dim] = score

    return ocean_scores
```

**Example Usage**:
```python
new_borrower = {
    'grade': 'B',
    'purpose': 'debt_consolidation',
    'term': '36 months',
    'home_ownership': 'RENT',
    'emp_length': '5 years',
    'verification_status': 'Verified'
}

ocean = compute_ocean_from_weights(new_borrower, learned_weights, encoder)
# Output: {'openness': 0.52, 'conscientiousness': 0.64,
#          'extraversion': 0.48, 'agreeableness': 0.55,
#          'neuroticism': 0.42}
```

---

### **Step 9: Compare Baseline Model vs Enhanced Model**

**Objective**: Validate that OCEAN features improve predictive performance

**Experimental Design**:

**Model A (Baseline)**:
- Features: Traditional financial + categorical variables
  - Numeric: loan_amnt, int_rate, annual_inc, dti, etc. (11 features)
  - Categorical: grade, purpose, term, etc. (8 features)
  - **Total**: 19 features

**Model B (Enhanced)**:
- Features: Baseline + OCEAN dimensions
  - Baseline features (19)
  - OCEAN: openness, conscientiousness, extraversion, agreeableness, neuroticism (5)
  - **Total**: 24 features

**Methodology**:
1. Train/test split: 80/20, stratified by default status, random_state=42
2. Model: XGBoost with identical hyperparameters
3. Evaluation metrics:
   - ROC-AUC (primary)
   - PR-AUC (important for imbalanced data)
   - KS statistic
   - Brier score
   - Expected Calibration Error (ECE)

**Statistical Testing**:
- **DeLong test**: Compare ROC-AUC between models
- **Bootstrap confidence intervals**: PR-AUC and KS
- **5-fold cross-validation**: Assess stability

**Implementation**:
```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# Model A: Baseline
model_A = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
model_A.fit(X_train[baseline_features], y_train)
y_proba_A = model_A.predict_proba(X_test[baseline_features])[:, 1]

# Model B: Baseline + OCEAN
model_B = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
model_B.fit(X_train[baseline_features + OCEAN_DIMS], y_train)
y_proba_B = model_B.predict_proba(X_test[baseline_features + OCEAN_DIMS])[:, 1]

# Evaluate
roc_auc_A = roc_auc_score(y_test, y_proba_A)
roc_auc_B = roc_auc_score(y_test, y_proba_B)

print(f"Model A (Baseline): ROC-AUC = {roc_auc_A:.4f}")
print(f"Model B (+OCEAN):   ROC-AUC = {roc_auc_B:.4f}")
print(f"Improvement:        Δ = {roc_auc_B - roc_auc_A:+.4f}")
```

**Success Criteria** (at least one):
- ROC-AUC improvement ≥ +0.010
- PR-AUC improvement ≥ +0.008
- KS improvement ≥ +1.0

**Expected Results** (based on similar studies):
```
Metric          Model A      Model B      Delta       Significant?
─────────────────────────────────────────────────────────────────
ROC-AUC         0.683        0.695        +0.012      Yes (p<0.05)
PR-AUC          0.264        0.271        +0.007      Yes (p<0.05)
KS              58.3         60.1         +1.8        Yes
```

---

### **Step 10: Analyze Interpretability**

**Objective**: Understand how OCEAN features contribute to predictions and validate psychological consistency

**Analysis Methods**:

#### **10.1 Feature Importance (XGBoost)**

```python
# Get feature importances
importances = model_B.feature_importances_
feature_names_full = baseline_features + OCEAN_DIMS

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names_full,
    'importance': importances
}).sort_values('importance', ascending=False)

# Check OCEAN ranking
ocean_ranks = importance_df[importance_df['feature'].isin(OCEAN_DIMS)]
print("OCEAN Feature Ranking:")
print(ocean_ranks)
```

**Expected Output**:
```
OCEAN Feature Ranking:
            feature  importance  rank
conscientiousness      0.0847     8
openness              0.0623    14
neuroticism           0.0521    18
agreeableness         0.0412    22
extraversion          0.0385    24
```

**Interpretation**:
If OCEAN features appear in top 20-30, they provide meaningful signal.

#### **10.2 SHAP Analysis (Global + Local)**

**Global Interpretation**:
```python
import shap

# Create explainer
explainer = shap.TreeExplainer(model_B)
shap_values = explainer(X_test_transformed)

# Summary plot (beeswarm)
shap.summary_plot(shap_values, X_test_transformed,
                  feature_names=feature_names_full, max_display=20)
```

**Questions to Answer**:
- Do OCEAN features appear in top 20 SHAP values?
- What is the direction of influence (positive/negative)?

**Local Interpretation** (Individual Prediction):
```python
# Select a high-risk prediction
high_risk_idx = y_proba_B.argmax()

# SHAP force plot
shap.force_plot(
    explainer.expected_value,
    shap_values.values[high_risk_idx],
    X_test_transformed[high_risk_idx],
    feature_names=feature_names_full
)
```

**Insights**:
- For high-risk borrowers: Is low conscientiousness a contributing factor?
- For low-risk borrowers: Is high conscientiousness protective?

#### **10.3 Psychological Consistency Check**

**Hypothesis Testing**:
```python
# Hypothesis 1: Conscientiousness inversely correlates with default
corr_C = df['conscientiousness'].corr(df['target'])
print(f"Conscientiousness vs Default: r = {corr_C:.3f}")
# Expected: r < -0.05 (negative correlation)

# Hypothesis 2: Neuroticism positively correlates with default
corr_N = df['neuroticism'].corr(df['target'])
print(f"Neuroticism vs Default: r = {corr_N:.3f}")
# Expected: r > +0.05 (positive correlation)
```

**Compare OCEAN by Default Status**:
```python
ocean_by_status = df.groupby('target')[OCEAN_DIMS].mean()
print("\nOCEAN Scores by Default Status:")
print(ocean_by_status)

# Compute difference
diff = ocean_by_status.loc[1] - ocean_by_status.loc[0]
print("\nDifference (Defaulted - Paid):")
for dim in OCEAN_DIMS:
    direction = "↑" if diff[dim] > 0 else "↓"
    print(f"  {dim:20s}: {diff[dim]:+.3f} {direction}")
```

**Expected Patterns**:
```
Defaulted borrowers should have:
  conscientiousness:   -0.08 ↓  (Lower, as expected)
  neuroticism:         +0.06 ↑  (Higher, as expected)
  openness:            +0.02 ↑  (Risk-taking)
  agreeableness:       -0.01 ↓  (Less cooperative)
  extraversion:         0.00 →  (Neutral)
```

#### **10.4 Partial Dependence Plots**

```python
from sklearn.inspection import PartialDependenceDisplay

# Plot partial dependence for OCEAN features
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
PartialDependenceDisplay.from_estimator(
    model_B, X_train, OCEAN_DIMS, ax=ax.flatten()
)
plt.suptitle('Partial Dependence: OCEAN → Default Probability')
plt.tight_layout()
```

**Interpretation**:
Shows how default probability changes as each OCEAN dimension varies (holding others constant).

---

## 📊 Summary Workflow Diagram

```
┌──────────────────────────────────────────────────────────┐
│ 1. Identify Variables                                     │
│    → grade, purpose, term, home_ownership, etc.          │
└───────────────────┬──────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ 2. Combine Context                                        │
│    → "grade=C | purpose=debt | term=60mo | ..."          │
└───────────────────┬──────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ 3. Analyze Variable-Target Relationship                  │
│    → Lift, Chi-square test, Default rates               │
└───────────────────┬──────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ 4. Psychological Hypothesis                               │
│    → grade_A → C↑, grade_G → N↑, etc.                   │
└───────────────────┬──────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ 5. Ground Truth (LLM/Manual, 500 samples)                │
│    → openness_truth, conscientiousness_truth, ...        │
│    Cost: $1-3 | Time: 10-20 min                          │
└───────────────────┬──────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ 6. One-Hot Encoding                                       │
│    → grade='A' → [1,0,0,...] (~70 features)              │
└───────────────────┬──────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ 7. Ridge Regression                                       │
│    → L(w) = Σ(ŷ-y)² + α·Σw²                              │
│    → Learned weights for each OCEAN dimension            │
└───────────────────┬──────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ 8. Create Scoring Algorithm                               │
│    → OCEAN = Σ(weight × feature) + intercept             │
└───────────────────┬──────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ 9. A/B Comparison                                         │
│    → Baseline vs Baseline+OCEAN                          │
│    → Target: ROC-AUC +0.010 or PR-AUC +0.008             │
└───────────────────┬──────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│ 10. Interpretability Analysis                             │
│     → Feature Importance, SHAP, Correlation              │
│     → Validate psychological consistency                 │
└──────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Advantages of This Methodology

1. **Data-Driven**: Weights learned from data, not subjective
2. **Scientifically Rigorous**: Ridge regression with statistical validation
3. **Interpretable**: Each weight has clear meaning
4. **Reproducible**: Fixed random seeds, documented hyperparameters
5. **Cost-Effective**: ~$2 for ground truth labeling
6. **Scalable**: Once weights learned, scoring is instant

---

## 📈 Expected Outcomes

Based on similar research (Yu et al. 2023):
- ROC-AUC improvement: +0.01 to +0.03
- PR-AUC improvement: +0.008 to +0.02
- Feature importance rank: OCEAN in top 20-30 features

---

## 📚 References

1. Yu et al. (2023). "Chatbot or Human? Using ChatGPT to Extract Personality Traits and Credit Scoring"
2. Costa & McCrae (1992). "NEO PI-R: Revised NEO Personality Inventory"
3. Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP)
4. Hoerl & Kennard (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems"

---

**Document Version**: 1.0
**Last Updated**: 2025-10-01
**Status**: ✅ Ready for Implementation
