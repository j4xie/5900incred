# Credibly Loan Default Prediction with OCEAN Personality Features

INFO 5900 Project - Cornell Tech

## Project Overview

This project enhances loan default prediction using OCEAN (Big Five) personality traits extracted from loan descriptions via Large Language Models (LLMs). We compare multiple LLM models to generate personality features and integrate them into XGBoost models for improved prediction accuracy.

## Key Features

- **OCEAN Personality Extraction**: Automated extraction of Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) from loan descriptions
- **Multi-Model Comparison**: Comprehensive evaluation of 5 different LLM models for personality extraction quality, speed, and cost
- **Advanced Text Features**: BGE (BAAI General Embedding) embeddings for semantic text representation
- **Ridge Regression Pipeline**: Automated mapping from embeddings to OCEAN scores
- **XGBoost Integration**: Enhanced loan default prediction using personality features

## Dataset

- **Source**: Lending Club loan data with text descriptions
- **Size**: Final dataset with 50+ character descriptions and OCEAN features
- **Features**:
  - Traditional loan features (amount, grade, income, etc.)
  - Text embeddings (BGE model)
  - OCEAN personality scores
  - Default status (target variable)

## Model Comparison Results

### LLM Models Evaluated:
1. **Llama-3.1-8B-Instruct**
2. **GPT-OSS-120B**
3. **Qwen2.5-72B-Instruct**
4. **Gemma-2-9B-it**
5. **DeepSeek-V3.1**

### Comparison Metrics:
- **Quality**: Data completeness, diversity, and variance
- **Speed**: Processing time for 500 samples
- **Cost**: API usage costs
- **Value**: Quality-to-cost ratio

For detailed comparison results, see: `ocean_ground_truth/model_comparison_summary.csv`

## Project Structure

```
Credibly-INFO-5900/
├── README.md                           # This file
├── PROJECT_STRUCTURE.md                # Detailed project structure
├── final_dataset_report.md             # Dataset analysis and statistics
│
├── notebooks/                          # Jupyter notebooks for analysis
│   ├── 05d_allmodel_comparison.ipynb  # LLM model comparison
│   ├── 05d_llama_3_8B.ipynb          # Llama model analysis
│   └── ...                            # Other analysis notebooks
│
├── scripts/                            # Python scripts
│   ├── create_ocean_notebooks.py      # Generate model comparison notebooks
│   ├── generate_ocean_ground_truth.py # Create OCEAN ground truth data
│   ├── run_ocean_pipeline.py          # Main OCEAN extraction pipeline
│   └── ocean_router_api.py            # API router for LLM access
│
├── data/                               # Raw and processed data
├── ocean_ground_truth/                 # OCEAN personality ground truth
│   ├── *_ocean_500.csv                # Model-specific OCEAN scores
│   ├── model_comparison_summary.csv   # Comprehensive comparison
│   └── model_rankings.csv             # Rankings across dimensions
│
├── models/                             # Trained models
├── results/                            # Analysis results and outputs
├── text_features/                      # Text feature extractors
├── utils/                              # Utility functions
├── docs/                               # Documentation
├── logs/                               # Execution logs
└── archive_old_files/                  # Archived old versions

```

## Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- API keys for LLM providers (OpenRouter, HuggingFace, etc.)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Credibly-INFO-5900.git
cd Credibly-INFO-5900

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # Create this if needed

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables
```
OPENROUTER_API_KEY=your_openrouter_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

## Usage

### 1. Generate OCEAN Ground Truth
```bash
python generate_ocean_ground_truth.py
```

### 2. Run Model Comparison
Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/05d_allmodel_comparison.ipynb
```

### 3. Execute Full Pipeline
```bash
python run_ocean_pipeline.py
```

## Key Files

### Core Scripts
- `generate_ocean_ground_truth.py`: Generate OCEAN personality scores from text
- `run_ocean_pipeline.py`: End-to-end pipeline for OCEAN extraction
- `ocean_router_api.py`: API routing for multiple LLM providers
- `create_ocean_notebooks.py`: Auto-generate comparison notebooks

### Data Files
- `loan_final_desc50plus_with_ocean_bge.csv`: Final dataset with all features
- `ocean_targets_500.csv`: OCEAN target scores for model training
- `bge_embeddings_500.npy`: BGE text embeddings
- `ridge_models_bge_large.pkl`: Trained Ridge regression models

### Notebooks
- `05d_allmodel_comparison.ipynb`: Comprehensive LLM model comparison with visualizations

## Results Summary

The model comparison analysis includes:
- Quality rankings based on data completeness and diversity
- Speed rankings (processing time)
- Cost efficiency rankings
- Value rankings (quality per dollar)
- Scenario-based recommendations

Best model recommendations:
- **For Quality**: [Model selected based on quality score]
- **For Speed**: Gemma-2-9B (19 minutes for 500 samples)
- **For Cost**: Gemma-2-9B ($0.03 for 500 samples)
- **For Value**: [Model selected based on quality/cost ratio]

## Model Selection Guide

Choose your LLM based on your requirements:

| Scenario | Recommended Model | Reason |
|----------|------------------|---------|
| Academic Research | [Best Quality Model] | Highest data quality |
| Production API | [Best Value Model] | Optimal cost-quality balance |
| Real-time Application | Gemma-2-9B | Fastest processing |
| Prototype/MVP | Gemma-2-9B | Lowest cost |

## Next Steps

1. Select appropriate LLM model based on requirements
2. Generate OCEAN features for full dataset
3. Train XGBoost model with integrated OCEAN features
4. Evaluate prediction performance improvement
5. Deploy model for loan default prediction

## Contributors

- Project Team
- Cornell Tech INFO 5900

## License

[Add appropriate license]

## Acknowledgments

- Lending Club for the dataset
- LLM providers (OpenRouter, HuggingFace, etc.)
- BAAI for BGE embeddings

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
