# Hugging Face AutoTrain Guide for Cross-Encoder Fine-Tuning

**Purpose**: Train 25 Cross-Encoder LoRA models for OCEAN prediction using HF AutoTrain

**Cost**: ~$50-75 USD total ($2-3 per model)

**Time**: ~5-8 hours total (models train in parallel)

---

## Prerequisites

1. **Hugging Face Account**: Create at https://huggingface.co/join
2. **Payment Method**: Add to https://huggingface.co/settings/billing
3. **Training Data**: Run `05f_crossencoder_prepare_training_data.ipynb` first

---

## Option A: Web UI (Recommended for Beginners)

### Step 1: Access AutoTrain

1. Go to: https://huggingface.co/spaces/autotrain-projects/autotrain-advanced
2. Click **"Duplicate this Space"** to create your own instance
3. Choose hardware: **T4 GPU** ($0.60/hour) or **A10G** ($1.10/hour)

### Step 2: Create New Project

1. Click **"New Project"**
2. Project name: `crossencoder-ocean-llama-openness` (example)
3. Task type: **Text Regression**
4. Base model: `cross-encoder/nli-deberta-v3-large`

### Step 3: Upload Data

1. Click **"Upload Dataset"**
2. Upload: `crossencoder_train_llama_openness.csv`
3. Verify columns:
   - `text_1`: Query (OCEAN definition)
   - `text_2`: Document (loan description)
   - `label`: Target score (0-1)

### Step 4: Configure Training

**Basic Settings**:
```
Learning Rate: 2e-5
Number of Epochs: 3
Batch Size: 8
Train/Eval Split: 80/20
```

**LoRA Settings** (Enable for parameter efficiency):
```
☑ Use LoRA
LoRA r: 8
LoRA alpha: 32
LoRA dropout: 0.1
Target modules: query,key,value
```

**Advanced Settings**:
```
Warmup ratio: 0.1
Weight decay: 0.01
Gradient accumulation: 1
FP16: Yes (for speed)
```

### Step 5: Start Training

1. Review configuration
2. Click **"Start Training"**
3. Monitor progress in logs
4. Training time: 10-20 minutes per model

### Step 6: Save Model

1. After training completes, click **"Push to Hub"**
2. Model name: `your-username/crossencoder-lora-llama-openness`
3. Make it public or private

### Step 7: Repeat for All 25 Models

**Tip**: You can run multiple AutoTrain spaces in parallel!

1. Duplicate the space 5 times
2. Train 5 models simultaneously
3. Saves time (but costs more per hour)

---

## Option B: AutoTrain CLI (Advanced)

### Installation

```bash
pip install autotrain-advanced
huggingface-cli login
```

### Single Model Training

```bash
autotrain text-regression \
  --model cross-encoder/nli-deberta-v3-large \
  --data ../crossencoder_training_data/crossencoder_train_llama_openness.csv \
  --text-column1 text_1 \
  --text-column2 text_2 \
  --target-column label \
  --lr 2e-5 \
  --epochs 3 \
  --batch-size 8 \
  --warmup-ratio 0.1 \
  --trainer sft \
  --peft \
  --lora-r 8 \
  --lora-alpha 32 \
  --project-name crossencoder-lora-llama-openness \
  --username your-hf-username \
  --push-to-hub
```

### Batch Training Script

Create `train_all_models.sh`:

```bash
#!/bin/bash

# Array of LLMs and OCEAN dimensions
LLMS=("llama" "gpt" "gemma" "deepseek" "qwen")
DIMS=("openness" "conscientiousness" "extraversion" "agreeableness" "neuroticism")

for llm in "${LLMS[@]}"; do
  for dim in "${DIMS[@]}"; do
    echo "Training: $llm - $dim"

    autotrain text-regression \
      --model cross-encoder/nli-deberta-v3-large \
      --data ../crossencoder_training_data/crossencoder_train_${llm}_${dim}.csv \
      --text-column1 text_1 \
      --text-column2 text_2 \
      --target-column label \
      --lr 2e-5 \
      --epochs 3 \
      --batch-size 8 \
      --warmup-ratio 0.1 \
      --trainer sft \
      --peft \
      --lora-r 8 \
      --lora-alpha 32 \
      --project-name crossencoder-lora-${llm}-${dim} \
      --username YOUR_USERNAME \
      --push-to-hub

    echo "Completed: $llm - $dim"
    echo "---"
  done
done

echo "All 25 models trained!"
```

Run:
```bash
chmod +x train_all_models.sh
./train_all_models.sh
```

---

## Option C: Python API (Most Flexible)

```python
from huggingface_hub import HfApi, create_repo
import subprocess
import os

api = HfApi()

# Model configurations
LLMS = ["llama", "gpt", "gemma", "deepseek", "qwen"]
DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Your HF username
HF_USERNAME = "your-username"

for llm in LLMS:
    for dim in DIMS:
        project_name = f"crossencoder-lora-{llm}-{dim}"
        data_file = f"../crossencoder_training_data/crossencoder_train_{llm}_{dim}.csv"

        print(f"Training: {project_name}")

        # Create repository
        repo_id = f"{HF_USERNAME}/{project_name}"
        try:
            create_repo(repo_id, private=False, exist_ok=True)
        except:
            pass

        # Train via CLI
        cmd = f"""
        autotrain text-regression \
          --model cross-encoder/nli-deberta-v3-large \
          --data {data_file} \
          --text-column1 text_1 \
          --text-column2 text_2 \
          --target-column label \
          --lr 2e-5 \
          --epochs 3 \
          --batch-size 8 \
          --trainer sft \
          --peft \
          --lora-r 8 \
          --lora-alpha 32 \
          --project-name {project_name} \
          --username {HF_USERNAME} \
          --push-to-hub
        """

        subprocess.run(cmd, shell=True, check=True)
        print(f"✓ Completed: {project_name}\n")

print("All 25 models trained and pushed to Hub!")
```

---

## Cost Breakdown

### Per Model

| Component | Cost |
|-----------|------|
| GPU time (T4, 15 min) | $0.15 |
| Storage (LoRA weights, ~50MB) | $0.00 |
| Inference API (first 100K) | $0.00 |
| **Total per model** | **~$0.15-0.30** |

### Total Project

| Scenario | Cost |
|----------|------|
| Sequential (1 GPU) | $4-7 |
| Parallel (5 GPUs) | $10-15 |
| Premium GPU (A10G) | $15-30 |

**Note**: AutoTrain also charges a small platform fee (~$1-2 per model)

**Estimated Total**: $50-75 for all 25 models

---

## Monitoring Training

### Web UI
- Watch real-time logs in AutoTrain interface
- Check loss curves
- Monitor eval metrics (R², RMSE)

### CLI/API
```bash
# Check training status
autotrain app --port 8080

# View logs
tail -f autotrain.log
```

### Hugging Face Hub
- Models appear at: `https://huggingface.co/your-username/model-name`
- View training metrics in model card

---

## After Training

### Verify Models

Check that all 25 models are on your HF profile:
```
your-username/crossencoder-lora-llama-openness
your-username/crossencoder-lora-llama-conscientiousness
...
your-username/crossencoder-lora-qwen-neuroticism
```

### Download for Local Evaluation

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification

# Load LoRA model
config = PeftConfig.from_pretrained("your-username/crossencoder-lora-llama-openness")
base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, "your-username/crossencoder-lora-llama-openness")

# Merge LoRA weights (optional, for faster inference)
model = model.merge_and_unload()
```

### Next Step

Run `05f_crossencoder_lora_evaluate.ipynb` to:
- Load all 25 trained models
- Evaluate on test set
- Compare with BGE, DeBERTa, and baseline methods
- Generate final performance report

---

## Troubleshooting

### Error: "Model loading failed"

**Solution**: Wait 2-3 minutes after starting AutoTrain. Large models take time to load.

### Error: "CUDA out of memory"

**Solution**: Reduce batch size to 4 or use gradient accumulation:
```
--batch-size 4 --gradient-accumulation 2
```

### Error: "Training loss not decreasing"

**Possible causes**:
1. Learning rate too high → Try 1e-5
2. Too few epochs → Increase to 5-7
3. Data quality issues → Check label distributions

### Error: "Rate limit exceeded"

**Solution**: AutoTrain has rate limits. Train in batches:
- Batch 1: 5 models
- Wait 1 hour
- Batch 2: 5 models
- ...

---

## Tips for Better Performance

1. **Start with one LLM**: Train 5 models for best-performing LLM first (e.g., Gemma)
2. **Monitor first model closely**: Adjust hyperparameters if needed
3. **Use validation loss**: Set aside 20% for validation, stop if overfitting
4. **Try different epochs**: If 3 epochs underfit, increase to 5
5. **Save checkpoints**: Enable in AutoTrain to resume if interrupted

---

## Alternative: Google Colab (Free GPU)

If budget is tight, use Colab's free T4 GPU:

```python
# In Colab notebook
!pip install transformers peft accelerate datasets

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import pandas as pd

# Load data
df = pd.read_csv("crossencoder_train_llama_openness.csv")

# Prepare dataset
from datasets import Dataset
dataset = Dataset.from_pandas(df)

# Load model and apply LoRA
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-large", num_labels=1)
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["query", "key", "value"])
model = get_peft_model(model, lora_config)

# Train
# ... (see full code in 05f_crossencoder_colab_training.ipynb)
```

**Pros**: Free
**Cons**:
- Must run 25 times manually
- Session timeout after 12 hours
- Slower than AutoTrain

---

## Summary

**Recommended Workflow**:

1. ✅ **Prepare data**: Run `05f_crossencoder_prepare_training_data.ipynb`
2. 🚀 **Train models**: Use AutoTrain Web UI (easiest)
3. 💰 **Cost**: Budget $50-75, can be done in batches
4. ⏱️ **Time**: 5-8 hours total (parallel training)
5. 📊 **Evaluate**: Run `05f_crossencoder_lora_evaluate.ipynb`

**Questions?** Check:
- AutoTrain docs: https://huggingface.co/docs/autotrain/
- PEFT docs: https://huggingface.co/docs/peft/
- Discord: https://discord.gg/hugging-face
