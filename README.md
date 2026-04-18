# Fine-tuning LoRA for Text-to-SQL with TinyLlama

This project demonstrates fine-tuning a small language model (TinyLlama-1.1B) using LoRA (Low-Rank Adaptation) for text-to-SQL generation.

## Setup

### Prerequisites

- Python 3.8+
- uv (fast Python package manager)

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Virtual Environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Requirements

```bash
uv pip install -r requirements.txt
```

## Project Structure

- `text_sql_lora.ipynb`: Main notebook for fine-tuning
- `tinyllama-sql-lora/`: Checkpoint directory with trained model
- `tinyllama-sql-lora-adapter/`: LoRA adapters only

## Notebook Summary

The notebook `text_sql_lora.ipynb` guides through the process of fine-tuning TinyLlama for text-to-SQL tasks using pure LoRA.

### Key Sections:

1. **GPU Check**: Verifies available GPU and VRAM
2. **Load Base Model**: Loads TinyLlama-1.1B in FP16
3. **Base Model Testing**: Tests the model before fine-tuning on text-to-SQL prompts
4. **Dataset Preparation**: Prepares a small dataset of 100 examples from WikiSQL
5. **LoRA Training**: Attaches LoRA adapters and trains the model
6. **Fine-tuned Model Testing**: Tests the same prompts after training
7. **Save Adapters**: Saves the LoRA weights (~10MB)
8. **Inference**: Demonstrates loading and using the saved adapters

### Key Takeaways:

- LoRA enables efficient fine-tuning with minimal parameters (only adapters trained)
- Small datasets can achieve good results for specific tasks
- Adapters are lightweight and portable across similar base models
- Pure LoRA (no quantization) provides clear understanding of the technique
- Suitable for GPUs with 16GB VRAM

## Running the Notebook

1. Activate the virtual environment
2. Open `text_sql_lora.ipynb` in Jupyter or VS Code
3. Run cells sequentially

Note: Requires GPU with at least 16GB VRAM for FP16 training.

## Checkpoints

- `tinyllama-sql-lora/`: Full checkpoint after training (includes optimizer state, etc.)
- `tinyllama-sql-lora-adapter/`: LoRA adapters only for inference

To use the adapters for inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(base_model, "./tinyllama-sql-lora-adapter")

# Use the model for text-to-SQL generation
```
