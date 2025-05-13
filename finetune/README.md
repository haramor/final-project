# Fine-tuning Llama 3 with LoRA for Women's Health Contraceptive Research

This directory contains scripts for fine-tuning the Llama 3 model with LoRA (Low-Rank Adaptation) to create a specialized model for women's health contraceptive research.

## Overview

The fine-tuning process involves:

1. Preprocessing PubMed contraception abstracts
2. Fine-tuning Llama 3 with LoRA
3. Evaluating the fine-tuned model
4. Converting the fine-tuned model for use with Ollama

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning) library
- HuggingFace datasets library
- Access to an Ollama instance running Llama 3

## Installation

Install the required Python packages:

```bash
cd finetune
pip install -r requirements.txt
```

## Directory Structure

```
finetune/
├── data/
│   └── pubmed_contraception_abstracts.json  # Raw PubMed abstracts
│   └── processed_abstracts.json  # Processed abstracts for fine-tuning
├── scripts/
│   ├── preprocess_data.py  # Script to preprocess the abstracts
│   ├── lora_finetune.py  # Script to fine-tune the model with LoRA
│   ├── evaluate_model.py  # Script to evaluate the fine-tuned model
│   └── convert_for_ollama.py  # Script to convert the model for Ollama
├── output/  # Directory for the fine-tuned model and evaluation results
├── requirements.txt  # Python dependencies
└── README.md  # This file
```

## Usage

### 1. Preprocess the Data

```bash
cd finetune/scripts
python preprocess_data.py --input_file ../data/pubmed_contraception_abstracts.json --output_file ../data/processed_abstracts.json
```

This will:
- Clean and normalize the PubMed abstracts
- Extract structured sections (Background, Methods, Results, Conclusion)
- Save the processed abstracts to a new JSON file

### 2. Fine-tune the Model with LoRA

```bash
python lora_finetune.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --data_path ../data/processed_abstracts.json \
  --output_dir ../output/llama3-finetuned \
  --num_epochs 3 \
  --batch_size 8 \
  --use_8bit
```

You can adjust the parameters:
- `--model_name`: Base model to use (should be accessible through Ollama)
- `--lora_r`: LoRA attention dimension (default: 8)
- `--lora_alpha`: LoRA alpha parameter (default: 16)
- `--learning_rate`: Learning rate for training (default: 3e-4)
- `--use_4bit`: Enable 4-bit quantization
- `--use_8bit`: Enable 8-bit quantization

### 3. Evaluate the Fine-tuned Model

```bash
python evaluate_model.py \
  --model_path ../output/llama3-finetuned \
  --data_path ../data/processed_abstracts.json \
  --output_file ../output/evaluation_results.json \
  --num_samples 5
```

This will:
- Load the fine-tuned model
- Generate summaries for a random sample of abstracts
- Save the results to a JSON file

### 4. Convert for Ollama

After fine-tuning, you can convert the model for use with Ollama:

```bash
python convert_for_ollama.py \
  --model_path ../output/llama3-finetuned \
  --base_model llama3 \
  --output_dir ../output/ollama \
  --model_name llama3-contraceptive \
  --build
```

Parameters:
- `--model_path`: Path to the fine-tuned model directory
- `--base_model`: Name of the base model in Ollama (e.g., "llama3")
- `--output_dir`: Directory to store the Ollama files
- `--model_name`: Name for the fine-tuned model in Ollama
- `--system_prompt`: System prompt for the model
- `--build`: Flag to build the model in Ollama automatically

After running this script, you can use the model with:

```bash
ollama run llama3-contraceptive
```

## Tips for Better Results

- **Data Quality**: Using high-quality, domain-specific abstracts improves performance
- **Hyperparameter Tuning**: Experiment with different LoRA configurations and learning rates
- **Instruction Format**: The prompt format is critical for good results
- **Evaluation**: Regularly evaluate the model to avoid overfitting

## License

This project is provided under the MIT License. 