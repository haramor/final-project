#!/usr/bin/env python3
# Fine-tuning Llama 3 with LoRA on contraception abstracts
# This script uses the PEFT library to perform parameter-efficient fine-tuning

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    set_seed,
    TrainingArguments, 
    Trainer
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from tqdm import tqdm

# Set seeds for reproducibility
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def preprocess_abstract(abstract):
    """Clean and preprocess the abstract text"""
    # Basic cleaning - you may need to expand this based on your data
    if isinstance(abstract, str):
        # Remove extra newlines, replace with a single space
        abstract = abstract.replace('\n', ' ')
        # Remove multiple spaces
        abstract = ' '.join(abstract.split())
        return abstract
    return ""

def load_data(file_path):
    """Load and prepare the dataset"""
    print(f"Loading data from {file_path}")
    
    # Check if the file is in the processed JSON format
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "abstracts" in data:
                print(f"Loading from processed JSON format with {len(data['abstracts'])} abstracts")
                return Dataset.from_dict({"text": data["abstracts"]})
    except (json.JSONDecodeError, KeyError):
        pass
    
    # If not in processed format, try the raw format
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split on document boundaries
        abstracts = content.split('","')
        
        # Clean up the abstracts
        cleaned_abstracts = []
        for abstract in abstracts:
            # Remove JSON artifacts
            abstract = abstract.replace('"}%', '').replace('{"', '')
            
            # Skip empty abstracts
            if not abstract.strip():
                continue
                
            # Preprocess and add to list
            cleaned_abstract = preprocess_abstract(abstract)
            if cleaned_abstract:
                cleaned_abstracts.append(cleaned_abstract)
        
        print(f"Loaded and cleaned {len(cleaned_abstracts)} abstracts")
        
        # Create a dataset
        return Dataset.from_dict({"text": cleaned_abstracts})
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def prepare_prompt(text):
    """Prepare the prompt format for instruction tuning"""
    return f"""<s>[INST] You are a women's health contraceptive research assistant trained to provide medically accurate information about contraception. Please summarize the following medical abstract about contraception:

{text} [/INST]

This abstract discusses """

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the examples and prepare them for training"""
    prompts = [prepare_prompt(text) for text in examples["text"]]
    
    # Tokenize the prompts
    tokenized_prompts = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Prepare the labels (same as input_ids for causal language modeling)
    tokenized_prompts["labels"] = tokenized_prompts["input_ids"].clone()
    
    return tokenized_prompts

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 with LoRA on contraception abstracts")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", help="Base model to use")
    parser.add_argument("--data_path", type=str, default="../data/pubmed_contraception_abstracts.json", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="../output", help="Directory to save the model")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every X steps")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of dataset to use for testing")
    parser.add_argument("--min_test_size", type=int, default=1, help="Minimum number of examples in test set")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token for accessing gated models")
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_data(args.data_path)
    
    # Ensure we have enough data for a meaningful split
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} examples")
    
    # Calculate test size based on dataset size
    if dataset_size <= 10:  # Very small dataset
        # For tiny datasets, use at least 1 example for testing, but keep most for training
        test_size = min(1, max(args.min_test_size, int(dataset_size * 0.1)))
        if test_size >= dataset_size:
            test_size = max(1, dataset_size - 1)  # Ensure at least 1 example for training
        split_ratio = {"train": dataset_size - test_size, "test": test_size}
        print(f"Small dataset detected. Using {split_ratio['train']} examples for training and {split_ratio['test']} for testing.")
        dataset = dataset.train_test_split(train_size=split_ratio["train"], test_size=split_ratio["test"], seed=args.seed)
    else:
        # For larger datasets, use the specified test_size ratio
        print(f"Splitting dataset with test_size={args.test_size}")
        dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    
    print(f"Train set: {len(dataset['train'])} examples")
    print(f"Test set: {len(dataset['test'])} examples")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the dataset
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"]
    )
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    # If CUDA is not available, disable quantization
    if not cuda_available and (args.use_4bit or args.use_8bit):
        print("CUDA is not available. Quantization (4-bit/8-bit) will be disabled.")
        args.use_4bit = False
        args.use_8bit = False
    
    # Set device to CPU if CUDA is not available
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model_kwargs = {}
    if args.use_4bit:
        model_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    elif args.use_8bit:
        model_kwargs = {"load_in_8bit": True}
    
    # For MacOS without CUDA, we need to handle memory more carefully
    if not cuda_available:
        print("Using CPU for training. This might be very slow with a large model.")
        print("Loading smaller model or modifying batch size, sequence length, and model parameters is recommended.")
        # Load model with CPU device_map for Mac (no auto device mapping)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,  # Use float32 instead of float16 without GPU
            device_map={"": device},
            token=args.hf_token,
            **model_kwargs
        )
    else:
        # For CUDA devices, use the auto device mapping
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=args.hf_token,
            **model_kwargs
        )
    
    # Prepare the model for kbit training if using quantization
    if args.use_4bit or args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Define the LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} ({100 * trainable_params / all_params:.2f}% of all params)")
    
    # Determine whether to use fp16 based on CUDA availability
    use_fp16 = cuda_available
    if not use_fp16:
        print("CUDA is not available. fp16 precision will be disabled.")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        fp16=use_fp16,
        push_to_hub=False,
        seed=args.seed,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )
    
    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    print(f"Model saved to {args.output_dir}/final_model")

if __name__ == "__main__":
    main() 