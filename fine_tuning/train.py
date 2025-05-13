import json
import torch
from datasets import Dataset
import evaluate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    IntervalStrategy
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # We'll use Mistral-7B as our base model
DATASET_PATH = "pubmed_contraception_abstracts2.json"  # Adjusted for Colab: place data in the same dir or provide full path
OUTPUT_DIR = "./results"
MAX_LENGTH = 512
BATCH_SIZE = 4  # Adjusted for GPU (e.g., T4 on Colab). May need tuning.
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
EVAL_STEPS = 100  # Evaluate every 100 steps

def load_and_process_data(file_path):
    """Load and process the JSON data into a format suitable for training."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Print the structure of the first item to debug
    print("First item structure:", json.dumps(data[0], indent=2))
    
    # Convert to the format expected by the model
    processed_data = []
    for item in data:
        # Extract the abstract text - adjust the key based on actual JSON structure
        abstract_text = item.get('abstract', '')  # Try 'abstract' first
        if not abstract_text:
            # Try alternative keys that might contain the abstract
            for key in ['Abstract', 'abstractText', 'text', 'content']:
                if key in item:
                    abstract_text = item[key]
                    break
        
        if abstract_text:
            # Create a prompt-completion pair
            text = f"Abstract: {abstract_text}\n"
            processed_data.append({"text": text})
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(processed_data, test_size=0.1, random_state=42)
    
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

def prepare_model_and_tokenizer():
    """Prepare the model and tokenizer with LoRA configuration."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- 4-bit Quantization Configuration ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, # Compute type during matrix multiplication
        bnb_4bit_quant_type="nf4",        # Use NF4 (recommended)
        bnb_4bit_use_double_quant=True,  # Optional: Use nested quantization for more memory saving
    )
    # ---------------------------------------
    
    # Load model for GPU with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config, # Apply 4-bit config
        torch_dtype=torch.float16,  # Still specify compute dtype if not using bnb_4bit_compute_dtype
        device_map="auto"           # Automatically maps model to available GPU
    )

    # Ensure model uses gradient checkpointing after loading and quantization
    # Note: prepare_model_for_kbit_training can also enable this, but we'll also set it in TrainingArgs
    model.gradient_checkpointing_enable()

    # Prepare model for k-bit training IS necessary when using bitsandbytes quantization
    model = prepare_model_for_kbit_training(model) 
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Standard rank for LoRA on GPU, can be tuned
        lora_alpha=32, # Standard alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Typical for Mistral/Llama
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def compute_metrics(eval_preds):
    """Compute evaluation metrics."""
    metric = evaluate.load("perplexity")
    logits, labels = eval_preds
    
    predictions = np.argmax(logits, axis=-1)

    # Filter out ignored indices (-100)
    valid_labels = labels[labels != -100]
    valid_logits = logits[labels != -100]

    results = metric.compute(predictions=logits, references=labels)

    perplexity_value = results.get("mean_perplexity", results.get("perplexity"))
    
    return {
        "perplexity": perplexity_value,
    }

def main():
    # Load and process data
    # Load and process data
    print("Loading and processing data...")
    train_dataset, val_dataset = load_and_process_data(DATASET_PATH)

    # Prepare model and tokenizer
    print("Preparing model and tokenizer...")
    model, tokenizer = prepare_model_and_tokenizer()

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Training arguments
    training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            eval_accumulation_steps=16,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            learning_rate=LEARNING_RATE,
            fp16=True,
            optim="paged_adamw_8bit",
            eval_strategy="steps", #IntervalStrategy.STEPS
            eval_steps=EVAL_STEPS,
            save_strategy="steps",#IntervalStrategy.STEPS
            save_steps=EVAL_STEPS,
            save_total_limit=2,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="perplexity",
            greater_is_better=False,
            logging_steps=10,
            report_to="tensorboard"
        )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Print final metrics
    final_metrics = trainer.evaluate()
    print("Final evaluation metrics:", final_metrics)

if __name__ == "__main__":
    main() 