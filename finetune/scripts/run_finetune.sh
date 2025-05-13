#!/bin/bash

# Run LoRA fine-tuning script with the Hugging Face token
python lora_finetune.py \
    --model_name "meta-llama/Llama-3.1-8B" \
    --hf_token "" \
    --data_path "../data/pubmed_contraception_abstracts.json" \
    --output_dir "../output" \
    --lora_r 8 \
    --lora_alpha 16 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --device cuda \
    --fp16 \
    --gradient_checkpointing 