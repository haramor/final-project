#!/usr/bin/env python3
# Evaluate the fine-tuned model on contraception abstracts

import os
import json
import random
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm

def load_model_and_tokenizer(model_path, device="cuda"):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from {model_path}")
    
    # Check if this is a PEFT/LoRA model
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Loading PEFT/LoRA model")
        # Load the PEFT configuration
        config = PeftConfig.from_pretrained(model_path)
        
        # Load the base model
        print(f"Loading base model: {config.base_model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load the LoRA adapter
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # Load the model directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_abstracts(file_path):
    """Load and parse the abstracts file"""
    print(f"Loading abstracts from {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "abstracts" in data:
            abstracts = data["abstracts"]
            print(f"Loaded {len(abstracts)} abstracts from processed JSON")
            return abstracts
        else:
            print("File does not contain 'abstracts' key. Trying alternative format...")
    except json.JSONDecodeError:
        print("File is not valid JSON. Trying to parse as raw text...")
    
    # Try to load as raw text if JSON parsing failed
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split on document boundaries
        raw_abstracts = content.split('","')
        
        # Clean up the abstracts
        abstracts = []
        for abstract in raw_abstracts:
            # Remove JSON artifacts
            abstract = abstract.replace('"}%', '').replace('{"', '')
            
            # Skip empty abstracts
            if not abstract.strip():
                continue
                
            # Remove extra newlines and spaces
            abstract = abstract.replace('\n', ' ')
            abstract = ' '.join(abstract.split())
            
            if abstract:
                abstracts.append(abstract)
        
        print(f"Loaded {len(abstracts)} abstracts from raw text")
        return abstracts
    except Exception as e:
        print(f"Error parsing file: {e}")
        raise
    
def prepare_prompt(abstract):
    """Prepare the prompt for the model"""
    return f"""<s>[INST] You are a women's health contraceptive research assistant trained to provide medically accurate information about contraception. Please summarize the following medical abstract about contraception:

{abstract} [/INST]

"""

def generate_summary(model, tokenizer, abstract, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Generate a summary for the given abstract"""
    prompt = prepare_prompt(abstract)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the summary
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response (after the prompt)
    response = generated_text[len(prompt):]
    
    return response

def evaluate_model(model, tokenizer, abstracts, num_samples=5, output_file=None, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Evaluate the model on a set of abstracts"""
    # Select a random sample of abstracts
    if num_samples > 0 and num_samples < len(abstracts):
        selected_abstracts = random.sample(abstracts, num_samples)
    else:
        selected_abstracts = abstracts
    
    results = []
    
    for i, abstract in enumerate(tqdm(selected_abstracts, desc="Generating summaries")):
        # Generate a summary
        summary = generate_summary(
            model, 
            tokenizer, 
            abstract, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Store the result
        result = {
            "id": i,
            "abstract": abstract,
            "summary": summary
        }
        results.append(result)
        
        # Print the result
        print(f"\n--- Example {i+1} ---")
        print(f"Abstract (truncated): {abstract[:200]}...")
        print(f"Summary: {summary}")
    
    # Save the results to a file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on contraception abstracts")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--data_path", type=str, default="../data/processed_abstracts.json", 
                        help="Path to the processed abstracts file")
    parser.add_argument("--output_file", type=str, default="../output/evaluation_results.json", 
                        help="Path to save the evaluation results")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of abstracts to evaluate (0 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=256, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    args = parser.parse_args()
    
    # Set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory if it doesn't exist
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load the abstracts
    abstracts = load_abstracts(args.data_path)
    
    print(f"Using {args.num_samples if args.num_samples > 0 and args.num_samples < len(abstracts) else len(abstracts)} abstracts for evaluation")
    
    # Evaluate the model
    evaluate_model(
        model, 
        tokenizer, 
        abstracts, 
        num_samples=args.num_samples, 
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

if __name__ == "__main__":
    main() 