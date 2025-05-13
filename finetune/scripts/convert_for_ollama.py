#!/usr/bin/env python3
# Convert fine-tuned model for use with Ollama

import os
import shutil
import argparse
import subprocess
from pathlib import Path

def create_modelfile(output_dir, base_model, model_name, system_prompt, template_format="llama3"):
    """Create a Modelfile for Ollama"""
    
    templates = {
        "llama3": """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{prompt}

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>
""",
        "simple": """
<s>[INST] {system_prompt}

{prompt} [/INST]

{response}
"""
    }
    
    template = templates.get(template_format, templates["llama3"])
    
    # Create multiline string with proper escaping
    modelfile_content = f"""FROM {base_model}

# Use the LoRA adapter weights
ADAPTER {output_dir}/merged-adapter

# Set the system prompt
SYSTEM """

    # Add the system prompt as a separate string
    modelfile_content += f'"""{system_prompt}"""'
    
    # Continue with the rest of the content
    modelfile_content += f"""

# Set the model template
TEMPLATE """

    # Add the template as a separate string
    modelfile_content += f'"""{template}"""'
    
    # Add the parameters
    modelfile_content += """

# Set model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
    
    # Write the Modelfile
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"Created Modelfile at {modelfile_path}")
    return modelfile_path

def merge_lora_adapter(model_path, base_model, output_dir):
    """Merge the LoRA adapter with the base model"""
    merged_dir = os.path.join(output_dir, "merged-adapter")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Copy the adapter model files
    for filename in os.listdir(model_path):
        if filename.endswith(".bin") or filename.endswith(".json") or filename.endswith(".safetensors"):
            shutil.copy(
                os.path.join(model_path, filename),
                os.path.join(merged_dir, filename)
            )
    
    # Create a configuration file to indicate this is a LoRA adapter for the base model
    config_file = os.path.join(merged_dir, "adapter_config.json")
    with open(config_file, "w") as f:
        f.write(f'{{"base_model_name": "{base_model}"}}')
    
    print(f"Prepared LoRA adapter in {merged_dir}")
    return merged_dir

def build_ollama_model(modelfile_path, model_name):
    """Build the model in Ollama"""
    try:
        # Check if Ollama is available
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        
        # Build the model
        cmd = ["ollama", "create", model_name, "-f", modelfile_path]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        print(f"\nSuccess! Model '{model_name}' has been created in Ollama.")
        print(f"You can use it with: ollama run {model_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error building Ollama model: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
    except FileNotFoundError:
        print("Error: Ollama not found. Please make sure Ollama is installed and available in your PATH.")

def main():
    parser = argparse.ArgumentParser(description="Convert fine-tuned model for use with Ollama")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="llama3.1", help="Base Ollama model name")
    parser.add_argument("--output_dir", type=str, default="../output/ollama", help="Output directory for Ollama files")
    parser.add_argument("--model_name", type=str, default="llama3.1-contraceptive", help="Name for the Ollama model")
    parser.add_argument("--system_prompt", type=str, 
                      default="You are a women's health contraceptive research assistant trained to provide medically accurate information about contraception based on scientific research. Use clear, factual language to explain complex medical concepts related to contraceptives. When discussing side effects, efficacy rates, or other statistical data, be precise and evidence-based. Provide balanced information about both benefits and risks. If uncertain about any details, acknowledge the limitations of your knowledge rather than speculating.", 
                      help="System prompt for the model")
    parser.add_argument("--template_format", type=str, default="llama3", choices=["llama3", "simple"], help="Template format to use")
    parser.add_argument("--build", action="store_true", help="Build the model in Ollama after conversion")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Merge the LoRA adapter
    merge_lora_adapter(args.model_path, args.base_model, args.output_dir)
    
    # Create the Modelfile
    modelfile_path = create_modelfile(
        args.output_dir, 
        args.base_model, 
        args.model_name, 
        args.system_prompt,
        args.template_format
    )
    
    # Build the model in Ollama if requested
    if args.build:
        build_ollama_model(modelfile_path, args.model_name)
    else:
        print(f"\nTo build the model in Ollama, run:")
        print(f"ollama create {args.model_name} -f {modelfile_path}")
        print(f"\nTo use the model after building:")
        print(f"ollama run {args.model_name}")

if __name__ == "__main__":
    main() 