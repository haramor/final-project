#!/usr/bin/env python3
# Preprocess PubMed contraception abstracts for fine-tuning

import os
import json
import re
import argparse
from tqdm import tqdm

def clean_text(text):
    """Clean and normalize text from PubMed abstracts"""
    # Remove JSON and XML artifacts
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\\n', ' ', text)  # Replace escaped newlines
    text = re.sub(r'Ã.\©', 'é', text)  # Fix common encoding issues
    text = re.sub(r'Ã.\'', 'è', text)  # Fix common encoding issues
    text = re.sub(r'ÃÂ[^\s]*', '', text)  # Remove other encoding artifacts
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    
    return text.strip()

def extract_abstracts(file_path, output_file, min_length=100):
    """Extract and preprocess abstracts from the PubMed dataset"""
    print(f"Processing file: {file_path}")
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the content by splitting on potential abstract boundaries
    abstracts = []
    raw_abstracts = content.split('","')
    
    for abstract in tqdm(raw_abstracts, desc="Processing abstracts"):
        # Clean up the abstract
        abstract = abstract.replace('"}%', '').replace('{"', '')
        
        # Extract the main text
        # Looking for patterns like "BACKGROUND:", "METHODS:", "RESULTS:", "CONCLUSION:" which are common in abstracts
        sections = re.findall(r'(?:BACKGROUND|OBJECTIVE|METHODS|RESULTS|CONCLUSION|DESIGN|SETTING|MATERIALS AND METHODS|STUDY DESIGN|PURPOSE|AIM|PATIENTS|INTERVENTIONS|MAIN OUTCOME MEASURES|FINDINGS|INTERPRETATION|FUNDING):\s*(.*?)(?=(?:BACKGROUND|OBJECTIVE|METHODS|RESULTS|CONCLUSION|DESIGN|SETTING|MATERIALS AND METHODS|STUDY DESIGN|PURPOSE|AIM|PATIENTS|INTERVENTIONS|MAIN OUTCOME MEASURES|FINDINGS|INTERPRETATION|FUNDING):|$)', abstract, re.IGNORECASE)
        
        # If we found structured sections, join them
        if sections:
            cleaned_abstract = ' '.join(sections)
        else:
            # Otherwise use the whole abstract
            cleaned_abstract = abstract
        
        # Final cleaning
        cleaned_abstract = clean_text(cleaned_abstract)
        
        # Only keep abstracts with sufficient content
        if len(cleaned_abstract) >= min_length:
            abstracts.append(cleaned_abstract)
    
    print(f"Extracted {len(abstracts)} valid abstracts")
    
    # Save the cleaned abstracts to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"abstracts": abstracts}, f, ensure_ascii=False, indent=2)
    
    print(f"Saved preprocessed data to {output_file}")
    
    # Save a small subset for faster testing if we have enough abstracts
    if len(abstracts) > 10:
        small_sample = abstracts[:10]
        small_output_file = output_file.replace('.json', '_small.json')
        with open(small_output_file, 'w', encoding='utf-8') as f:
            json.dump({"abstracts": small_sample}, f, ensure_ascii=False, indent=2)
        print(f"Saved small sample (10 abstracts) to {small_output_file} for quick testing")
    
    return abstracts

def main():
    parser = argparse.ArgumentParser(description="Preprocess PubMed contraception abstracts")
    parser.add_argument("--input_file", type=str, default="../data/pubmed_contraception_abstracts.json", 
                        help="Path to the input PubMed abstracts file")
    parser.add_argument("--output_file", type=str, default="../data/processed_abstracts.json", 
                        help="Path to save the processed abstracts")
    parser.add_argument("--min_length", type=int, default=100, 
                        help="Minimum length of abstract to keep (in characters)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Process the data
    abstracts = extract_abstracts(args.input_file, args.output_file, args.min_length)
    
    # Display some statistics
    abstract_lengths = [len(abstract) for abstract in abstracts]
    if abstract_lengths:
        print(f"Statistics:")
        print(f"  Total abstracts: {len(abstracts)}")
        print(f"  Average length: {sum(abstract_lengths) / len(abstracts):.1f} characters")
        print(f"  Min length: {min(abstract_lengths)} characters")
        print(f"  Max length: {max(abstract_lengths)} characters")
        
        # Show sample of first abstract (truncated)
        if abstracts:
            first_abstract = abstracts[0]
            print("\nSample abstract (truncated to 300 chars):")
            print(f"  {first_abstract[:300]}...")

if __name__ == "__main__":
    main() 