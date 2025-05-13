import json
import csv
import sys
import os

# Add the RAG service directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import the query_rag function from rag_service.py
from app.rag.rag_service import query_rag

# File paths
INPUT_JSON_PATH = "/Users/jankahamori/Documents/final-project/backend/app/evals_backend/birth_control_factual_qa.json"
OUTPUT_CSV_PATH = "questions_and_answers.csv"

def main():
    # Load questions and expected answers from the JSON file
    with open(INPUT_JSON_PATH, "r") as json_file:
        data = json.load(json_file)

    # Prepare the CSV file for writing
    with open(OUTPUT_CSV_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Write the header row
        writer.writerow(["Question", "Expected Answer", "Actual Answer"])

        # Iterate through each question in the JSON file
        for item in data:
            question = item["question"]
            expected_answer = item["answer"]

            # Use the query_rag function directly to get the actual answer
            actual_answer = query_rag(question)

            # Write the question, expected answer, and actual answer to the CSV file
            writer.writerow([question, expected_answer, actual_answer])

    print(f"Results saved to {OUTPUT_CSV_PATH}")    
    print(f"Results saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()