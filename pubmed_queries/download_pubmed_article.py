#!/usr/bin/env python3
import sys
import argparse
import requests

def download_pubmed_article(pmid, email, output_file):
    # URL for PubMed E-utilities efetch endpoint
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    # Set up parameters; include a tool name and email as recommended by NCBI
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "tool": "download_pubmed_article.py",
        "email": email,
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses
    except requests.RequestException as e:
        print(f"Error fetching article with PMID {pmid}: {e}")
        sys.exit(1)
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Article {pmid} downloaded successfully and saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a PubMed article by PMID without using Biopython")
    parser.add_argument("pmid", help="PubMed ID of the article")
    parser.add_argument("email", help="Your email address (required by NCBI)")
    parser.add_argument("-o", "--output", help="Output file name (default: article.xml)", default="article.xml")
    args = parser.parse_args()
    
    download_pubmed_article(args.pmid, args.email, args.output)
