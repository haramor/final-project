#!/usr/bin/env python3
import requests
import xml.etree.ElementTree as ET

def search_pubmed(query, email, retmax=10):
    """
    Search PubMed for a query term and return the XML response.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,      # Maximum number of results to retrieve
        "retmode": "xml",
        "tool": "pubmed_search_script",
        "email": email
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raise an error for bad responses
    return response.text

def fetch_articles(pmid_list, email):
    """
    Fetch detailed article metadata for a list of PMIDs from PubMed.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    id_str = ",".join(pmid_list)
    params = {
        "db": "pubmed",
        "id": id_str,
        "retmode": "xml",
        "tool": "pubmed_search_script",
        "email": email,
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.text

def parse_esearch(xml_str):
    """
    Parse the ESearch XML response to extract a list of PMIDs.
    """
    root = ET.fromstring(xml_str)
    # Find all <Id> elements under the <IdList> node
    pmid_list = [id_elem.text for id_elem in root.findall(".//Id")]
    return pmid_list

def parse_pubmed_articles(xml_str):
    """
    Parse the EFetch XML response and extract details for each article.
    """
    root = ET.fromstring(xml_str)
    articles = []
    # Loop over each PubmedArticle element
    for article in root.findall("PubmedArticle"):
        article_data = {}
        # Extract PMID
        pmid_elem = article.find(".//PMID")
        article_data["PMID"] = pmid_elem.text if pmid_elem is not None else "N/A"

        # Extract Article Title
        title_elem = article.find(".//ArticleTitle")
        article_data["Title"] = title_elem.text if title_elem is not None else "N/A"

        # Extract Abstract text (may have multiple sections)
        abstract = article.find(".//Abstract")
        if abstract is not None:
            abstract_texts = [at.text for at in abstract.findall("AbstractText") if at.text]
            article_data["Abstract"] = " ".join(abstract_texts)
        else:
            article_data["Abstract"] = "N/A"

        articles.append(article_data)
    return articles

if __name__ == "__main__":
    # Update your email address (NCBI requires this)
    email = "ljanjig@gmail.com"
    # Define your search query
    query = "birth control"
    # Maximum number of articles you want to fetch
    retmax = 5

    # Step 1: Search PubMed and get a list of PMIDs
    esearch_xml = search_pubmed(query, email, retmax=retmax)
    pmid_list = parse_esearch(esearch_xml)
    print("Found PMIDs:", pmid_list)

    # Step 2: Fetch articles for the retrieved PMIDs
    efetch_xml = fetch_articles(pmid_list, email)
    articles = parse_pubmed_articles(efetch_xml)

    # Step 3: Print details from each article
    for article in articles:
        print("\n" + "-"*40)
        print("PMID:", article["PMID"])
        print("Title:", article["Title"])
        print("Abstract:", article["Abstract"])
