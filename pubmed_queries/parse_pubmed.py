import xml.etree.ElementTree as ET

def parse_pubmed_xml(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Iterate through all PubmedArticle elements
    for article in root.findall("PubmedArticle"):
        # Extract PMID
        pmid_elem = article.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else "N/A"
        print("PMID:", pmid)

        # Extract Article Title
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else "N/A"
        print("Article Title:", title)

        # Extract Abstract texts (there may be multiple abstract sections)
        abstract = article.find(".//Abstract")
        if abstract is not None:
            for abstract_text in abstract.findall("AbstractText"):
                label = abstract_text.attrib.get("Label", "Abstract")
                text = abstract_text.text or ""
                print(f"{label}:", text)
        else:
            print("Abstract: N/A")

        # Extract Publication Date from ArticleDate (if available)
        pub_date = article.find(".//ArticleDate")
        if pub_date is not None:
            year = pub_date.find("Year").text if pub_date.find("Year") is not None else ""
            month = pub_date.find("Month").text if pub_date.find("Month") is not None else ""
            day = pub_date.find("Day").text if pub_date.find("Day") is not None else ""
            print("Publication Date:", f"{year}-{month}-{day}")
        else:
            print("Publication Date: N/A")

        # Extract Authors from the AuthorList
        authors = article.findall(".//AuthorList/Author")
        if authors:
            print("Authors:")
            for author in authors:
                # Get the first and last names if available
                fore_name = author.find("ForeName").text if author.find("ForeName") is not None else ""
                last_name = author.find("LastName").text if author.find("LastName") is not None else ""
                print(f"  {fore_name} {last_name}")
        else:
            print("Authors: N/A")
        
        print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    file_path = "output.xml"  # Replace with the path to your XML file
    parse_pubmed_xml(file_path)
