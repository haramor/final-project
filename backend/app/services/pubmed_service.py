from Bio import Entrez
from flask import current_app
import urllib.parse
from openai import OpenAI
import json

class PubMedService:
    def __init__(self):
        Entrez.email = current_app.config['PUBMED_EMAIL']
        Entrez.api_key = current_app.config['PUBMED_API_KEY']
        Entrez.tool = current_app.config['PUBMED_TOOL']
        self.max_results = current_app.config['MAX_RESULTS']
        self.api_key = current_app.config['OPENAI_API_KEY']
        # Common birth control methods for dropdown
        self.birth_control_methods = [
            "Combined Oral Contraceptives",
            "Progestin-Only Pills",
            "Intrauterine Devices",
            "IUD Copper",
            "IUD Hormonal",
            "Contraceptive Implants",
            "Nexplanon",
            "Contraceptive Patch",
            "Vaginal Ring",
            "Depo-Provera",
            "Condoms",
            "Diaphragm",
            "Cervical Cap",
            "Spermicides",
            "Emergency Contraception (Plan B, Ella, etc.)",
            "Vasectomy"
        ]
        
        # Side effects for dropdown
        self.side_effects = [
            "Weight Gain",
            "Mood Changes",
            "Depression",
            "Anxiety",
            "Headache",
            "Migraine",
            "Nausea",
            "Breast Tenderness",
            "Decreased Libido",
            "Menstrual Changes",
            "Amenorrhea",
            "Spotting",
            "Heavy Bleeding",
            "Blood Clots",
            "Thromboembolism",
            "Hypertension",
            "Acne",
            "Hair Loss"
        ]
        
        # Age groups for dropdown
        self.age_groups = [
            "Adolescent (13-18)",
            "Young Adult (19-24)",
            "Adult (25-44)",
            "Middle Aged (45-64)",
            "Aged (65+)"
        ]
        
        # Additional filters
        self.additional_filters = [
            "Efficacy",
            "Safety",
            "Cost-effectiveness",
            "Long-term Effects",
            "Breastfeeding",
            "Postpartum",
            "Drug Interactions",
            "Hormonal Effects",
            "Non-contraceptive Benefits",
            "Compliance",
            "Continuation Rates",
            "Patient Satisfaction"
        ]
        
        # MeSH terms for more precise searches
        self.mesh_terms = [
            "Contraception",
            "Contraceptive Agents",
            "Contraceptive Devices",
            "Contraceptives, Oral",
            "Contraceptives, Oral, Hormonal",
            "Intrauterine Devices",
            "Contraceptive Agents, Female",
            "Contraceptive Agents, Male",
            "Family Planning Services"
        ]



    def search_articles(self, filters=None, natural_language_query=None):
        try:
            # Build query using only filters
            pubmed_query = self._build_filter_query(filters)
            print(f"Final PubMed query: {pubmed_query}")
            
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed", 
                term=pubmed_query,
                retmax=self.max_results,
                sort='relevance'
            )
            record = Entrez.read(handle)
            handle.close()

            # Get article details
            articles = []
            if record["IdList"]:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=record["IdList"],
                    rettype="medline",
                    retmode="xml"
                )
                papers = Entrez.read(handle)
                handle.close()

                # First collect all articles and their abstracts
                for paper in papers['PubmedArticle']:
                    article_data = self._format_article(paper)
                    articles.append(article_data)

                # Build the preamble for LLM from filters
                preamble = self._build_llm_preamble(filters)

                # Combine all abstracts for a single LLM query
                all_abstracts = "\n\n".join([
                    f"Article: {article['title']}\nAbstract: {article['abstract']}\nLink: {article['url']}"
                    for article in articles
                ])

                # Get single LLM response for all abstracts
                llm_response = self._get_llm_response(
                    preamble=preamble,
                    question=natural_language_query,
                    article_abstracts=all_abstracts
                )
            print("llm_response", llm_response)
            return llm_response, None

        except Exception as e:
            return None, str(e)

    def _build_filter_query(self, filters):
        """Build PubMed query using only filters"""
        terms = []
        
        if filters:
            if filters.get('birth_control'):
                bc_terms = [f'"{bc}"' for bc in filters['birth_control']]
                terms.append(f"({' OR '.join(bc_terms)})")
            
            if filters.get('side_effects'):
                se_terms = [f'"{se}"' for se in filters['side_effects']]
                terms.append(f"({' OR '.join(se_terms)})")
            
            # Add other filters as needed
        
        # Always include base contraception term
        base_query = "Contraception[MeSH Terms]"
        if terms:
            return f"{base_query} AND {' '.join(terms)}"
        return base_query

    def _build_llm_preamble(self, filters):
        """Build a natural language preamble from filters"""
        parts = []
        
        if filters:
            if filters.get('age_group'):
                parts.append(f"I am {filters['age_group'][0]}")
            
            if filters.get('birth_control'):
                bc_list = ', '.join(filters['birth_control'])
                parts.append(f"taking {bc_list}")
            
            if filters.get('side_effects'):
                se_list = ', '.join(filters['side_effects'])
                parts.append(f"experiencing these side effects: {se_list}")

        if parts:
            return "Context: " + "; ".join(parts) + "."
        return ""

    def _get_llm_response(self, preamble, question, article_abstracts):
        """Get response from GPT"""
        prompt = f"""
        {preamble}

        Question: {question}

        Based on these research articles:
        {article_abstracts}

        Please provide a clear, concise answer that:
        1. Relates the research findings to my specific situation and question
        2. Synthesizes information from all provided articles
        3. Focuses on practical implications and relevant findings
        4. Highlights any consensus or conflicts in the research
        5. Cites the research articles in your response using the provided links
        
        Format your response in clear paragraphs. If there are any medical disclaimers needed, include them at the end.
        """
        print("prompt", prompt)
        try:
            self._client = OpenAI(api_key=self.api_key)
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-3.5-turbo" for a more economical option
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing medical research articles about contraception. Provide clear, evidence-based responses while maintaining appropriate medical disclaimers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused, consistent responses
                max_tokens=1000   # Adjust based on your needs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling GPT: {str(e)}")
            return "Sorry, I encountered an error while analyzing the research articles. Please try again."

    def get_dropdown_options(self):
        """
        Get all dropdown options for the frontend
        
        Returns:
            dict: Dictionary containing all dropdown options
        """
        return {
            'birth_control_methods': self.birth_control_methods,
            'side_effects': self.side_effects,
            'age_groups': self.age_groups,
            'additional_filters': self.additional_filters,
            'mesh_terms': self.mesh_terms
        }

    def _format_article(self, paper):
        article_data = paper['MedlineCitation']['Article']
        paper_id = paper['MedlineCitation']['PMID']
        
        # Get abstract text, handling potential missing abstract
        abstract = ""
        if 'Abstract' in article_data and 'AbstractText' in article_data['Abstract']:
            abstract_parts = article_data['Abstract']['AbstractText']
            if isinstance(abstract_parts, list):
                # Join multiple abstract sections if present
                for part in abstract_parts:
                    if isinstance(part, str):
                        abstract += part + " "
                    elif hasattr(part, 'attributes') and 'Label' in part.attributes:
                        # Handle structured abstract with labeled sections
                        abstract += f"{part.attributes['Label']}: {part} "
                    else:
                        abstract += str(part) + " "
            else:
                abstract = abstract_parts
        
        return {
            'id': paper_id,
            'title': article_data.get('ArticleTitle', 'No title available'),
            'url': f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
            'abstract': abstract.strip(),
            'authors': [
                f"{author.get('LastName', '')} {author.get('ForeName', '')}"
                for author in article_data.get('AuthorList', [])
            ] if 'AuthorList' in article_data else [],
            'journal': article_data.get('Journal', {}).get('Title', 'Unknown Journal'),
            'publication_date': self._get_publication_date(article_data)
        }

    def _get_publication_date(self, article_data):
        pub_date = article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
        year = pub_date.get('Year', '')
        month = pub_date.get('Month', '')
        return f"{month} {year}".strip()
