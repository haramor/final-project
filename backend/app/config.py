import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev')
    
    # PubMed settings
    PUBMED_EMAIL = os.getenv('PUBMED_EMAIL', 'sbentley@mit.edu')
    PUBMED_API_KEY = os.getenv('PUBMED_API_KEY')
    PUBMED_TOOL = os.getenv('PUBMED_TOOL', 'BirthControlInfoApp')
    
    # API settings
    MAX_RESULTS = int(os.getenv('MAX_RESULTS', 10))
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')