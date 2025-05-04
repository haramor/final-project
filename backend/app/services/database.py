# 1. Store Relevant PubMed Articles in a Vectorized Database
# Tasks:
# - Extract and preprocess article data
# - Vectorize article abstracts
# - Store vectors in a database


# Import necessary libraries
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3

class ArticleVectorizer:
    def __init__(self, db_path='articles.db'):
        """
        Initialize the vectorizer with a pre-trained model and database connection.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        """
        Create a table in the database to store article information and vectors.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                vector BLOB
            )
        ''')
        self.conn.commit()

    def vectorize_and_store(self, articles):
        """
        Vectorize article abstracts and store them in the database.
        
        Parameters:
        - articles: List of dictionaries containing article data (id, title, abstract).
        """
        for article in articles:
            vector = self.model.encode(article['abstract'])
            vector_blob = vector.tobytes()
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO articles (id, title, abstract, vector)
                VALUES (?, ?, ?, ?)
            ''', (article['id'], article['title'], article['abstract'], vector_blob))
        self.conn.commit()

    def close_connection(self):
        """
        Close the database connection.
        """
        self.conn.close()

# Example usage
# vectorizer = ArticleVectorizer()
# vectorizer.vectorize_and_store(articles)
# vectorizer.close_connection()