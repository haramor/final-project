# 2. Create a Retriever LLM to Retrieve Relevant Articles
# Tasks:
# - Implement a search mechanism to retrieve articles based on query vectors
# - Use cosine similarity or another metric to find relevant articles

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3

class ArticleRetriever:
    def __init__(self, db_path='articles.db'):
        """
        Initialize the retriever with a pre-trained model and database connection.
        """
        self.conn = sqlite3.connect(db_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, query, top_k=5):
        """
        Retrieve the most relevant articles based on the query.
        
        Parameters:
        - query: The search query as a string.
        - top_k: Number of top articles to return.
        
        Returns:
        - List of tuples containing article data (id, title, abstract).
        """
        query_vector = self.model.encode(query)
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, title, abstract, vector FROM articles')
        articles = cursor.fetchall()

        similarities = []
        for article in articles:
            vector = np.frombuffer(article[3], dtype=np.float32)
            similarity = cosine_similarity([query_vector], [vector])[0][0]
            similarities.append((article, similarity))

        # Sort articles by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [article for article, _ in similarities[:top_k]]

    def close_connection(self):
        """
        Close the database connection.
        """
        self.conn.close()

# Example usage
# retriever = ArticleRetriever()
# relevant_articles = retriever.retrieve("query text")
# retriever.close_connection()
