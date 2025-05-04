# 3. Create a Question-Answering LLM to Answer User Questions
# Tasks:
# Use retrieved articles to generate a comprehensive answer
# Implement a question-answering model using an LLM

from openai import OpenAI

class QuestionAnsweringLLM:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def answer_question(self, question, articles):
        context = "\n\n".join([f"Title: {article[1]}\nAbstract: {article[2]}" for article in articles])
        prompt = f"""
        Context: {context}

        Question: {question}

        Please provide a clear, concise answer based on the above articles.
        """
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content

# Example usage
# qa_llm = QuestionAnsweringLLM(api_key='your-openai-api-key')
# answer = qa_llm.answer_question("What are the side effects of contraceptives?", relevant_articles)
