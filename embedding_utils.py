# embedding_utils.py

import openai
import os

# Load your OpenAI API key securely from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text, model="text-embedding-3-small"):
    """
    Gets the OpenAI embedding for a given text string using the specified embedding model.
    """
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']
