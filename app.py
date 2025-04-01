import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load environment variables from .env file
load_dotenv()

# Using Hugging Face's sentence-transformers model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(texts):
    if isinstance(texts, str):  # Ensure input is a list
        texts = [texts]
    return embedding_model.encode(texts).tolist()


chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
)

# Using Hugging Face LLM
chat_pipeline = pipeline(
    "text-genaration",
    model="mistralai/Mistral-7B-Instruct-v0.1", device="cpu"
)

def chat_with_model(prompt):
    response = chat_pipeline(prompt, max_length=200, do_sample=True)
    return response[0]['generated_text']

user_prompt = "What is human life expectancy in the US?"
response = chat_with_model(user_prompt)

print(response)
