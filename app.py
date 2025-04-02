import os
from dotenv import load_dotenv
import chromadb
import ollama
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name
)

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# Function to generate embeddings using Ollama (without using embedding_functions.OllamaEmbeddingFunction)
def get_ollama_embedding(text):
    print("==== Generating embeddings... ====")
    response = ollama.embeddings(model="gemma:2b", prompt=text)   
    return response["embedding"]

# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_ollama_embedding(doc["text"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )

# Function to query documents
def query_documents(question, n_results=2):
    # Generate the embedding for the query using Ollama
    query_embedding = get_ollama_embedding(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

# Function to generate a response from Ollama
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = ollama.chat(model="gemma:2b", messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}])
    answer = response["message"]["content"]
    return answer

# Example query and response generation
question = "tell me about databricks?"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)