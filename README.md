# RAG-Based Data Retrieval with Ollama

This project demonstrates a **Retrieval-Augmented Generation (RAG)** system using the **Ollama** model for document-based question answering. The goal of the project is to integrate documents, retrieve relevant information, and generate concise answers using the **Ollama Gemma 2B** model.

## Features

- Loads and processes news articles from a specified directory (`news_articles`).
- Splits documents into manageable chunks for efficient retrieval.
- Uses **ChromaDB** for vector storage and retrieval of document embeddings.
- Generates embeddings for documents using the **Ollama Gemma 2B** model.
- Queries the document database and generates answers using **Ollama's** conversational model.

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/RAG-Based-Data-Retrieval.git
    cd RAG-Based-Data-Retrieval
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

5. Set up environment variables by creating a `.env` file in the root directory:

    ```bash
    CHROMA_PERSISTENT_STORAGE_PATH="chroma_persistent_storage"
    ```

6. Make sure you have access to **Ollama** for the embeddings and model queries.

## Usage

1. Prepare a folder `news_articles/` containing `.txt` files of articles you wish to process.
2. Run the application:

    ```bash
    python app.py
    ```

3. Query the system by providing a question. The system will fetch relevant document chunks, generate embeddings, and generate a response based on the retrieved context.

## Code Explanation

1. **Loading Documents**: The `load_documents_from_directory` function loads all `.txt` files from a specified directory and stores their content as documents.
2. **Text Chunking**: The `split_text` function splits long documents into smaller, manageable chunks to allow for more accurate retrieval.
3. **Embedding Generation**: The `get_ollama_embedding` function generates embeddings for document chunks using the **Gemma 2B** model from **Ollama**.
4. **Storage in ChromaDB**: The embeddings are stored in **ChromaDB** for fast retrieval during queries.
5. **Query Processing**: The `query_documents` function generates an embedding for the user’s query and retrieves the most relevant document chunks. The system uses the retrieved chunks to generate an answer.
6. **Answer Generation**: The `generate_response` function combines the relevant context and generates concise answers using **Ollama's conversational model**.

## Example Query

You can use this example to query the system:

```python
question = "What is human life expectancy in the US and Bangladesh?"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)
print(answer)
