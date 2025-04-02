# RAG-Based Data Retrieval with Ollama

This project demonstrates a **Retrieval-Augmented Generation (RAG)** system using the **Ollama** model for document-based question answering. The goal of the project is to integrate documents, retrieve relevant information, and generate concise answers using the Ollama LLM.

## Features

- Loads and processes news articles from a directory.
- Splits documents into manageable chunks.
- Uses **Chroma** for vector storage and retrieval.
- Generates embeddings for documents using the **Ollama** model.
- Queries the document database and generates responses using the **Ollama** LLM.

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

5. Set up your **Ollama** API key in a `.env` file:

    ```bash
    OLLAMA_API_KEY=your_ollama_api_key
    ```

## Usage

1. Prepare a folder `news_articles/` containing `.txt` files of articles.
2. Run the application:

    ```bash
    python app.py
    ```

3. Query the system by providing a question, and the system will fetch relevant documents and generate a response.

## Example Query

To query the system:

```python
question = "What is human life expectancy in the US and Bangladesh?"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)
print(answer)
