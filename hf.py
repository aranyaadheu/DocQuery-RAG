from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "What is the capital of Bangladesh & Bhutan?"
embedding = model.encode(text)

print(embedding[:5])
