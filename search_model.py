import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load product data
df = pd.read_csv("data_csv/product_data_cleaned.csv")  # Adjust path if needed

# Combine name and description for embedding (adjust column names as needed)
if 'description' in df.columns:
    texts = (df['name'] + " " + df['description']).fillna("")
else:
    texts = df['name'].fillna("")

# Load a pre-trained sentence transformer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed(texts):
    # Tokenize and get embeddings
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    # Mean pooling
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Compute product embeddings
product_embeddings = embed(texts)

# Save embeddings for later use (optional)
np.save("product_embeddings.npy", product_embeddings)

# --- Search Function ---
def search(query, top_k=5):
    query_emb = embed([query])
    similarities = cosine_similarity(query_emb, product_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    results = df.iloc[top_indices][['name', 'description']] if 'description' in df.columns else df.iloc[top_indices][['name']]
    results['score'] = similarities[top_indices]
    return results

# --- Example Usage ---
if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    results = search(user_query)
    print("\nTop search results:")
    print(results) 