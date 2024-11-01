import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
from groq import Groq

# Load CSV data
file_path = '/content/movie_dataset.csv.zip'  # Update with your dataset path
df = pd.read_csv(file_path)

# Combine columns for better context in embeddings
df['text'] = df['title'] + " " + df['overview'].fillna("")  # Customize columns as needed

# Load the embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute embeddings
embeddings = embedding_model.encode(df['text'].tolist())
embeddings = np.array(embeddings).astype('float32')

# Create a FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Save index for later use
faiss.write_index(index, 'movie_index.faiss')

# Initialize Groq API with your API key
client = Groq(api_key="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")

# Function to query FAISS and generate answer
def query_rag_system(question):
    # Embed the user question
    question_embedding = embedding_model.encode([question])[0].astype('float32')
    
    # Retrieve top 5 relevant documents
    D, I = index.search(np.array([question_embedding]), k=5)
    context = " ".join(df.iloc[idx]['text'] for idx in I[0])
    
    # Use Groq to generate the answer
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Question: {question}\nContext: {context}"}
        ],
        model="llama3-8b-8192",
    )
    
    return chat_completion.choices[0].message.content

# Streamlit frontend
def main():
    st.title("Movie Knowledge Chatbot")
    question = st.text_input("Ask a question about a movie:")
    
    if st.button("Get Answer"):
        answer = query_rag_system(question)
        st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()
