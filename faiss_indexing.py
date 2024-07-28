import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss

# Load the resume data
resume_df = pd.read_csv('dataset.csv')

# Combine relevant columns into a single text for embedding
def combine_features(row):
    return f"{row['name']} {row['job_skills']} {row['experience']} {row['projects']} {row['comments']}"

resume_df['combined_text'] = resume_df.apply(combine_features, axis=1)

# Use TF-IDF for feature extraction
vectorizer = TfidfVectorizer(max_features=100)  # Adjust max_features as needed
embeddings = vectorizer.fit_transform(resume_df['combined_text']).toarray()

# Save the embeddings and vectorizer for future use
np.save('embeddings.npy', embeddings)
from joblib import dump
dump(vectorizer, 'vectorizer.joblib')

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))  # FAISS requires float32

# Save the FAISS index
faiss.write_index(index, 'candidates.index')

print("Data preprocessing and indexing complete.")
