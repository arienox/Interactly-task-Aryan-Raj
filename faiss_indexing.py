import pandas as pd
from transformers import TFAutoModel, AutoTokenizer
import numpy as np
import faiss

# Load the resume data
resume_df = pd.read_csv('dataset.csv')

# Combine relevant columns into a single text for embedding
def combine_features(row):
    return f"{row['name']} {row['job_skills']} {row['experience']} {row['projects']} {row['comments']}"

resume_df['combined_text'] = resume_df.apply(combine_features, axis=1)

# Load a pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = TFAutoModel.from_pretrained('bert-base-uncased')

# Tokenize the text data
inputs = tokenizer(resume_df['combined_text'].tolist(), return_tensors='tf', padding=True, truncation=True)

# Generate embeddings using TensorFlow model
outputs = model(**inputs)
embeddings = outputs.last_hidden_state[:, 0, :].numpy()

# Save the embeddings for future use
np.save('embeddings.npy', embeddings)

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, 'candidates.index')

print("Data preprocessing and indexing complete.")
