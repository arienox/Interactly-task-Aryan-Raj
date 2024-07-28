import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import psycopg2

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('fine_tuned_tokenizer')
model = TFAutoModel.from_pretrained('fine_tuned_model')

# Load FAISS index and embeddings
index = faiss.read_index('candidates.index')
embeddings = np.load('embeddings.npy')

# Load resume data
resume_df = pd.read_csv('dataset.csv')
print(resume_df.columns)

def retrieve_candidates(query, index, model, tokenizer, resume_df):
    query_embedding = model(tokenizer(query, return_tensors='tf', truncation=True, padding=True, max_length=128)).last_hidden_state[:, 0, :].numpy()
    distances, indices = index.search(query_embedding, 5)  # Retrieve top 5 closest candidates

    # Debugging: print distances and corresponding candidate details
    print("Query:", query)
    print("Distances:", distances)
    for idx in indices[0]:
        print("Candidate:", resume_df.iloc[idx])
        print("Match Distance:", distances[0][np.where(indices[0] == idx)[0][0]])
        print("-----------")

    return indices[0]

def display_candidates(candidate_indices, resume_df):
    for idx in candidate_indices:
        candidate = resume_df.iloc[idx]
        print(f"Name: {candidate['name']}")
        print(f"Contact: {candidate['contact_details']}")
        print(f"Location: {candidate['location']}")
        print(f"Skills: {candidate['job_skills']}")
        print(f"Experience: {candidate['experience']}")
        print(f"Projects: {candidate['projects']}")
        print(f"Comments: {candidate['comments']}")
        print('-' * 80)

if __name__ == "__main__":
    while True:
        print("Enter a job description (or type 'exit' to quit):")
        job_description = input().strip()
        if job_description.lower() == 'exit':
            break
        candidate_indices = retrieve_candidates(job_description, index, model, tokenizer, resume_df)
        display_candidates(candidate_indices, resume_df)

