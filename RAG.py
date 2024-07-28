# rag.py

import numpy as np
import faiss
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from joblib import load

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def preprocess(text):
    return ' '.join([word.lower() for word in text.split() if word.lower() not in stop_words])


def calculate_similarity(query, candidate, vectorizer):
    query_prep = preprocess(query)
    candidate_prep = preprocess(f"{candidate['job_skills']} {candidate['experience']} {candidate['projects']}")

    tfidf_matrix = vectorizer.transform([query_prep, candidate_prep])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def retrieve_candidates(query, index, vectorizer, resume_df, k1=100, k2=5):
    # Stage 1: Broad retrieval using FAISS
    query_embedding = vectorizer.transform([query]).toarray().astype('float32')
    distances, indices = index.search(query_embedding, k1)

    # Stage 2: Re-rank using TF-IDF and cosine similarity
    candidates = resume_df.iloc[indices[0]]
    candidates['score'] = candidates.apply(lambda x: calculate_similarity(query, x, vectorizer), axis=1)
    top_candidates = candidates.nlargest(k2, 'score')

    return top_candidates.index


# Load the TF-IDF vectorizer
vectorizer = load('vectorizer.joblib')

# Load the FAISS index
index = faiss.read_index('candidates.index')

# Load resume data
resume_df = pd.read_csv('dataset.csv')

print("RAG system loaded and ready.")
