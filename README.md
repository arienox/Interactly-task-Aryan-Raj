# Interactly-task-Aryan-Raj
# Job Candidate Matching System
## Overview
This project implements an intelligent system that matches job descriptions with candidate profiles using information retrieval techniques and Large Language Models (LLMs). It provides a fast and accurate way to find suitable candidates for given job descriptions.

## Features
- Efficient candidate retrieval using FAISS indexing
- Two-stage ranking process for improved accuracy
- LLM-generated explanations for each match
- User-friendly command-line interface

## Requirements
- Python 3.7+
- pandas
- scikit-learn
- faiss-cpu
- nltk
- joblib
- transformers


   Usage

Prepare your dataset:

Ensure you have a CSV file named dataset.csv with columns: name, contact_details, location, job_skills, experience, projects, comments


Create the FAISS index:
Copypython faiss.py

Run the chat interface:
Copypython chat.py

Enter job descriptions when prompted. The system will display matching candidates with explanations.

File Structure

database.py: Handles database operations
faiss.py: Creates embeddings and FAISS index
rag.py: Implements the Retrieval-Augmented Generation system
llm.py: Manages the LLM for generating explanations
chat.py: Provides the user interface

How It Works

TF-IDF creates embeddings for candidate profiles
FAISS index enables fast initial retrieval of potential matches
Second-stage ranking refines results using cosine similarity
BART LLM generates explanations for candidate matches
