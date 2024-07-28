# chat.py

from RAG import retrieve_candidates, vectorizer, index, resume_df
from llm import get_match_explanation


def display_candidates(candidate_indices, resume_df, job_description):
    for idx in candidate_indices:
        candidate = resume_df.iloc[idx]
        print(f"Name: {candidate['name']}")
        print(f"Contact: {candidate['contact_details']}")
        print(f"Location: {candidate['location']}")
        print(f"Skills: {candidate['job_skills']}")
        print(f"Experience: {candidate['experience']}")
        print(f"Projects: {candidate['projects']}")
        print(f"Comments: {candidate['comments']}")
        print(f"Match Explanation: {get_match_explanation(job_description, candidate)}")
        print('-' * 80)


def main():
    print("Welcome to the Job Candidate Matching System!")
    print("Enter a job description, and I'll find the best matching candidates.")
    print("Type 'exit' to quit the program.")
    print("-" * 80)

    while True:
        print("\nEnter a job description:")
        job_description = input().strip()

        if job_description.lower() == 'exit':
            print("Thank you for using the Job Candidate Matching System. Goodbye!")
            break

        if not job_description:
            print("Please enter a valid job description.")
            continue

        print("\nSearching for matching candidates...")
        candidate_indices = retrieve_candidates(job_description, index, vectorizer, resume_df)

        if len(candidate_indices) == 0:
            print("No matching candidates found. Please try a different job description.")
        else:
            print(f"\nTop {len(candidate_indices)} matching candidates:\n")
            display_candidates(candidate_indices, resume_df, job_description)


if __name__ == "__main__":
    main()
