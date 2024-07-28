from transformers import pipeline


class LLMExplainer:
    def __init__(self):
        # Initialize the summarization pipeline
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # device=-1 uses CPU

    def generate_explanation(self, query, candidate):
        # Combine job description and candidate information
        combined_text = f"Job Description: {query}\n\nCandidate Profile: {candidate['job_skills']} {candidate['experience']} {candidate['projects']}"

        # Generate summary using the LLM
        summary = self.summarizer(combined_text, max_length=45, min_length=30, do_sample=False)[0]['summary_text']

        return summary

    def explain_match(self, query, candidate):
        return self.generate_explanation(query, candidate)


# Create an instance of the LLMExplainer
explainer = LLMExplainer()


# Function to be imported and used in other modules
def get_match_explanation(query, candidate):
    return explainer.explain_match(query, candidate)
