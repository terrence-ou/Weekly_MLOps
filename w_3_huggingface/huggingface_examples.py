from transformers import pipeline

# Zero shot classification
classifier = pipeline("zero-shot-classification")
classifier("This is a course about the Transformers library", 
            candidate_labels=["introduction", "summary", "question"])