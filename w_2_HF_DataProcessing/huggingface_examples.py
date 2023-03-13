from transformers import pipeline
import warnings

# Zero shot classification
print("\n======== Zero-shot Classification ========")
classifier = pipeline("zero-shot-classification")
result = classifier("This is a course about the Transformers library", 
            candidate_labels=["introduction", "debate", "question"])

print("\nZero-shot classification result: \n", result)


# Text generation
print("\n======== Text Generation ========")
generator = pipeline("text-generation")
results = generator("It is a difficult year that big companies like Amazon",
                    num_return_sequences=2, max_length=30)
print("\nText generation result: \n")
for i, ans in enumerate(results):
    print("\ngenerated sequence {}\n".format(i), ans)


# Using any model from the Hub in a pipeline
print("\n======== Text Generation with Model ========")
generator = pipeline("text-generation", model="distilgpt2")
result = generator("It is a difficult year that big companies like Amazon",
                    max_length=30, num_return_sequences=2, eos_token_id=50256)

print("\nText generation result: \n")
for i, ans in enumerate(results):
    print("\ngenerated sequence {}\n".format(i), ans)


# Using any model from the Hub in a pipeline
print("\n======== Translation ========")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result = translator("Ce cours est produit par Hugging Face.")
print("Translation result: \n", result)