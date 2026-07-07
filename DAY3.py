import pandas as pd
import time
from transformers import pipeline



df = pd.read_csv("IMDB Dataset.csv")
df = df.head(5)

print("IMDb DATASET LOADED")
print("="*70)


# ALBERT 


print("\nALBERT ")

classifier = pipeline(
    "sentiment-analysis",
    model="textattack/albert-base-v2-imdb",
    device=-1
)

for i, row in df.iterrows():

    review = row["review"][:512]
    prediction = classifier(review)

    print("Actual :", row["sentiment"])
    print("Prediction :", prediction)


# GPT-Neo (Decoder Only)

print("\nGPT-Neo")

generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M",
    device=-1
)

prompt = "India is my country"

generated = generator(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.1)


print("\nPrompt:\n")
print(prompt)

print("\nGenerated Text:\n")
print(generated[0]["generated_text"])



# BART (Encoder Decoder)


print("\n BART")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

article = df.iloc[1]["review"][:1000]

summary = summarizer(
    article,
    max_length=80,
    min_length=25,
    do_sample=False
)

print("\nOriginal Review:\n")
print(article)

print("\nSummary:\n")
print(summary[0]["summary_text"])

print("\n")
print("="*70)
print("ALL THREE TRANSFORMER MODELS EXECUTED SUCCESSFULLY")
print("="*70)
