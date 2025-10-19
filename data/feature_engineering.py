import pandas as pd
import re
from textblob import TextBlob
from spellchecker import SpellChecker

# Initialize spell checker
spell = SpellChecker()

# Load dataset
df = pd.read_csv("fake_job_postings.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Define helper functions
def avg_sentence_length(text):
    """Average number of words per sentence."""
    sentences = re.split(r'[.!?]', str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    return sum(len(s.split()) for s in sentences)/len(sentences) if sentences else 0

def count_uppercase_words(text):
    """Count of fully uppercase words."""
    return sum(1 for w in str(text).split() if w.isupper() and len(w) > 1)

def fast_spelling_errors(text):
    """Count of misspelled words (faster than TextBlob)."""
    words = str(text).split()
    return sum(1 for w in words[:100] if w.lower() not in spell)

# Create linguistic features
print("\nCreating linguistic features...")

df["char_count"] = df["description"].astype(str).apply(len)
df["word_count"] = df["description"].astype(str).apply(lambda x: len(x.split()))
df["avg_word_length"] = df["description"].astype(str).apply(
    lambda x: (sum(len(w) for w in x.split())/len(x.split())) if len(x.split()) > 0 else 0
)
df["num_exclamations"] = df["description"].astype(str).apply(lambda x: x.count("!"))
df["num_dollar"] = df["description"].astype(str).apply(lambda x: x.count("$"))
df["num_uppercase_words"] = df["description"].astype(str).apply(count_uppercase_words)
df["avg_sentence_length"] = df["description"].astype(str).apply(avg_sentence_length)
df["spelling_errors"] = df["description"].astype(str).apply(fast_spelling_errors)

print("Linguistic features created!")

# Add metadata presence features
print("\nAdding metadata presence features...")

df["has_company_profile"] = df["company_profile"].notnull().astype(int)
df["has_requirements"] = df["requirements"].notnull().astype(int)
df["has_benefits"] = df["benefits"].notnull().astype(int)

print("Metadata features added!")

# Save dataset
df.to_csv("engineered_features.csv", index=False)
print("\n" \
"Feature engineering complete! File saved as 'engineered_features.csv'")

# Verification
new_features = [
    "char_count", "word_count", "avg_word_length", 
    "num_exclamations", "num_dollar", "num_uppercase_words",
    "avg_sentence_length", "spelling_errors",
    "has_company_profile", "has_requirements", "has_benefits"
]

print("\nSample of engineered features:")
print(df[new_features].head())

print("\nFeature summary statistics:")
print(df[new_features].describe())