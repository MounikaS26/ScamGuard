import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Load dataset
# Prefer engineered_features.csv if available
data_path = "C:/Users/Lenovo/OneDrive - Kent State University/Documents/ScamGuard/data"
models_path = "C:/Users/Lenovo/OneDrive - Kent State University/Documents/ScamGuard/models"

# Load data file
file_path = os.path.join(data_path, "fake_job_postings.csv")
 #   if os.path.exists(os.path.join(data_path, "engineered_features.csv")) \
 #   else os.path.join(data_path, "fake_job_postings.csv")

df = pd.read_csv(file_path)
print(f"Loaded: {file_path}")
print("Shape:", df.shape)

# Combine important text columns
text_columns = ["title", "company_profile", "description", "requirements", "benefits"]
df["combined_text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

print("\nSample combined text:")
print(df["combined_text"].head(2))

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=4000,     # limit vocabulary size
    ngram_range=(1,2)      # include unigrams + bigrams
)

# Fit and transform text data
X_tfidf = tfidf.fit_transform(df["combined_text"])
print("\nTF-IDF matrix shape:", X_tfidf.shape)

# Save vectorizer in models folder
tfidf_vectorizer_path = os.path.join(models_path, "tfidf_vectorizer.pkl")
with open(tfidf_vectorizer_path, "wb") as f:
    pickle.dump(tfidf, f)
print(f"TF-IDF vectorizer saved at: {tfidf_vectorizer_path}")

# Convert TF-IDF to DataFrame (optional for merging)
tfidf_df = pd.DataFrame(
    X_tfidf.toarray(),
    columns=tfidf.get_feature_names_out()
)

# Add label column back for training
if "fraudulent" in df.columns:
    tfidf_df["fraudulent"] = df["fraudulent"]

# Save TF-IDF features in data folder
tfidf_features_path = os.path.join(data_path, "tfidf_features.csv")
tfidf_df.to_csv(tfidf_features_path, index=False)
print(f"TF-IDF features saved at: {tfidf_features_path}")

# Preview feature names
print("\nTop 10 TF-IDF feature names:")
print(tfidf.get_feature_names_out()[:10])