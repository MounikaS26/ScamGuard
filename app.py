from flask import Flask, render_template, request, redirect, url_for, flash, session
import pickle
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import LabelEncoder
import secrets

from auth import auth, login_required

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data")
MODELS_PATH = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")
ENGINEERED_PATH = os.path.join(DATA_PATH, "engineered_features.csv")

app.register_blueprint(auth, url_prefix="/auth")

with open(VECTORIZER_PATH, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

engineered_df = pd.read_csv(ENGINEERED_PATH)


engineered_cols = [c for c in engineered_df.columns if c != "fraudulent"]

cat_cols = [
    "title", "location", "department", "salary_range",
    "company_profile", "description", "requirements", "benefits",
    "employment_type", "required_experience", "required_education",
    "industry", "function"
]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    engineered_df[col] = engineered_df[col].astype(str).fillna("Unknown")
    engineered_df[col] = le.fit_transform(engineered_df[col])
    label_encoders[col] = le

defaults_series = engineered_df[engineered_cols].apply(pd.to_numeric, errors="coerce").median()

print(f"Using {len(engineered_cols)} engineered features for prediction.")
print(f"Model expects: {getattr(model, 'n_features_in_', 'N/A')}")
print(f"TF-IDF features: {len(tfidf_vectorizer.get_feature_names_out())}")

def parse_job_posting(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    full_text = "\n".join(lines)

    if lines:
        title = lines[0]
    else:
        title = " ".join(text.split()[:5])

    def extract_section(keyword: str, block: str):
        m = re.search(rf"{keyword}\s*[:\-]\s*(.*?)(?=\n[A-Z][A-Za-z ]+:\s*|\Z)",
                      block, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    company_profile = extract_section("Company", full_text)
    description = extract_section("Description", full_text)
    requirements = extract_section("Requirements", full_text)
    benefits = extract_section("Benefits", full_text)

    return title, company_profile, description, requirements, benefits

def combine_text_fields(title, company_profile, description, requirements, benefits):
    return " ".join([str(f).strip() for f in [title, company_profile, description, requirements, benefits] if str(f).strip()])

def make_engineered_vector(title, company_profile, description, requirements, benefits):
    """
    Build a single-row engineered feature vector aligned with engineered_cols.
    Unset engineered columns default to zero; categorical get label-encoded.
    """

    row = {c: 0 for c in engineered_cols}

    desc = str(description or "").strip()
    words = desc.split()

    row["char_count"] = len(desc)
    row["word_count"] = len(words)
    row["avg_word_length"] = (sum(len(w) for w in words) / len(words)) if words else 0.0
    row["num_exclamations"] = desc.count("!")
    row["num_dollar"] = desc.count("$")
    row["num_uppercase_words"] = sum(1 for w in words if w.isupper() and len(w) > 1)

    sentences = re.split(r"[.!?]", desc)
    sentences = [s.strip() for s in sentences if s.strip()]
    row["avg_sentence_length"] = (sum(len(s.split()) for s in sentences) / len(sentences)) if sentences else 0.0

    row["spelling_errors"] = 0.0  
    row["has_company_profile"] = int(bool(str(company_profile).strip()))
    row["has_requirements"] = int(bool(str(requirements).strip()))
    row["has_benefits"] = int(bool(str(benefits).strip()))

    df_row = pd.DataFrame([{
        **row,
        "title": str(title).strip() or "Unknown",
        "company_profile": str(company_profile).strip() or "Unknown",
        "description": desc or "Unknown",
        "requirements": str(requirements).strip() or "Unknown",
        "benefits": str(benefits).strip() or "Unknown"
    }])

    for col, le in label_encoders.items():
        if col in df_row.columns:
            df_row[col] = df_row[col].astype(str).replace("", "Unknown")
            df_row[col] = df_row[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)


    df_row = df_row[engineered_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df_row.values.flatten().astype(float)


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    prediction, confidence, input_text, error_message = None, None, None, None

    if request.method == "POST":
        raw_text = request.form.get("job_posting", "").strip()
        input_text = raw_text

        if not raw_text:
            error_message = "Please paste a job posting first."
            return render_template("index.html", error_message=error_message)

      
        title, company_profile, description, requirements, benefits = parse_job_posting(raw_text)

    
        engineered_vec = make_engineered_vector(title, company_profile, description, requirements, benefits)
        combined_text = combine_text_fields(title, company_profile, description, requirements, benefits)
        X_tfidf = tfidf_vectorizer.transform([combined_text]).toarray()[0]
        combined = np.concatenate([engineered_vec, X_tfidf]).reshape(1, -1)

        if hasattr(model, "n_features_in_") and combined.shape[1] != model.n_features_in_:
            error_message = f"Feature size mismatch: got {combined.shape[1]}, expected {model.n_features_in_}"
        else:
            
            pred = int(model.predict(combined)[0])

            p_fake = None
            try:
                proba_vec = model.predict_proba(combined)[0]
                if proba_vec.shape[0] == 2:
                    p_fake = float(proba_vec[1])
            except Exception:
                pass

            # Optional thresholding on fake probability (helps recall)
            THRESHOLD_FAKE = 0.40
            if p_fake is not None:
                pred = 1 if p_fake >= THRESHOLD_FAKE else 0
                confidence = round((p_fake if pred == 1 else 1 - p_fake) * 100, 2)
            else:
                confidence = None

            prediction = "⚠️ Fake Job Posting" if pred == 1 else "✅ Real Job Posting"

        
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        input_text=input_text,
        error_message=error_message
    )

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True)
