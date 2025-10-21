from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os, re, pickle, pandas as pd, numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from functools import wraps

# ------------------ Import Blueprints ------------------
from auth import auth as auth_blueprint
import auth as auth_module

# ------------------ Load .env ------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ------------------ Initialize Supabase ------------------
if SUPABASE_URL and SUPABASE_KEY:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Supabase client initialized successfully")
else:
    supabase = None
    print("✗ Supabase URL or KEY not found in .env")

# Inject Supabase client into auth blueprint
auth_module.supabase = supabase

# ------------------ Initialize Flask ------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key_here")

# ------------------ Register Blueprints ------------------
app.register_blueprint(auth_blueprint, url_prefix="/auth")

# Import and register account blueprint
from account import account as account_blueprint
app.register_blueprint(account_blueprint, url_prefix="/account")

# ------------------ Decorators ------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in first!", "error")
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session or int(session.get("permission_level", 0)) < 1:
            flash("Access denied: admin privileges required.", "error")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return wrapper

# ------------------ Home & Redirect ------------------
@app.route("/")
def home():
    if 'user_id' in session:
        return redirect(url_for("index"))
    return redirect(url_for("auth.login"))

@app.route("/dashboard")
@login_required
def dashboard():
    level = int(session.get("permission_level", 0))
    if level >= 1:
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("user_dashboard"))

@app.route("/user-dashboard")
@login_required
def user_dashboard():
    username = session.get("username")
    return render_template("user_dashboard.html", username=username)

@app.route("/admin-dashboard")
@admin_required
def admin_dashboard():
    try:
        if supabase:
            response = supabase.table("login").select("id, Name, Email, Permission_level").execute()
            users_list = response.data
        else:
            users_list = []
        return render_template("admin_dashboard.html", users=users_list)
    except Exception as e:
        flash(f"Error loading admin dashboard: {e}", "error")
        return redirect(url_for("dashboard"))

# ------------------ Load Model & Data ------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data")
MODELS_PATH = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")
ENGINEERED_PATH = os.path.join(DATA_PATH, "engineered_features.csv")

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

# ------------------ Helper Functions ------------------
def parse_job_posting(text: str):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    full_text = "\n".join(lines)
    title = lines[0] if lines else " ".join(text.split()[:5])
    def extract_section(keyword: str, block: str):
        import re
        m = re.search(rf"{keyword}\s*[:\-]\s*(.*?)(?=\n[A-Z][A-Za-z ]+:\s*|\Z)", block, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""
    company_profile = extract_section("Company", full_text)
    description = extract_section("Description", full_text)
    requirements = extract_section("Requirements", full_text)
    benefits = extract_section("Benefits", full_text)
    return title, company_profile, description, requirements, benefits

def make_engineered_vector(title, company_profile, description, requirements, benefits):
    row = {c: 0 for c in engineered_cols}
    desc = str(description or "").strip()
    words = desc.split()
    row["char_count"] = len(desc)
    row["word_count"] = len(words)
    row["avg_word_length"] = (sum(len(w) for w in words)/len(words)) if words else 0
    row["num_exclamations"] = desc.count("!")
    row["num_dollar"] = desc.count("$")
    row["num_uppercase_words"] = sum(1 for w in words if w.isupper() and len(w)>1)
    sentences = [s.strip() for s in re.split(r"[.!?]", desc) if s.strip()]
    row["avg_sentence_length"] = (sum(len(s.split()) for s in sentences)/len(sentences)) if sentences else 0
    row["spelling_errors"] = 0.0
    row["has_company_profile"] = int(bool(company_profile))
    row["has_requirements"] = int(bool(requirements))
    row["has_benefits"] = int(bool(benefits))

    df_row = pd.DataFrame([{**row,
                            "title": title or "Unknown",
                            "company_profile": company_profile or "Unknown",
                            "description": desc or "Unknown",
                            "requirements": requirements or "Unknown",
                            "benefits": benefits or "Unknown"}])
    for col, le in label_encoders.items():
        if col in df_row.columns:
            df_row[col] = df_row[col].astype(str).replace("", "Unknown")
            df_row[col] = df_row[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    df_row = df_row[engineered_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df_row.values.flatten().astype(float)

# ------------------ Prediction Page ------------------
@app.route("/index", methods=["GET", "POST"])
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
        combined_text = " ".join([title, company_profile, description, requirements, benefits])
        X_tfidf = tfidf_vectorizer.transform([combined_text]).toarray()[0]
        combined = np.concatenate([engineered_vec, X_tfidf]).reshape(1, -1)
        pred = int(model.predict(combined)[0])
        try:
            p_fake = float(model.predict_proba(combined)[0][1])
            THRESHOLD_FAKE = 0.40
            pred = 1 if p_fake >= THRESHOLD_FAKE else 0
            confidence = round((p_fake if pred else 1 - p_fake)*100,2)
        except:
            confidence = None
        prediction = "⚠️ Fake Job Posting" if pred else "✅ Real Job Posting"

    return render_template("index.html", prediction=prediction, confidence=confidence,
                           input_text=input_text, error_message=error_message)

# ------------------ Health Check ------------------
@app.route("/health")
def health():
    return {"status": "ok"}

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(debug=True)
