# ScamGuard – Fake Job Detection System

ScamGuard is a machine-learning–powered web application that detects **fake job postings**, identifies **scam-like behavior**, and provides **confidence-based explanations** using an ensemble classification model.

When the model is uncertain, the system marks a posting as **“Needs Review”** and collects additional information from the user. This data is forwarded to an **Admin Review Dashboard** where human reviewers make the final decision.

---

##  **Core Features**

### **1. Job Posting Scanner**

Paste any job description to receive:

*  *Real Job*
*  *Fake Job*
*  *Needs Review* (low-confidence zone)

The ML model is an ensemble of:

* Logistic Regression
* Random Forest
* XGBoost

The system also shows a prediction confidence score and model-based explanation.

---

### **2. Needs Review Workflow**

If the prediction is uncertain, the user is redirected to a short questionnaire asking:

* Where did you find this job posting?
* Were you asked to pay money, deposits, or fees?
* Did the company ask for sensitive information?
* Any other unusual or suspicious details?

These answers are saved and shown to admins for manual review.

---

### **3. Admin Dashboard**

Admins can:

* View all **pending reviews**
* Read **user-submitted answers**
* See job text previews and full text
* Mark jobs as **Real** or **Fake**
* Access operational analytics such as:

  * Total scans
  * Percentage of fake jobs detected
  * Number of reviews completed

---

### **4. Playground Mode (No Login Required)**

A public demo environment where users can:

* Try the scanner using sample job descriptions
* Extract structured fields using an NLP parser (title, salary, benefits, etc.)
* Preview the system without creating an account

---

### **5. Text Extraction Engine**

The system automatically extracts:

* Job title
* Employment type
* Remote/onsite status
* Required experience
* Salary mentions
* Benefits
* Presence of sensitive-information requests

---

### **6. User Accounts**

Authentication is powered by **Supabase**.
Logged-in users can:

* Access the full scanner
* View prediction history
* Save and revisit past analysis logs

---

##  **Project Architecture**

### **Backend**

* Python (Flask)
* Supabase Auth (users)
* SQLite database
* ML Model (Logistic Regression + Random Forest + XGBoost)

### **Frontend**

* Jinja2 Templates
* HTML / CSS / JavaScript
* Responsive layout with modern UI components

---

## **How to Run the Project Locally**

### **1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/ScamGuard.git
cd ScamGuard
```

---

### **2. Create a virtual environment**

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

---

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

### **4. Add environment variables**

Create a `.env` file in the project root:

```bash
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

---

### **5. Initialize the database**

SQLite will be created automatically, but ensure tables exist:

```bash
python
```

```python
from data.db import init_db
init_db()
exit()
```

Tables created:

* `prediction_logs`
* `review_answers`

---

### **6. Run the Flask application**

```bash
python app.py
```

Or using Flask CLI:

```bash
flask run
```

The application will start at:

```
http://127.0.0.1:5000/
```

---

##  **Admin Access**

To make a user an admin:

1. Go to Supabase → `login` table
2. Set:

```
permission_level = 1
```

Admin pages:

```
/admin_dashboard
/admin/review
```

---

## Live Deployment

The project is live at:
 **[http://cassini.cs.kent.edu:9620/](http://cassini.cs.kent.edu:9620/)**

---
