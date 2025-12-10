# ScamGuard

ScamGuard is a web application that detects fake job postings, identifies scam-like patterns, and provides confidence-based explanations using an ensemble ML model.
Users can analyze job descriptions, view predictions, and receive warnings when the system detects suspicious or unclear behavior.

For uncertain predictions, the system marks the posting as "Needs Review", asks the user extra verification questions, and forwards everything to an Admin Review Dashboard where human reviewers make the final judgment.

Core Features
1. Job Posting Scanner
Users can paste any job description.
The ensemble model (Logistic Regression + Random Forest + XGBoost) predicts:
Real Job
Fake Job
Needs Review (low confidence zone)
Confidence score & model breakdown are displayed.

2. Needs Review Workflow
If the model is unsure:
User is redirected to a Review Questions Page asking:
Where did you find this job?
Were you asked to pay any money?
Did they ask sensitive personal info?
Any other suspicious details?
Answers are saved and shown to admins.

3. Admin Dashboard
Admins can:
View all users
Review pending “Needs Review” predictions
View user-submitted answers
Mark postings as Real or Fake
See statistical summaries (total scans, accuracy, etc.)

4. Playground Mode (No Login Needed)
Allows new visitors to test the system using sample job postings.
Extracts structured information using an NLP text extractor.

5. Text Extraction
Automatically pulls key details from raw job postings:
Position title
Employment type
Remote or onsite
Experience level
Benefits snippet
Salary mention
Sensitive info request detection

6. User Accounts
Register/Login using Supabase authentication

Logged-in users receive:
Full scanner access
Prediction history
Saved analysis logs

Project Architecture
Backend:
Flask (Python)
Supabase (Auth + Users table)
SQLite
Ensemble ML Model (Scikit-learn + XGBoost)

Frontend :
Jinja2 Templates
HTML + CSS + JavaScript

ML:
Logistic Regression
Random Forest
XGBoost
Ensemble averaging with adjustable threshold & review band

How to Run the Project Locally
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ScamGuard.git
cd ScamGuard

2. Create a virtual environment
python -m venv venv

Activate it:
On Windows:
venv\Scripts\activate

On Mac/Linux:
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Add environment variables
Create a .env file in the project folder:

SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key

5. Initialize the database
SQLite database will auto-create on first run, but ensure the tables exist:

python
>>> from data.db import init_db
>>> init_db()
>>> exit()

This creates:
prediction_logs
review_answers

6. Run the Flask app
python app.py

If using Flask CLI:
flask run
App will be available at:
http://127.0.0.1:/


Admin Access:
To make a user an admin:
Go to Supabase → table login
Set Permission_level = 1
Admins gain access to:
/admin_dashboard
/admin/review

Project is live on http://cassini.cs.kent.edu:9620/