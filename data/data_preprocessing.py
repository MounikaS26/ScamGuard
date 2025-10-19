import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Loading dataset
df = pd.read_csv("fake_job_postings.csv")

print("Shape:", df.shape)
print(df.head(3))
print(df.info())

text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
cat_cols = ["employment_type", "required_experience", "required_education", "industry", "function"]

# Replacing whitespace only text entries with empty strings
df[text_cols] = df[text_cols].replace(r'^\s*$', '', regex=True)

# Add missing ones like department, salary_range, location
extra_cols = ["department", "salary_range", "location"]
text_cols = text_cols + extra_cols

# Filling missing values
df[text_cols] = df[text_cols].fillna("")
df[cat_cols] = df[cat_cols].fillna("Unknown")
print("Missing after fill:", df.isnull().sum().sum())

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

binary_cols = ["telecommuting", "has_company_logo", "has_questions"]

for col in binary_cols:
    df[col] = df[col].astype(int)

X = df.drop("fraudulent", axis=1)
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Preprocessing completed")

## Verification

print("Total missing values:", df.isnull().sum().sum())

print("\nSample of encoded categorical columns:")
print(df[cat_cols].head())

print("\nBinary column data types:")
print(df[binary_cols].dtypes)

print("\nFraud ratio in train:", y_train.mean())
print("Fraud ratio in test:", y_test.mean())

print("\nSaved files:", [f for f in os.listdir() if f.endswith('.csv')])