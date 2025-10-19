import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

data_path = "C:/Users/Lenovo/OneDrive - Kent State University/Documents/ScamGuard/data"
models_path = "C:/Users/Lenovo/OneDrive - Kent State University/Documents/ScamGuard/models"

os.makedirs(data_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

engineered_path = os.path.join(data_path, "engineered_features.csv")
tfidf_path = os.path.join(data_path, "tfidf_features.csv")

# Load Data
engineered = pd.read_csv(engineered_path)
tfidf = pd.read_csv(tfidf_path)

print(f"Loaded engineered features: {engineered.shape}")
print(f"Loaded TF-IDF features: {tfidf.shape}")

# Prepare Data
y = engineered["fraudulent"]
engineered = engineered.drop(columns=["fraudulent"], errors="ignore")
tfidf = tfidf.drop(columns=["fraudulent"], errors="ignore")

# Encode Categorical Columns
print("\nEncoding categorical columns...")
cat_cols = engineered.select_dtypes(include="object").columns
for col in cat_cols:
    le = LabelEncoder()
    engineered[col] = le.fit_transform(engineered[col].astype(str))
print(f"Encoded {len(cat_cols)} categorical columns: {list(cat_cols)}")

# Combine engineered + TF-IDF
X = pd.concat([engineered.reset_index(drop=True), tfidf.reset_index(drop=True)], axis=1)
print(f"Combined feature matrix shape: {X.shape}")

X = X.select_dtypes(include=[np.number])
print(f"Final numeric feature shape: {X.shape}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Logistic Regression
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 4))
print(classification_report(y_test, y_pred_lr, zero_division=0))


# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf), 4))
print(classification_report(y_test, y_pred_rf, zero_division=0))

# Save Best Model
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)

best_model = rf if acc_rf > acc_lr else log_reg
best_name = "Random Forest" if acc_rf > acc_lr else "Logistic Regression"

model_path = os.path.join(models_path, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest Model: {best_name}")
print(f"Model saved successfully at: {model_path}")

# Saving metrics summary
metrics = {
    "Model": best_name,
    "Accuracy_LR": round(acc_lr, 4),
    "Accuracy_RF": round(acc_rf, 4),
}
pd.DataFrame([metrics]).to_csv(os.path.join(models_path, "model_metrics.csv"), index=False)
print(f"Metrics saved at: {os.path.join(models_path, 'model_metrics.csv')}")