import pandas as pd

# ----------------------------
# STEP 1: LOAD DATA
# ----------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ----------------------------
# STEP 2: DROP USELESS COLUMN
# ----------------------------
df.drop("customerID", axis=1, inplace=True)

# ----------------------------
# STEP 3: FIX TotalCharges
# ----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Missing in TotalCharges:", df["TotalCharges"].isnull().sum())

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ----------------------------
# STEP 4: BASIC ENCODING
# ----------------------------
yes_no_cols = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"
]

for col in yes_no_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# ----------------------------
# STEP 5: ONE-HOT ENCODING
# ----------------------------
df = pd.get_dummies(df, drop_first=True)

df = df.astype(int)

# ----------------------------
# STEP 6: SPLIT DATA
# ----------------------------
from sklearn.model_selection import train_test_split

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# STEP 7: TRAIN MODEL
# ----------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# STEP 8: PREDICTION
# ----------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.3).astype(int)

# ----------------------------
# STEP 9: EVALUATION
# ----------------------------
from sklearn.metrics import accuracy_score, classification_report

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# STEP 10: SAVE MODEL
# ----------------------------
import pickle

with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")