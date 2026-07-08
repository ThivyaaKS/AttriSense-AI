import pandas as pd

# Load Dataset
df = pd.read_csv("dataset/employee_attrition.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

# Load Dataset
df = pd.read_csv("dataset/employee_attrition.csv")

# ==========================
# DATA CLEANING
# ==========================

print("========== DATA CLEANING ==========")

# 1. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Check Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# 3. Remove Duplicate Rows (if any)
df = df.drop_duplicates()

# 4. Check Data Types
print("\nData Types:")
print(df.dtypes)

# 5. Remove Unnecessary Columns
columns_to_drop = ["EmployeeCount", "Over18", "StandardHours"]

df = df.drop(columns=columns_to_drop)

print("\nRemaining Columns:")
print(df.columns.tolist())

# 6. Final Dataset Shape
print("\nCleaned Dataset Shape:")
print(df.shape)
import matplotlib.pyplot as plt
import seaborn as sns

print("\n========== EDA ==========")

# Attrition Count
plt.figure(figsize=(6,4))
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Distribution")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Monthly Income Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["MonthlyIncome"], bins=30, kde=True)
plt.title("Monthly Income Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.select_dtypes(include="number").corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
import numpy as np

# ==========================
# FEATURE ENGINEERING
# ==========================

np.random.seed(42)

df["RemoteWork"] = np.random.choice(["Yes", "No"], size=len(df))

df["TrainingHours"] = np.random.randint(5, 51, size=len(df))

df["PromotionDelay"] = np.random.randint(0, 6, size=len(df))

df["ManagerFeedback"] = np.random.randint(1, 6, size=len(df))

df["MentalHealthScore"] = np.random.randint(1, 11, size=len(df))

print("\n========== CUSTOM FEATURES ADDED ==========")
print(df[[
    "RemoteWork",
    "TrainingHours",
    "PromotionDelay",
    "ManagerFeedback",
    "MentalHealthScore"
]].head())
from sklearn.preprocessing import LabelEncoder

print("\n========== LABEL ENCODING ==========")

le = LabelEncoder()

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

print(df.head())
print("\n========== FEATURE ENGINEERING ==========")

# Monthly Income Category
df["IncomeCategory"] = pd.cut(
    df["MonthlyIncome"],
    bins=[0, 5000, 10000, 20000],
    labels=["Low", "Medium", "High"]
)

# Total Experience Category
df["ExperienceLevel"] = pd.cut(
    df["TotalWorkingYears"],
    bins=[-1, 5, 10, 40],
    labels=["Fresher", "Mid", "Senior"]
)
df["IncomeCategory"] = df["IncomeCategory"].cat.codes
df["ExperienceLevel"] = df["ExperienceLevel"].cat.codes

print(df[["MonthlyIncome", "IncomeCategory",
          "TotalWorkingYears", "ExperienceLevel"]].head())
from sklearn.model_selection import train_test_split

print("\n========== TRAIN TEST SPLIT ==========")

X = df[[
    "Age",
    "MonthlyIncome",
    "TotalWorkingYears"
]]
print(X.columns.tolist())
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n========== RANDOM FOREST MODEL ==========")

# Create Model
model = RandomForestClassifier(random_state=42)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import joblib

joblib.dump(model, "models/employee_attrition_model.pkl")

print("✅ Model Saved Successfully!")