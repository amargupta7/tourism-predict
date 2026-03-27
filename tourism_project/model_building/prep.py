
# =========================
# Imports
# =========================
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from huggingface_hub import HfApi

# =========================
# Hugging Face Setup
# =========================
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/amarg7/tourism-predict/tourism.csv"

# =========================
# Load Data
# =========================
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# =========================
# Drop unnecessary columns
# =========================
df.drop(columns=['CustomerID'], inplace=True)

# =========================
# Define columns
# =========================
target_col = "ProdTaken"

# Categorical columns
nominal_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"
]

ordinal_cols = [
    "CityTier",
    "PreferredPropertyStar"
]

binary_cols = [
    "Passport",
    "OwnCar"
]

# =========================
# Handle Missing Values
# =========================
for col in nominal_cols:
    df[col].fillna("Unknown", inplace=True)

for col in ordinal_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in binary_cols:
    df[col].fillna(0, inplace=True)

df.fillna(df.median(numeric_only=True), inplace=True)

# =========================
# Encoding
# =========================

# Ordinal Encoding (preserves order)
ordinal_encoder = OrdinalEncoder()
df[ordinal_cols] = ordinal_encoder.fit_transform(df[ordinal_cols])

# One-hot encoding for nominal variables
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# =========================
# Split Data
# =========================
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Save Files
# =========================
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train-test split saved successfully.")

# =========================
# Upload to Hugging Face
# =========================
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="amarg7/tourism-predict",
        repo_type="dataset",
    )

print("Files uploaded to Hugging Face successfully.")
