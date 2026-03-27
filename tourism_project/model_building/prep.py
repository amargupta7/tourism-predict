import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/amarg7/tourism-predict/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# =========================
# Drop unnecessary columns
# =========================
df.drop(columns=['CustomerID', 'Unnamed: 0'], errors='ignore', inplace=True)

# =========================
# Separate features
# =========================
target_col = "ProdTaken"

# Identify column types
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove(target_col)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# =========================
# Handle missing values
# =========================

# Numerical → median
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical → mode
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# =========================
# Split
# =========================
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# =========================
# Upload to Hugging Face
# =========================
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id="amarg7/tourism-predict",
        repo_type="dataset",
    )

print("Cleaned data uploaded successfully.")
