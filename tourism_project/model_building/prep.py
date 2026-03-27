
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/amarg7/tourism-predict/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unnecessary column
df.drop(columns=['CustomerID'], inplace=True)

# Handle missing values ONLY
df.fillna(method="ffill", inplace=True)

target_col = "ProdTaken"

X = df.drop(columns=[target_col])
y = df[target_col]

# Split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id="amarg7/tourism-predict",
        repo_type="dataset",
    )

print("Cleaned data uploaded.")
