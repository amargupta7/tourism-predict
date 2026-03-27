
# =========================
# Imports
# =========================
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import joblib

# Hugging Face
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

import mlflow

# =========================
# MLflow Setup
# =========================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

# =========================
# Hugging Face Login (FIX)
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

api = HfApi(token=HF_TOKEN)

# =========================
# Load Data from HF Dataset
# =========================
repo_id = "amarg7/tourism-predict"

Xtrain = pd.read_csv(f"hf://datasets/{repo_id}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{repo_id}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{repo_id}/ytrain.csv").values.ravel()
ytest = pd.read_csv(f"hf://datasets/{repo_id}/ytest.csv").values.ravel()

print("Dataset loaded from Hugging Face.")

# =========================
# Feature Groups
# =========================
numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'NumberOfTrips',
    'NumberOfChildrenVisiting', 'MonthlyIncome',
    'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
]

binary_features = ['Passport', 'OwnCar']

categorical_features = [
    col for col in Xtrain.columns
    if col not in numeric_features + binary_features
]

# =========================
# Preprocessor
# =========================
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    ("passthrough", binary_features),
    ("passthrough", categorical_features)
)

# =========================
# Model (Classification)
# =========================
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# =========================
# Hyperparameter Grid
# =========================
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8],
    'xgbclassifier__colsample_bytree': [0.7, 0.8]
}

# =========================
# Pipeline
# =========================
model_pipeline = make_pipeline(preprocessor, xgb_model)

# =========================
# Training + MLflow
# =========================
with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring='f1'
    )

    grid_search.fit(Xtrain, ytrain)

    # Log all parameter sets
    results = grid_search.cv_results_

    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])

    # Best model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_prob_test = best_model.predict_proba(Xtest)[:, 1]

    # Metrics
    mlflow.log_metrics({
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "test_accuracy": accuracy_score(ytest, y_pred_test),
        "train_f1": f1_score(ytrain, y_pred_train),
        "test_f1": f1_score(ytest, y_pred_test),
        "test_roc_auc": roc_auc_score(ytest, y_prob_test)
    })

    # =========================
    # Save Model
    # =========================
    model_path = "tourism_model.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved: {model_path}")

    # =========================
    # Upload to HF Model Hub
    # =========================
    model_repo_id = "amarg7/tourism-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
        print(f"Model repo '{model_repo_id}' exists.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{model_repo_id}'...")
        create_repo(repo_id=model_repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=model_repo_id,
        repo_type=repo_type,
    )

    print("Model uploaded to Hugging Face Hub.")
