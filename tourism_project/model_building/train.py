
# =========================
# Imports
# =========================
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import joblib
import mlflow

from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# =========================
# Setup
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

api = HfApi(token=HF_TOKEN)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-prod")

# =========================
# Load HF Train/Test
# =========================
repo_id = "amarg7/tourism-predict"

Xtrain = pd.read_csv(f"hf://datasets/{repo_id}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{repo_id}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{repo_id}/ytrain.csv").values.ravel()
ytest = pd.read_csv(f"hf://datasets/{repo_id}/ytest.csv").values.ravel()

print("HF data loaded")

# =========================
# Features
# =========================
numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'NumberOfTrips',
    'NumberOfChildrenVisiting', 'MonthlyIncome',
    'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'MaritalStatus', 'Designation', 'ProductPitched'
]

binary_features = ['Passport', 'OwnCar', 'CityTier', 'PreferredPropertyStar']

# =========================
# Pipeline
# =========================
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ("passthrough", binary_features)
)

model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')

pipeline = make_pipeline(preprocessor, model)

# =========================
# Hyperparameters
# =========================
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8],
    'xgbclassifier__colsample_bytree': [0.7, 0.8]
}

# =========================
# Training
# =========================
with mlflow.start_run():

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # =========================
    # Log ALL parameter combinations
    # =========================
    results = grid.cv_results_

    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])

    # =========================
    # Best Model
    # =========================
    best_model = grid.best_estimator_
    mlflow.log_params(grid.best_params_)

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

    # =========================
    # Upload Model
    # =========================
    model_repo = "amarg7/tourism-model"

    try:
        api.repo_info(repo_id=model_repo, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=model_repo, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=model_repo,
        repo_type="model"
    )

print("PROD model uploaded")
