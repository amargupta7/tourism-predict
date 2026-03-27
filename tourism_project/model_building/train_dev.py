
# =========================
# Imports
# =========================
import pandas as pd
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow

# =========================
# Load Local Dataset
# =========================
df = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully.")

df.drop(columns=['CustomerID'], inplace=True)

target_col = "ProdTaken"

X = df.drop(columns=[target_col])
y = df[target_col]

# Split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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
    'xgbclassifier__learning_rate': [0.01, 0.1]
}

# =========================
# MLflow
# =========================
mlflow.set_experiment("tourism-dev")

with mlflow.start_run():

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(Xtest)
    y_prob = best_model.predict_proba(Xtest)[:, 1]

    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics({
        "accuracy": accuracy_score(ytest, y_pred),
        "f1": f1_score(ytest, y_pred),
        "roc_auc": roc_auc_score(ytest, y_prob)
    })

    print("DEV Training Complete")
