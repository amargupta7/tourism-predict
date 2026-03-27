# 🌍 Tourism Package Prediction - MLOps Project

## 📌 Overview

This project builds a **complete end-to-end MLOps pipeline** to predict whether a customer will purchase the **Wellness Tourism Package** offered by *Visit with Us*.

The system leverages **data-driven decision making**, enabling targeted marketing and improved customer acquisition through automated workflows.

---

## 🎯 Objective

- Predict customer purchase behavior (`ProdTaken`)
- Automate the ML lifecycle using MLOps principles
- Deploy a real-time prediction app
- Enable continuous integration and deployment

---

## 🧠 Problem Type

- **Supervised Learning**
- **Binary Classification**
  - `0` → No Purchase  
  - `1` → Purchase  

---

## 📊 Dataset Description

The dataset includes:

### 👤 Customer Details
- Age, Gender, Occupation, Monthly Income  
- Marital Status, Designation  
- Passport, Car Ownership  

### 🌍 Travel Behavior
- Number of Trips  
- Preferred Property Star  
- Number of Persons/Children Visiting  

### 📞 Interaction Data
- Pitch Satisfaction Score  
- Number of Follow-ups  
- Duration of Pitch  
- Product Pitched  

---

## ⚙️ MLOps Pipeline

The project implements a **fully automated CI/CD pipeline**:

### 🔄 Workflow Steps

#### 1. Data Registration
- Upload dataset to Hugging Face

#### 2. Data Preparation
- Clean data  
- Handle missing values  
- Train-test split  
- Upload processed data  

#### 3. Model Training
- Load data from Hugging Face  
- Train XGBoost model  
- Hyperparameter tuning (GridSearchCV)  
- Experiment tracking (MLflow)  

#### 4. Model Registry
- Best model uploaded to Hugging Face Model Hub  

#### 5. Deployment
- Streamlit app  
- Docker containerization  

#### 6. Hosting
- Deploy to Hugging Face Spaces  

#### 7. Automation
- GitHub Actions triggers full pipeline on every push  

---

## 🤖 Model Details

- **Algorithm:** XGBoost Classifier  

### Features:
- Numerical + Categorical (encoded)

### Techniques:
- Pipeline (preprocessing + model)  
- Hyperparameter tuning  
- Class imbalance handling  

---

## 🚀 Deployment

### 🌐 Streamlit App

The app allows users to:

- Input customer details  
- Get real-time predictions  

---

