
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# =========================
# Debug: App Start
# =========================
st.write("✅ App started successfully")

# =========================
# Load Model from HF (with caching)
# =========================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="amarg7/tourism-model",
        filename="tourism_model.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# =========================
# UI Title
# =========================
st.title("🌍 Tourism Package Prediction App")

st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

# =========================
# User Inputs
# =========================
Age = st.slider("Age", 18, 70, 30)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
NumberOfTrips = st.number_input("Number of Trips per Year", 0, 10, 2)
NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 5, 0)
MonthlyIncome = st.number_input("Monthly Income", 1000, 100000, 30000)

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
NumberOfFollowups = st.number_input("Number of Followups", 0, 10, 2)
DurationOfPitch = st.number_input("Duration of Pitch", 5, 60, 15)

Passport = st.selectbox("Has Passport?", [0, 1])
OwnCar = st.selectbox("Owns Car?", [0, 1])

# Categorical inputs
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

CityTier = st.selectbox("City Tier", [1, 2, 3])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

# =========================
# Create Input DataFrame
# =========================
input_data = pd.DataFrame([{
    "Age": Age,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfTrips": NumberOfTrips,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "ProductPitched": ProductPitched,
    "CityTier": CityTier,
    "PreferredPropertyStar": PreferredPropertyStar
}])

# =========================
# Prediction
# =========================
if st.button("Predict"):
    try:
        st.write("🔍 Running prediction...")

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Customer is likely to purchase the package!")
        else:
            st.error("❌ Customer is unlikely to purchase the package.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
