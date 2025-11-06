import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("👔 Employee Attrition Prediction Dashboard")
st.markdown("""Dashboard predicts the chances of an employee leaving the current company.""")

# Load model and preprocessor
model = joblib.load("employee_attrition_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

# Sidebar inputs
st.sidebar.header("🔧 Employee Input Features")
st.sidebar.markdown("Enter the details for prediction below:")

# --- Categorical Inputs ---
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
job_role = st.sidebar.selectbox(
    "Job Role",
    ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager",
     "Manufacturing Director", "Healthcare Representative", "Human Resources", "Sales Representative"]
)
overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])

# --- Numeric & Ordinal Inputs ---
age = st.sidebar.number_input("Age", min_value=18, max_value=60, value=30)
distance_from_home = st.sidebar.number_input("Distance From Home (km)", min_value=1, max_value=50, value=5)
monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
performance_rating = st.sidebar.slider("Performance Rating", 1, 4, 3)
training_times_last_year = st.sidebar.number_input("Training Times Last Year", min_value=0, max_value=10, value=3)
job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
years_since_last_promotion = st.sidebar.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=2)
years_in_current_role = st.sidebar.number_input("Years In Current Role", min_value=0, max_value=20, value=4)
work_life_balance = st.sidebar.slider("Work Life Balance", 1, 4, 3)
years_at_company = st.sidebar.number_input("Years At Company", min_value=0, max_value=40, value=6)

# --- Already present in your code ---
relationship_satisfaction = st.sidebar.slider('Relationship Satisfaction', 1, 4, 3)
standard_hours = st.sidebar.number_input('Standard Hours', min_value=1, max_value=100, value=40)
job_involvement = st.sidebar.slider('Job Involvement', 1, 4, 3)
education = st.sidebar.slider('Education', 1, 5, 3)
daily_rate = st.sidebar.number_input('Daily Rate', min_value=0, max_value=1500, value=800)
monthly_rate = st.sidebar.number_input('Monthly Rate', min_value=1000, max_value=30000, value=20000)
stock_option_level = st.sidebar.slider('Stock Option Level', 0, 3, 1)
hourly_rate = st.sidebar.number_input('Hourly Rate', min_value=0, max_value=200, value=50)
education_field = st.sidebar.selectbox('Education Field',
    ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Technical Degree', 'Other'])
job_level = st.sidebar.slider('Job Level', 1, 5, 2)
environment_satisfaction = st.sidebar.slider('Environment Satisfaction', 1, 4, 3)
percent_salary_hike = st.sidebar.number_input('Percent Salary Hike', min_value=0, max_value=100, value=15)
years_with_curr_manager = st.sidebar.number_input('Years With Current Manager', min_value=0, max_value=40, value=5)
num_companies_worked = st.sidebar.number_input('Number of Companies Worked', min_value=0, max_value=10, value=3)
total_working_years = st.sidebar.number_input('Total Working Years', min_value=0, max_value=40, value=10)

# --- Combine all inputs ---
user_input = pd.DataFrame([{
    'Age': age,
    'BusinessTravel': business_travel,
    'Department': department,
    'DistanceFromHome': distance_from_home,
    'Education': education,
    'EducationField': education_field,
    'EnvironmentSatisfaction': environment_satisfaction,
    'Gender': gender,
    'JobInvolvement': job_involvement,
    'JobLevel': job_level,
    'JobRole': job_role,
    'JobSatisfaction': job_satisfaction,
    'MaritalStatus': marital_status,
    'MonthlyIncome': monthly_income,
    'NumCompaniesWorked': num_companies_worked,
    'OverTime': overtime,
    'PercentSalaryHike': percent_salary_hike,
    'PerformanceRating': performance_rating,
    'RelationshipSatisfaction': relationship_satisfaction,
    'StandardHours': standard_hours,
    'StockOptionLevel': stock_option_level,
    'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': training_times_last_year,
    'WorkLifeBalance': work_life_balance,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': years_in_current_role,
    'YearsSinceLastPromotion': years_since_last_promotion,
    'YearsWithCurrManager': years_with_curr_manager,
    'DailyRate': daily_rate,
    'HourlyRate': hourly_rate,
    'MonthlyRate': monthly_rate
}])

# --- Display preview ---
st.markdown("---")
st.subheader("📋 Employee Data Preview")
st.dataframe(user_input)

# --- Preprocessing and prediction ---
try:
    user_input_processed = preprocessor.transform(user_input)
    prediction = model.predict(user_input_processed)
    probability = model.predict_proba(user_input_processed)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Summary")
    if prediction[0] == 1:
        st.error(f"🚨 The employee is likely to **LEAVE**. (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"✅ The employee is likely to **STAY**. (Probability of leaving: {probability*100:.2f}%)")

    st.metric(label="Attrition Probability", value=f"{probability*100:.2f}%")
    st.progress(int(probability*100))

except ValueError as e:
    st.error(f"⚠️ Error during preprocessing: {e}")
    st.write("Missing columns in your input:", set(preprocessor.feature_names_in_) - set(user_input.columns))
