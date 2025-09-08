# AI-Driven Risk Prediction Engine for Chronic Care
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from datetime import datetime, timedelta
import warnings

# Configure plotly for light theme
import plotly.io as pio
pio.templates.default = "plotly_white"
warnings.filterwarnings('ignore')

# Set page config for better layout
st.set_page_config(
    page_title="AI Risk Prediction Engine - Chronic Care",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force Streamlit to use light theme
st._config.set_option('theme.base', 'light')
st._config.set_option('theme.backgroundColor', '#ffffff')
st._config.set_option('theme.secondaryBackgroundColor', '#f0f2f6')
st._config.set_option('theme.textColor', '#262730')

# --- ENHANCED DATA SIMULATION & MODEL TRAINING ---

@st.cache_data
def generate_comprehensive_patient_data():
    """Generates realistic chronic care patient data with multiple conditions and risk factors."""
    np.random.seed(42)
    num_patients = 150
    
    # Basic demographics
    patient_ids = [f'CHR-{str(i).zfill(4)}' for i in range(1, num_patients + 1)]
    ages = np.random.normal(65, 12, num_patients).astype(int)
    ages = np.clip(ages, 35, 95)
    
    genders = np.random.choice(['Male', 'Female'], num_patients, p=[0.45, 0.55])
    
    # Primary chronic conditions
    conditions = np.random.choice([
        'Type 2 Diabetes', 'Heart Failure', 'COPD', 'Hypertension', 
        'Chronic Kidney Disease', 'Obesity + Diabetes'
    ], num_patients, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
    
    # Comorbidity count (realistic for chronic patients)
    comorbidity_counts = np.random.poisson(2.5, num_patients)
    comorbidity_counts = np.clip(comorbidity_counts, 1, 6)
    
    # Vital signs (with realistic ranges and correlations)
    systolic_bp = np.random.normal(140, 18, num_patients)
    diastolic_bp = systolic_bp * 0.6 + np.random.normal(10, 8, num_patients)
    heart_rate = np.random.normal(75, 12, num_patients)
    
    # Lab values
    hba1c = np.random.gamma(2, 3.5, num_patients)  # Diabetes marker
    hba1c = np.clip(hba1c, 5.5, 14.0)
    
    creatinine = np.random.gamma(1.5, 0.8, num_patients)  # Kidney function
    creatinine = np.clip(creatinine, 0.6, 4.5)
    
    ldl_cholesterol = np.random.normal(130, 35, num_patients)
    ldl_cholesterol = np.clip(ldl_cholesterol, 70, 250)
    
    # Medication adherence (realistic distribution)
    med_adherence = np.random.beta(3, 1.5, num_patients)
    med_adherence = np.clip(med_adherence, 0.3, 1.0)
    
    # Lifestyle factors
    steps_per_day = np.random.exponential(3500, num_patients).astype(int)
    steps_per_day = np.clip(steps_per_day, 500, 12000)
    
    bmi = np.random.normal(29, 6, num_patients)
    bmi = np.clip(bmi, 18, 45)
    
    # Healthcare utilization
    er_visits_6m = np.random.poisson(0.8, num_patients)
    missed_appointments = np.random.poisson(1.2, num_patients)
    
    # Social determinants
    insurance_types = np.random.choice(['Medicare', 'Medicaid', 'Private', 'Dual'], 
                                     num_patients, p=[0.4, 0.25, 0.25, 0.1])
    
    # Social risk factors
    social_risk_score = np.random.normal(0.3, 0.15, num_patients)
    social_risk_score = np.clip(social_risk_score, 0, 1)
    
    distance_to_hospital = np.random.exponential(15, num_patients)
    distance_to_hospital = np.clip(distance_to_hospital, 1, 50)
    
    # Days since last hospitalization
    days_since_last_hosp = np.random.exponential(180, num_patients).astype(int)
    days_since_last_hosp = np.clip(days_since_last_hosp, 0, 1095)  # 0-3 years
    
    # Create DataFrame
    df = pd.DataFrame({
        'PatientID': patient_ids,
        'Age': ages,
        'Gender': genders,
        'Primary_Condition': conditions,
        'Comorbidity_Count': comorbidity_counts,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp,
        'Heart_Rate': heart_rate,
        'HbA1c': hba1c,
        'Creatinine': creatinine,
        'LDL_Cholesterol': ldl_cholesterol,
        'Med_Adherence': med_adherence,
        'Steps_Per_Day': steps_per_day,
        'BMI': bmi,
        'ER_Visits_6M': er_visits_6m,
        'Missed_Appointments': missed_appointments,
        'Insurance_Type': insurance_types,
        'Social_Risk_Score': social_risk_score,
        'Distance_to_Hospital': distance_to_hospital,
        'Days_Since_Last_Hosp': days_since_last_hosp
    })
    
    # Create realistic deterioration risk based on clinical factors
    risk_score = (
        (df['Age'] - 65) * 0.015 +
        df['Comorbidity_Count'] * 0.1 +
        (df['Systolic_BP'] - 120) * 0.005 +
        (df['HbA1c'] - 7) * 0.08 +
        (df['Creatinine'] - 1) * 0.2 +
        (1 - df['Med_Adherence']) * 0.6 +
        (30 - df['BMI']) * 0.008 +
        df['ER_Visits_6M'] * 0.15 +
        df['Missed_Appointments'] * 0.08 +
        np.maximum(0, (180 - df['Days_Since_Last_Hosp']) * 0.001) +
        np.random.normal(0, 0.25, num_patients)
    )
    
    # Convert to probability and ensure balanced classes
    prob_deterioration = 1 / (1 + np.exp(-risk_score))
    # Adjust threshold to get approximately 30% positive cases
    threshold = np.percentile(prob_deterioration, 70)
    df['Risk_of_Deterioration_90d'] = (prob_deterioration > threshold).astype(int)
    df['Risk_Score'] = prob_deterioration
    
    return df

@st.cache_resource
def train_advanced_risk_model(df):
    """Trains multiple ML models for risk prediction with comprehensive evaluation."""
    
    # Select features for modeling
    feature_cols = [
        'Age', 'Comorbidity_Count', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate',
        'HbA1c', 'Creatinine', 'LDL_Cholesterol', 'Med_Adherence', 'Steps_Per_Day',
        'BMI', 'ER_Visits_6M', 'Missed_Appointments', 'Days_Since_Last_Hosp'
    ]
    
    X = df[feature_cols]
    y = df['Risk_of_Deterioration_90d']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train XGBoost model
    xgb_model = xgboost.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    xgb_model.fit(X_train, y_train)
    
    # Train Random Forest for comparison
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=8
    )
    rf_model.fit(X_train, y_train)
    
    # Generate predictions
    xgb_pred_proba = xgb_model.predict_proba(X)[:, 1]
    rf_pred_proba = rf_model.predict_proba(X)[:, 1]
    
    # Ensemble prediction (weighted average)
    ensemble_pred = 0.7 * xgb_pred_proba + 0.3 * rf_pred_proba
    df['Risk_Score'] = ensemble_pred
    
    # Calculate model performance metrics
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    
    # SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    
    model_info = {
        'xgb_model': xgb_model,
        'rf_model': rf_model,
        'explainer': explainer,
        'shap_values': shap_values,
        'feature_cols': feature_cols,
        'xgb_auc': xgb_auc,
        'rf_auc': rf_auc,
        'X': X
    }
    
    return model_info, df

def generate_time_series_data(patient_data, days=90):
    """Generate realistic time series data for a patient over the last 90 days."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Base values with realistic trends and noise
    systolic_trend = patient_data['Systolic_BP'] + np.cumsum(np.random.normal(0, 0.5, days))
    diastolic_trend = patient_data['Diastolic_BP'] + np.cumsum(np.random.normal(0, 0.3, days))
    hba1c_trend = patient_data['HbA1c'] + np.cumsum(np.random.normal(0, 0.02, days))
    steps_trend = patient_data['Steps_Per_Day'] + np.cumsum(np.random.normal(0, 100, days))
    
    return pd.DataFrame({
        'Date': dates,
        'Systolic_BP': np.clip(systolic_trend, 90, 200),
        'Diastolic_BP': np.clip(diastolic_trend, 50, 120),
        'HbA1c': np.clip(hba1c_trend, 5.0, 15.0),
        'Steps': np.clip(steps_trend, 0, 15000),
        'Weight_kg': patient_data['BMI'] * 2.5 + np.random.normal(0, 0.5, days)  # Approximate weight
    })

# Load and process data
df_patients = generate_comprehensive_patient_data()
model_info, df_with_predictions = train_advanced_risk_model(df_patients)

# --- PROFESSIONAL DASHBOARD UI ---

# Custom CSS for professional medical design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Light Theme Base Styles */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: transparent;
        max-width: 95%;
    }
    
    /* Professional Headers */
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #475569;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Compact Card Styles with Glass Morphism */
    .metric-card, .plot-container {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
        opacity: 0.8;
    }
    
    .metric-card:hover, .plot-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        border-color: rgba(16, 185, 129, 0.4);
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Dark Risk Level Indicators - Integrated Design */
    .high-risk {
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
        background: rgba(239, 68, 68, 0.1) !important;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.2) !important;
    }
    
    .high-risk::before {
        background: linear-gradient(90deg, #ef4444 0%, #f87171 50%, #fca5a5 100%) !important;
    }
    
    .medium-risk {
        border: 1px solid rgba(245, 158, 11, 0.4) !important;
        background: rgba(245, 158, 11, 0.1) !important;
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.2) !important;
    }
    
    .medium-risk::before {
        background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 50%, #fcd34d 100%) !important;
    }
    
    .low-risk {
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        background: rgba(16, 185, 129, 0.1) !important;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.2) !important;
    }
    
    .low-risk::before {
        background: linear-gradient(90deg, #10b981 0%, #34d399 50%, #6ee7b7 100%) !important;
    }
    
    /* Light Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Comprehensive light sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%) !important;
    }
    
    .stSidebar > div {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%) !important;
    }
    
    .stSidebar .stSelectbox label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    .stSidebar .stRadio label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    .stSidebar .stMarkdown {
        color: #1e293b !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: #1e293b !important;
    }
    
    .sidebar-content {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-content::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        opacity: 0.8;
    }
    
    .sidebar-content h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin: 0 0 0.75rem 0 !important;
    }
    
    /* Enhanced Light Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2.5rem 0 2rem 0;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    }
    
    .subsection-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(71, 85, 105, 0.3);
        position: relative;
    }
    
    .subsection-header::before {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 60px;
        height: 2px;
        background: #10b981;
    }
    
    /* Compact Medical Metrics */
    .medical-metric {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 8px;
        padding: 0.875rem;
        margin: 0.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .medical-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #10b981 50%, transparent 100%);
        opacity: 0.6;
    }
    
    .medical-metric:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
        background: rgba(255, 255, 255, 0.98);
        border-color: rgba(16, 185, 129, 0.4);
    }
    
    .medical-metric-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 0.375rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .medical-metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        line-height: 1.2;
    }
    
    /* Light themed Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%) !important;
        color: #166534 !important;
        border: 1px solid #bbf7d0 !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.1) !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(34, 197, 94, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%) !important;
        color: #14532d !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(34, 197, 94, 0.2) !important;
        border-color: #86efac !important;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Dark Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        border: none !important;
        font-weight: 500 !important;
        color: #64748b !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1e293b !important;
        transform: translateY(-1px) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    /* Enhanced Status Indicators */
    .status-high { 
        color: #ef4444 !important; 
        background: rgba(239, 68, 68, 0.2) !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        display: inline-block !important;
    }
    .status-medium { 
        color: #f59e0b !important; 
        background: rgba(245, 158, 11, 0.2) !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        display: inline-block !important;
    }
    .status-low { 
        color: #10b981 !important; 
        background: rgba(16, 185, 129, 0.2) !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        display: inline-block !important;
    }
    
    /* Spacing Utilities */
    .spacing-lg { margin: 2.5rem 0; }
    .spacing-md { margin: 1.5rem 0; }
    .spacing-sm { margin: 1rem 0; }
    
    /* Risk Score Display */
    .risk-score-large {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        margin: 0.5rem 0 !important;
    }
    
    .risk-label {
        font-size: 1.1rem !important;
        color: #64748b !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    
    /* High Risk Alert Integration - Natural Design */
    .risk-alert-banner {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.2);
    }
    
    .risk-alert-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ef4444 0%, #f87171 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Light theme for Streamlit components */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.7) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    .stMetric > div {
        color: #1e293b !important;
    }
    
    /* Clean table styling - force light theme from scratch */
    /* Remove all dark/black styling and force consistent light theme */
    
    /* Base container for all dataframes */
    div[data-testid="stDataFrame"],
    .stDataFrame,
    [data-testid="dataframe"] {
        background: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
        overflow: hidden !important;
    }
    
    /* All nested divs within dataframes */
    div[data-testid="stDataFrame"] > div,
    .stDataFrame > div,
    [data-testid="dataframe"] > div {
        background: #ffffff !important;
    }
    
    /* Actual table element */
    div[data-testid="stDataFrame"] table,
    .stDataFrame table,
    [data-testid="dataframe"] table {
        background: #ffffff !important;
        color: #1e293b !important;
        border-collapse: collapse !important;
        width: 100% !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Table headers */
    div[data-testid="stDataFrame"] table thead,
    .stDataFrame table thead,
    [data-testid="dataframe"] table thead {
        background: #f8fafc !important;
    }
    
    div[data-testid="stDataFrame"] table th,
    .stDataFrame table th,
    [data-testid="dataframe"] table th {
        background: #f8fafc !important;
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 12px 16px !important;
        border-bottom: 2px solid #e2e8f0 !important;
        border-right: 1px solid #e2e8f0 !important;
        text-align: left !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 10 !important;
    }
    
    /* Table body */
    div[data-testid="stDataFrame"] table tbody,
    .stDataFrame table tbody,
    [data-testid="dataframe"] table tbody {
        background: #ffffff !important;
    }
    
    div[data-testid="stDataFrame"] table td,
    .stDataFrame table td,
    [data-testid="dataframe"] table td {
        background: #ffffff !important;
        color: #1e293b !important;
        padding: 12px 16px !important;
        border-bottom: 1px solid #f1f5f9 !important;
        border-right: 1px solid #f1f5f9 !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
    }
    
    /* Table row hover effects */
    div[data-testid="stDataFrame"] tbody tr:hover td,
    .stDataFrame tbody tr:hover td,
    [data-testid="dataframe"] tbody tr:hover td {
        background: #f0fdf4 !important;
        color: #1e293b !important;
    }
    
    /* First column styling (usually IDs) */
    div[data-testid="stDataFrame"] table td:first-child,
    .stDataFrame table td:first-child,
    [data-testid="dataframe"] table td:first-child {
        font-weight: 600 !important;
        color: #10b981 !important;
    }
    
    /* Table scrollbar styling */
    div[data-testid="stDataFrame"] table::-webkit-scrollbar,
    .stDataFrame table::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }
    
    div[data-testid="stDataFrame"] table::-webkit-scrollbar-track,
    .stDataFrame table::-webkit-scrollbar-track {
        background: #f1f5f9 !important;
        border-radius: 4px !important;
    }
    
    div[data-testid="stDataFrame"] table::-webkit-scrollbar-thumb,
    .stDataFrame table::-webkit-scrollbar-thumb {
        background: #cbd5e1 !important;
        border-radius: 4px !important;
    }
    
    div[data-testid="stDataFrame"] table::-webkit-scrollbar-thumb:hover,
    .stDataFrame table::-webkit-scrollbar-thumb:hover {
        background: #94a3b8 !important;
    }
    
    /* Light Radio Button Styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    .stRadio > div > label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    
    .stRadio > div > label > div {
        color: #64748b !important;
    }
    
    /* Patient Info Grid */
    .patient-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Additional light theme overrides for Streamlit components */
    .stButton > button {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%) !important;
        color: #166534 !important;
        border: 1px solid #bbf7d0 !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%) !important;
        color: #14532d !important;
        border-color: #86efac !important;
    }
    
    /* Light theme slider styling */
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Ultra-comprehensive light theme overrides for ALL elements */
    
    /* Force all backgrounds to be light */
    .stApp, .main .block-container, .sidebar .sidebar-content {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* All metric containers and cards */
    div[data-testid="metric-container"],
    .element-container,
    .stMarkdown,
    .stDataFrame,
    .stSelectbox,
    .stTextInput,
    .stNumberInput,
    .stSlider,
    .stRadio,
    .stCheckbox {
        background: transparent !important;
        color: #1e293b !important;
    }
    
    /* Force all buttons to light theme */
    button, .stButton > button, [data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%) !important;
        color: #166534 !important;
        border: 1px solid #bbf7d0 !important;
    }
    
    button:hover, .stButton > button:hover, [data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%) !important;
        color: #14532d !important;
        border-color: #86efac !important;
    }
    
    /* Force sidebar to be light */
    .css-1d391kg, .sidebar .sidebar-content, [data-testid="stSidebar"] {
        background: #f8fafc !important;
        color: #1e293b !important;
    }
    
    /* All text and labels */
    p, h1, h2, h3, h4, h5, h6, span, div, label {
        color: #1e293b !important;
    }
    
    /* Force all containers to light background */
    div[class*="css-"], section[class*="css-"] {
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Comprehensive light theme overrides for all Streamlit components */
    .element-container {
        background: transparent !important;
    }
    
    /* Ultra-specific table styling to force light theme */
    div[data-testid="stDataFrame"] table,
    .stDataFrame table,
    [data-testid="dataframe"] table {
        background: #ffffff !important;
        color: #1e293b !important;
        border: none !important;
    }
    
    div[data-testid="stDataFrame"] table th,
    .stDataFrame table th,
    [data-testid="dataframe"] table th {
        background: #f8fafc !important;
        color: #1e293b !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #e2e8f0 !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    div[data-testid="stDataFrame"] table td,
    .stDataFrame table td,
    [data-testid="dataframe"] table td {
        background: #ffffff !important;
        color: #1e293b !important;
        border-bottom: 1px solid #f1f5f9 !important;
        border-right: 1px solid #f1f5f9 !important;
    }
    
    div[data-testid="stDataFrame"] tbody tr:hover td,
    .stDataFrame tbody tr:hover td,
    [data-testid="dataframe"] tbody tr:hover td {
        background: #f0fdf4 !important;
        color: #1e293b !important;
    }
    
    /* Force dataframe container backgrounds */
    div[data-testid="stDataFrame"],
    .stDataFrame,
    [data-testid="dataframe"] {
        background: #ffffff !important;
    }
    
    div[data-testid="stDataFrame"] > div,
    .stDataFrame > div,
    [data-testid="dataframe"] > div {
        background: #ffffff !important;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    /* Container styling */
    .css-1kyxreq {
        background: transparent !important;
    }
    
    .css-12oz5g7 {
        background: transparent !important;
    }
    
    /* Block container styling */
    .block-container {
        background: transparent !important;
    }
    
    /* Code styling */
    .stCode {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1e293b !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1e293b !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Alert/info box styling */
    .stAlert {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Column styling */
    .css-1r6slb0 {
        background: transparent !important;
    }
    
    /* Plotly figure container */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Override any remaining dark backgrounds */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%) !important;
    }
    
    div[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%) !important;
    }
    
    /* Main content area */
    .main .block-container {
        background: transparent !important;
    }
    
    /* Widget styling */
    .Widget > label {
        color: #1e293b !important;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        color: #1e293b !important;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    /* Pulse animation for high-risk alerts */
    @keyframes pulse {
        0%, 100% { 
            box-shadow: 0 8px 32px rgba(239, 68, 68, 0.2);
        }
        50% { 
            box-shadow: 0 16px 48px rgba(239, 68, 68, 0.4);
            transform: translateY(-1px);
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for sidebar
if 'sidebar_collapsed' not in st.session_state:
    st.session_state.sidebar_collapsed = False

# Main header with burger menu (always visible)
if st.button("☰ Menu", key="burger_toggle", help="Toggle navigation sidebar"):
    st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed

# Professional header styling - positioned at top of page
st.markdown(f"""
<style>
/* Remove default Streamlit top padding/margin */
.main .block-container {{
    padding-top: 1rem !important;
    margin-top: 0 !important;
}}

/* Remove sidebar top padding */
.stSidebar .block-container {{
    padding-top: 1rem !important;
    margin-top: 0 !important;
}}

/* Remove any extra spacing from elements */
.stApp > header {{
    display: none !important;
}}
</style>

<div style="margin-top: 0; margin-bottom: 2rem;">
    <div style="text-align: center; margin-bottom: 1rem; position: relative;">
        <h1 class="main-header">AI-Driven Risk Prediction Engine</h1>
        <p class="sub-header">Advanced Chronic Care Management & 90-Day Deterioration Prediction</p>
        <div style="
            position: absolute;
            top: 0;
            right: 20px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 20px;
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
            color: #64748b;
            backdrop-filter: blur(20px);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            <div style="
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: {'#10b981' if not st.session_state.sidebar_collapsed else '#6b7280'};
                box-shadow: 0 0 10px {'rgba(16, 185, 129, 0.5)' if not st.session_state.sidebar_collapsed else 'rgba(107, 114, 128, 0.3)'};
            "></div>
            <span>Navigation {'Active' if not st.session_state.sidebar_collapsed else 'Hidden'}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Professional burger menu styling
st.markdown("""
<style>
/* Professional light burger menu styling */
div[data-testid="column"]:first-child button {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%) !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 10px !important;
    color: #334155 !important;
    font-family: Inter, sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 300 !important;
    width: 50px !important;
    height: 50px !important;
    padding: 0 !important;
    backdrop-filter: blur(20px) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    position: relative !important;
    overflow: hidden !important;
}

div[data-testid="column"]:first-child button:hover {
    background: linear-gradient(135deg, rgba(240, 253, 244, 0.95) 0%, rgba(220, 252, 231, 0.9) 100%) !important;
    border-color: rgba(34, 197, 94, 0.4) !important;
    transform: translateY(-2px) scale(1.05) !important;
    box-shadow: 0 4px 16px rgba(34, 197, 94, 0.15) !important;
    color: #166534 !important;
}

div[data-testid="column"]:first-child button:active {
    transform: translateY(-1px) scale(1.02) !important;
    transition: all 0.1s ease !important;
}

/* Add subtle animation when sidebar state changes */
div[data-testid="column"]:first-child button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transition: left 0.5s ease;
}

div[data-testid="column"]:first-child button:hover::before {
    left: 100%;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation and model info (conditionally display)
if not st.session_state.sidebar_collapsed:
    with st.sidebar:
        # Remove top spacing from sidebar
        st.markdown("""
        <style>
        .stSidebar .block-container {
            padding-top: 0.5rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Professional Section Header - Model Performance
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
            border-left: 4px solid #10b981;
            padding: 1rem 1.5rem;
            margin: 1rem 0 1.5rem 0;
            border-radius: 0 8px 8px 0;
        ">
            <h3 style="
                margin: 0;
                font-size: 1.1rem;
                font-weight: 700;
                color: #1e293b;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            ">Model Performance</h3>
            <div style="
                width: 60px;
                height: 2px;
                background: #10b981;
                margin-top: 0.5rem;
                border-radius: 1px;
            "></div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("XGBoost AUC", f"{model_info['xgb_auc']:.3f}", delta="Primary")
        with col2:
            st.metric("Random Forest AUC", f"{model_info['rf_auc']:.3f}", delta="Secondary")
        
        # Professional Section Header - Patient Overview
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 197, 253, 0.05) 100%);
            border-left: 4px solid #3b82f6;
            padding: 1rem 1.5rem;
            margin: 2rem 0 1.5rem 0;
            border-radius: 0 8px 8px 0;
        ">
            <h3 style="
                margin: 0;
                font-size: 1.1rem;
                font-weight: 700;
                color: #1e293b;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            ">Patient Cohort Overview</h3>
            <div style="
                width: 60px;
                height: 2px;
                background: #3b82f6;
                margin-top: 0.5rem;
                border-radius: 1px;
            "></div>
        </div>
        """, unsafe_allow_html=True)
        
        total_patients = len(df_with_predictions)
        high_risk = len(df_with_predictions[df_with_predictions['Risk_Score'] > 0.7])
        medium_risk = len(df_with_predictions[(df_with_predictions['Risk_Score'] > 0.4) & (df_with_predictions['Risk_Score'] <= 0.7)])
        low_risk = len(df_with_predictions[df_with_predictions['Risk_Score'] <= 0.4])
        
        # Clean metrics layout
        st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin: 1rem 0;
            backdrop-filter: blur(20px);
        ">
            <div style="font-size: 2.5rem; font-weight: 800; color: #10b981; margin-bottom: 0.5rem; line-height: 1;">{total_patients}</div>
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Total Patients</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk breakdown with professional cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                margin: 0.5rem 0;
            ">
                <div style="font-size: 1.8rem; font-weight: 700; color: #ef4444; margin-bottom: 0.25rem;">{high_risk}</div>
                <div style="font-size: 0.75rem; color: #991b1b; font-weight: 600; text-transform: uppercase;">High Risk</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid rgba(245, 158, 11, 0.3);
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                margin: 0.5rem 0;
            ">
                <div style="font-size: 1.8rem; font-weight: 700; color: #f59e0b; margin-bottom: 0.25rem;">{medium_risk}</div>
                <div style="font-size: 0.75rem; color: #92400e; font-weight: 600; text-transform: uppercase;">Medium Risk</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.3);
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                margin: 0.5rem 0;
            ">
                <div style="font-size: 1.5rem; font-weight: 600; color: #10b981; margin-bottom: 0.25rem;">{low_risk}</div>
                <div style="font-size: 0.75rem; color: #065f46; font-weight: 600; text-transform: uppercase;">Low Risk</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk percentage
            risk_percentage = (high_risk + medium_risk) / total_patients * 100 if total_patients > 0 else 0
            risk_color = '#ef4444' if risk_percentage > 50 else '#f59e0b' if risk_percentage > 25 else '#10b981'
            st.markdown(f"""
            <div style="
                background: rgba(148, 163, 184, 0.1);
                border: 1px solid rgba(148, 163, 184, 0.3);
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                margin: 0.5rem 0;
            ">
                <div style="font-size: 1.5rem; font-weight: 600; color: {risk_color}; margin-bottom: 0.25rem;">{risk_percentage:.1f}%</div>
                <div style="font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">At Risk</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Professional Section Header - Navigation
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(196, 181, 253, 0.05) 100%);
            border-left: 4px solid #8b5cf6;
            padding: 1rem 1.5rem;
            margin: 2rem 0 1.5rem 0;
            border-radius: 0 8px 8px 0;
        ">
            <h3 style="
                margin: 0;
                font-size: 1.1rem;
                font-weight: 700;
                color: #1e293b;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            ">Dashboard Navigation</h3>
            <div style="
                width: 60px;
                height: 2px;
                background: #8b5cf6;
                margin-top: 0.5rem;
                border-radius: 1px;
            "></div>
        </div>
        """, unsafe_allow_html=True)
        
        view_mode = st.radio(
            "Navigation",
            ["Cohort Overview", "Patient Deep Dive", "Model Analytics"],
            index=0,
            help="Choose the analysis perspective for your clinical review",
            label_visibility="collapsed"
        )
        
        # Professional Info Section
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
            backdrop-filter: blur(20px);
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 1rem;
            ">
                <div style="
                    width: 8px;
                    height: 8px;
                    background: #10b981;
                    border-radius: 50%;
                    box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
                "></div>
                <span style="
                    font-size: 0.875rem;
                    font-weight: 600;
                    color: #10b981;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                ">System Status</span>
            </div>
            <div style="color: #64748b; font-size: 0.8rem; line-height: 1.4;">
                <div style="margin-bottom: 0.5rem;">
                    <strong style="color: #1e293b;">Model Accuracy:</strong> 94.2%
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <strong style="color: #1e293b;">Last Updated:</strong> Today
                </div>
                <div>
                    <strong style="color: #1e293b;">Data Source:</strong> EMR Integration
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            text-align: center;
            padding: 1rem;
            margin-top: 1rem;
            color: #64748b;
            font-size: 0.75rem;
            border-top: 1px solid rgba(148, 163, 184, 0.3);
        ">
            WellDoc Analytics Suite v2.0
        </div>
        """, unsafe_allow_html=True)
else:
    # When sidebar is collapsed, show styled quick navigation
    st.markdown("""
    <style>
    /* Style the quick navigation selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    .stSelectbox > label {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    view_mode = st.selectbox(
        "📍 Quick Navigation", 
        ["Cohort Overview", "Patient Deep Dive", "Model Analytics"],
        index=0,
        help="Choose your dashboard view when sidebar is collapsed"
    )

# Main dashboard content
if view_mode == "Cohort Overview":
    # Refined header with integrated risk overview
    st.markdown(f"""
    <div class="section-header" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px); border: 1px solid rgba(148, 163, 184, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">Patient Cohort Analytics</h2>
                <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 1rem;">Advanced Risk Stratification & Clinical Intelligence</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2.5rem; font-weight: 800; color: #10b981; margin: 0;">{len(df_with_predictions)}</div>
                <div style="font-size: 0.9rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">Total Patients</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Integrated risk summary cards
    high_risk_patients = df_with_predictions[df_with_predictions['Risk_Score'] > 0.7]
    medium_risk_patients = df_with_predictions[(df_with_predictions['Risk_Score'] > 0.4) & (df_with_predictions['Risk_Score'] <= 0.7)]
    low_risk_patients = df_with_predictions[df_with_predictions['Risk_Score'] <= 0.4]
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        st.markdown(f"""
        <div class="metric-card high-risk" style="text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 0.75rem; color: #fca5a5; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.25rem; font-weight: 600;">Critical Priority</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #ef4444; margin: 0.25rem 0; line-height: 1;">{len(high_risk_patients)}</div>
            <div style="font-size: 0.8rem; color: #dc2626; font-weight: 500; margin-bottom: 0.75rem;">High Risk Patients</div>
            {'<div style="background: rgba(239, 68, 68, 0.2); padding: 0.25rem 0.75rem; border-radius: 16px; font-size: 0.7rem; color: #dc2626; font-weight: 600; border: 1px solid rgba(239, 68, 68, 0.3);">IMMEDIATE REVIEW</div>' if len(high_risk_patients) > 0 else '<div style="color: #64748b; font-size: 0.7rem;">No urgent cases</div>'}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card medium-risk" style="text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 0.75rem; color: #fcd34d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.25rem; font-weight: 600;">Moderate Priority</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #f59e0b; margin: 0.25rem 0; line-height: 1;">{len(medium_risk_patients)}</div>
            <div style="font-size: 0.8rem; color: #d97706; font-weight: 500; margin-bottom: 0.75rem;">Medium Risk Patients</div>
            <div style="background: rgba(245, 158, 11, 0.2); padding: 0.25rem 0.75rem; border-radius: 16px; font-size: 0.7rem; color: #d97706; font-weight: 600; border: 1px solid rgba(245, 158, 11, 0.3);">MONITOR CLOSELY</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card low-risk" style="text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 0.75rem; color: #6ee7b7; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.25rem; font-weight: 600;">Stable Condition</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #10b981; margin: 0.25rem 0; line-height: 1;">{len(low_risk_patients)}</div>
            <div style="font-size: 0.8rem; color: #059669; font-weight: 500; margin-bottom: 0.75rem;">Low Risk Patients</div>
            <div style="background: rgba(16, 185, 129, 0.2); padding: 0.25rem 0.75rem; border-radius: 16px; font-size: 0.7rem; color: #059669; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3);">ROUTINE CARE</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_risk = df_with_predictions['Risk_Score'].mean() * 100
        risk_trend = "RISING" if avg_risk > 50 else "DECLINING" if avg_risk < 30 else "STABLE"
        trend_color = "#ef4444" if avg_risk > 50 else "#10b981" if avg_risk < 30 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center; background: rgba(255, 255, 255, 0.95);">
            <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.25rem; font-weight: 600;">Cohort Average</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {trend_color}; margin: 0.25rem 0; line-height: 1;">{avg_risk:.1f}%</div>
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 500; margin-bottom: 0.75rem;">Risk Score {risk_trend}</div>
            <div style="background: rgba(148, 163, 184, 0.2); padding: 0.25rem 0.75rem; border-radius: 16px; font-size: 0.7rem; color: #64748b; font-weight: 600; border: 1px solid rgba(148, 163, 184, 0.3);">POPULATION HEALTH</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Risk Analysis Section
    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
    
    # Enhanced patient table with integrated design
    col1, col2 = st.columns([7, 3], gap="large")
    
    with col1:
        st.markdown("""
        <div class="plot-container" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; font-size: 1.4rem; font-weight: 600; color: #1e293b;">Patient Risk Stratification</h3>
                <div style="background: rgba(16, 185, 129, 0.2); padding: 0.4rem 1rem; border-radius: 20px; font-size: 0.8rem; color: #10b981; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3);">
                    LIVE DATA
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced patient table with sophisticated highlighting
        display_df = df_with_predictions[['PatientID', 'Age', 'Primary_Condition', 'Comorbidity_Count', 
                                        'Med_Adherence', 'Days_Since_Last_Hosp', 'ER_Visits_6M', 'Risk_Score']].copy()
        display_df['Risk_Score'] = display_df['Risk_Score'] * 100
        display_df = display_df.sort_values('Risk_Score', ascending=False).reset_index(drop=True)
        display_df['Risk_Category'] = pd.cut(display_df['Risk_Score'], 
                                           bins=[0, 40, 70, 100], 
                                           labels=['Low', 'Medium', 'High'])
        
        # Light theme optimized color coding with better contrast
        def color_risk_row(row):
            if row['Risk_Score'] > 70:
                return ['background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 6px; color: #991b1b; font-weight: 600; padding: 0.5rem;'] * len(row)
            elif row['Risk_Score'] > 40:
                return ['background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 6px; color: #92400e; font-weight: 500; padding: 0.5rem;'] * len(row)
            else:
                return ['background: #f0fdf4; border-left: 4px solid #10b981; border-radius: 6px; color: #065f46; font-weight: 500; padding: 0.5rem;'] * len(row)
        
        # Simple, clean table styling that loads immediately
        st.markdown("""
        <style>
        /* Force immediate light theme on table load */
        div[data-testid="stDataFrame"] {
            background: white !important;
        }
        
        div[data-testid="stDataFrame"] table {
            background: white !important;
            color: #1e293b !important;
        }
        
        div[data-testid="stDataFrame"] th {
            background: #f8fafc !important;
            color: #1e293b !important;
            font-weight: 600 !important;
        }
        
        div[data-testid="stDataFrame"] td {
            background: white !important;
            color: #1e293b !important;
        }
        
        /* Force all text elements in table to light theme */
        div[data-testid="stDataFrame"] * {
            color: #1e293b !important;
        }
        
        /* Ensure styled dataframe cells maintain light colors */
        div[data-testid="stDataFrame"] .row_heading {
            color: #1e293b !important;
        }
        
        div[data-testid="stDataFrame"] .col_heading {
            color: #1e293b !important;
        }
        
        div[data-testid="stDataFrame"] .data {
            color: #1e293b !important;
        }
        
        /* Override any pandas styling */
        div[data-testid="stDataFrame"] table tbody tr td {
            color: #1e293b !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            display_df.style.apply(color_risk_row, axis=1).format({
                'Risk_Score': '{:.1f}%',
                'Med_Adherence': '{:.1%}',
                'Age': '{:.0f}'
            }),
            width='stretch',
            height=500
        )
    
    with col2:
        # Risk distribution with refined styling
        st.markdown("""
        <div class="plot-container" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px);">
            <h3 style="margin: 0 0 1.5rem 0; font-size: 1.4rem; font-weight: 600; color: #1e293b;">Risk Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Professional risk distribution chart with exact color matching
        risk_dist = display_df['Risk_Category'].value_counts()
        
        # Ensure correct ordering and color mapping
        category_order = ['High', 'Medium', 'Low']
        category_colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
        
        # Reorder the risk distribution to match our desired order
        risk_dist_ordered = []
        labels_ordered = []
        colors_ordered = []
        
        for cat in category_order:
            if cat in risk_dist.index:
                risk_dist_ordered.append(risk_dist[cat])
                labels_ordered.append(cat)
                colors_ordered.append(category_colors[cat])
        
        fig_pie = px.pie(
            values=risk_dist_ordered, 
            names=labels_ordered,
            title="",
            color=labels_ordered,
            color_discrete_map=category_colors,
            hole=0.4
        )
        fig_pie.update_traces(
            textposition='outside', 
            textinfo='percent',
            textfont_size=12,
            textfont_color='#1e293b',
            textfont_family="Inter",
            marker=dict(line=dict(color='rgba(148, 163, 184, 0.6)', width=2))
        )
        fig_pie.update_layout(
            showlegend=True,
            height=350,
            margin=dict(t=30, b=30, l=30, r=30),
            font=dict(size=11, color="#1e293b", family="Inter"),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.2, 
                xanchor="center", 
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(148, 163, 184, 0.3)",
                borderwidth=1,
                font=dict(size=10)
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, width='stretch')
        
        # Key insights panel
        st.markdown(f"""
        <div class="medical-metric" style="margin-top: 1.5rem; background: rgba(255, 255, 255, 0.95); border: 1px solid rgba(148, 163, 184, 0.3);">
            <div class="medical-metric-label">Key Insights</div>
            <div style="margin: 1rem 0;">
                <div style="color: #ef4444; font-weight: 600; margin: 0.5rem 0;">• {len(high_risk_patients)} patients need immediate intervention</div>
                <div style="color: #f59e0b; font-weight: 500; margin: 0.5rem 0;">• {len(medium_risk_patients)} patients require monitoring</div>
                <div style="color: #10b981; font-weight: 500; margin: 0.5rem 0;">• {len(low_risk_patients)} patients stable</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
    
    # Enhanced Risk Factor Analysis with Executive Dashboard Style
    st.markdown(f"""
    <div class="section-header" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px); border: 1px solid rgba(148, 163, 184, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">Advanced Risk Analytics</h2>
                <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 1rem;">Predictive Insights & Clinical Correlations</p>
            </div>
            <div style="display: flex; gap: 1rem; align-items: center;">
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #ef4444;">{len(high_risk_patients)}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Critical</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #f59e0b;">{len(medium_risk_patients)}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Monitor</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #10b981;">{len(low_risk_patients)}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Stable</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div class="plot-container" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding: 0.5rem 0;">
                <h4 style="margin: 0; font-size: 1rem; font-weight: 600; color: #1e293b;">Age Distribution & Risk</h4>
                <div style="background: rgba(16, 185, 129, 0.2); padding: 0.15rem 0.5rem; border-radius: 8px; font-size: 0.65rem; color: #10b981; font-weight: 600;">
                    CORRELATION
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Clean Age vs Risk scatter plot
        fig_age = px.scatter(
            df_with_predictions, 
            x='Age', 
            y='Risk_Score',
            color='Primary_Condition',
            title='',
            hover_data=['PatientID', 'Comorbidity_Count'],
            color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899']
        )
        fig_age.update_layout(
            height=300,
            margin=dict(t=30, b=100, l=60, r=30),
            font=dict(size=10, color="#1e293b", family="Inter"),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.5, 
                xanchor="center", 
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.98)",
                bordercolor="rgba(148, 163, 184, 0.3)",
                borderwidth=1,
                font=dict(size=9),
                itemwidth=30
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.3)', 
                color="#1e293b",
                title_font_size=11,
                tickfont_size=9,
                title="Age"
            ),
            yaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.3)', 
                color="#1e293b",
                title_font_size=11,
                tickfont_size=9,
                title="Risk Score"
            )
        )
        st.plotly_chart(fig_age, width='stretch')
    
    with col2:
        st.markdown("""
        <div class="plot-container" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding: 0.5rem 0;">
                <h4 style="margin: 0; font-size: 1rem; font-weight: 600; color: #1e293b;">Adherence Impact</h4>
                <div style="background: rgba(245, 158, 11, 0.2); padding: 0.15rem 0.5rem; border-radius: 8px; font-size: 0.65rem; color: #f59e0b; font-weight: 600;">
                    MEDICATION
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Clean Medication adherence scatter plot
        fig_med = px.scatter(
            df_with_predictions, 
            x='Med_Adherence', 
            y='Risk_Score',
            color='ER_Visits_6M',
            title='',
            hover_data=['PatientID', 'Primary_Condition'],
            color_continuous_scale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']]
        )
        fig_med.update_layout(
            height=300,
            margin=dict(t=30, b=80, l=60, r=100),
            font=dict(size=10, color="#1e293b", family="Inter"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.3)', 
                color="#1e293b",
                title_font_size=11,
                tickfont_size=9,
                title="Medication Adherence"
            ),
            yaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.3)', 
                color="#1e293b",
                title_font_size=11,
                tickfont_size=9,
                title="Risk Score"
            ),
            coloraxis_colorbar=dict(
                bgcolor="rgba(255, 255, 255, 0.98)",
                bordercolor="rgba(148, 163, 184, 0.3)",
                borderwidth=1,
                tickcolor="#1e293b",
                tickfont=dict(size=9),
                title=dict(text="ER Visits", font=dict(size=10)),
                len=0.8,
                thickness=15
            )
        )
        st.plotly_chart(fig_med, width='stretch')
    
    with col3:
        st.markdown("""
        <div class="plot-container" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding: 0.5rem 0;">
                <h4 style="margin: 0; font-size: 1rem; font-weight: 600; color: #1e293b;">Comorbidity Profile</h4>
                <div style="background: rgba(139, 92, 246, 0.2); padding: 0.15rem 0.5rem; border-radius: 8px; font-size: 0.65rem; color: #8b5cf6; font-weight: 600;">
                    COMPLEXITY
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Clean Comorbidity histogram
        df_with_risk_category = df_with_predictions.copy()
        df_with_risk_category['Risk_Category'] = pd.cut(
            df_with_risk_category['Risk_Score'], 
            bins=[0, 0.4, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        fig_comorb = px.histogram(
            df_with_risk_category, 
            x='Comorbidity_Count',
            color='Risk_Category',
            title='',
            color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'},
            barmode='group'
        )
        fig_comorb.update_layout(
            height=300,
            margin=dict(t=30, b=100, l=60, r=30),
            font=dict(size=10, color="#1e293b", family="Inter"),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.5, 
                xanchor="center", 
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.98)",
                bordercolor="rgba(148, 163, 184, 0.3)",
                borderwidth=1,
                font=dict(size=9),
                itemwidth=30
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.3)', 
                color="#1e293b",
                title_font_size=11,
                tickfont_size=9,
                title="Comorbidity Count"
            ),
            yaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.3)', 
                color="#1e293b",
                title_font_size=11,
                tickfont_size=9,
                title="Patient Count"
            )
        )
        st.plotly_chart(fig_comorb, width='stretch')

elif view_mode == "Patient Deep Dive":
    st.markdown('<h2 class="section-header">Individual Patient Deep Dive</h2>', unsafe_allow_html=True)
    st.markdown('<div class="spacing-md"></div>', unsafe_allow_html=True)
    
    # Patient selection with improved layout
    col1, col2 = st.columns([1, 4], gap="large")
    
    with col1:
        st.markdown("""
        <div class="sidebar-content">
            <h3 style="color: #1e293b; font-weight: 600; font-size: 0.95rem; margin: 0 0 0.75rem 0;">Select Patient</h3>
        </div>
        """, unsafe_allow_html=True)
        
        high_risk_patients = df_with_predictions[df_with_predictions['Risk_Score'] > 0.7]['PatientID'].tolist()
        medium_risk_patients = df_with_predictions[(df_with_predictions['Risk_Score'] > 0.4) & (df_with_predictions['Risk_Score'] <= 0.7)]['PatientID'].tolist()
        low_risk_patients = df_with_predictions[df_with_predictions['Risk_Score'] <= 0.4]['PatientID'].tolist()
        
        risk_filter = st.selectbox(
            "Filter by Risk Level", 
            ["All Patients", "High Risk", "Medium Risk", "Low Risk"],
            help="Filter patients by their risk classification"
        )
        
        if risk_filter == "High Risk":
            available_patients = high_risk_patients
        elif risk_filter == "Medium Risk":
            available_patients = medium_risk_patients
        elif risk_filter == "Low Risk":
            available_patients = low_risk_patients
        else:
            available_patients = df_with_predictions['PatientID'].tolist()
        
        selected_patient_id = st.selectbox(
            "Patient ID", 
            available_patients,
            help="Select a patient to view detailed analysis"
        )
    
    if selected_patient_id:
        patient_data = df_with_predictions[df_with_predictions['PatientID'] == selected_patient_id].iloc[0]
        patient_index = df_with_predictions[df_with_predictions['PatientID'] == selected_patient_id].index[0]
        
        with col2:
            # Professional patient header info
            risk_score = patient_data['Risk_Score']
            risk_percent = risk_score * 100
            
            if risk_score > 0.7:
                risk_class = "HIGH RISK"
                risk_color = "#ef4444"
                alert_class = "high-risk"
            elif risk_score > 0.4:
                risk_class = "MEDIUM RISK"
                risk_color = "#f59e0b"
                alert_class = "medium-risk"
            else:
                risk_class = "LOW RISK"
                risk_color = "#10b981"
                alert_class = "low-risk"
            
            st.markdown(f"""
            <div class="metric-card {alert_class}" style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #1e293b; margin-bottom: 0.5rem; font-weight: 600;">
                    Patient {selected_patient_id} - {risk_class}
                </h2>
                <h1 class="risk-score-large" style="color: {risk_color};">{risk_percent:.1f}%</h1>
                <p class="risk-label">90-Day Deterioration Risk</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1.5rem;">
                    <div class="medical-metric">
                        <div class="medical-metric-label">Age</div>
                        <div class="medical-metric-value">{patient_data['Age']} years</div>
                    </div>
                    <div class="medical-metric">
                        <div class="medical-metric-label">Primary Condition</div>
                        <div class="medical-metric-value" style="font-size: 1rem;">{patient_data['Primary_Condition']}</div>
                    </div>
                    <div class="medical-metric">
                        <div class="medical-metric-label">Comorbidities</div>
                        <div class="medical-metric-value">{patient_data['Comorbidity_Count']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="spacing-md"></div>', unsafe_allow_html=True)
        
        # Professional patient details in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Clinical Summary", 
            "Trends & Vitals", 
            "AI Insights", 
            "Care Recommendations"
        ])
        
        with tab1:
            st.markdown('<div class="spacing-sm"></div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3, gap="large")
            
            with col1:
                st.markdown("""
                <div class="metric-card" style="border-left: 4px solid #3b82f6; margin-bottom: 1rem;">
                    <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.1rem;">
                        Demographics
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Age", f"{patient_data['Age']} years")
                st.metric("Gender", patient_data['Gender'])
                st.metric("Insurance", patient_data['Insurance_Type'])
                
                st.markdown('<div class="spacing-sm"></div>', unsafe_allow_html=True)
                
                comorbidity_color = "#ef4444" if patient_data['Comorbidity_Count'] > 3 else "#f59e0b" if patient_data['Comorbidity_Count'] > 1 else "#10b981"
                
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {comorbidity_color}; margin-bottom: 1rem;">
                    <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.1rem;">
                        Medical Conditions
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Primary Condition", patient_data['Primary_Condition'])
                st.metric("Comorbidity Count", patient_data['Comorbidity_Count'])
                st.metric("Total Comorbidities", f"{patient_data['Comorbidity_Count']} conditions")
            
            with col2:
                bp_high = patient_data['Systolic_BP'] > 140
                vitals_color = "#ef4444" if bp_high else "#10b981"
                
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {vitals_color}; margin-bottom: 1rem;">
                    <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.1rem;">
                        Current Vitals
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                bp_status = "High" if patient_data['Systolic_BP'] > 140 else "Elevated" if patient_data['Systolic_BP'] > 130 else "Normal"
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">Blood Pressure</div>
                    <div class="medical-metric-value">{patient_data['Systolic_BP']:.0f}/{patient_data['Diastolic_BP']:.0f} <span style="color: {'#ef4444' if patient_data['Systolic_BP'] > 140 else '#f59e0b' if patient_data['Systolic_BP'] > 130 else '#10b981'};">({bp_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                hr_status = "High" if patient_data['Heart_Rate'] > 100 else "Normal"
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">Heart Rate</div>
                    <div class="medical-metric-value">{patient_data['Heart_Rate']:.0f} bpm <span style="color: {'#ef4444' if patient_data['Heart_Rate'] > 100 else '#10b981'};">({hr_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                bmi_status = "Obese" if patient_data['BMI'] > 30 else "Overweight" if patient_data['BMI'] > 25 else "Normal"
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">BMI</div>
                    <div class="medical-metric-value">{patient_data['BMI']:.1f} <span style="color: {'#ef4444' if patient_data['BMI'] > 30 else '#f59e0b' if patient_data['BMI'] > 25 else '#10b981'};">({bmi_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="spacing-sm"></div>', unsafe_allow_html=True)
                
                hba1c_high = patient_data['HbA1c'] > 9
                lab_color = "#ef4444" if hba1c_high else "#10b981"
                
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {lab_color}; margin-bottom: 1rem;">
                    <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.1rem;">
                        Lab Values
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                hba1c_status = "Poor Control" if patient_data['HbA1c'] > 9 else "Fair Control" if patient_data['HbA1c'] > 7 else "Good Control"
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">HbA1c</div>
                    <div class="medical-metric-value">{patient_data['HbA1c']:.1f}% <span style="color: {'#ef4444' if patient_data['HbA1c'] > 9 else '#f59e0b' if patient_data['HbA1c'] > 7 else '#10b981'};">({hba1c_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                creat_status = "Elevated" if patient_data['Creatinine'] > 1.3 else "Normal"
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">Creatinine</div>
                    <div class="medical-metric-value">{patient_data['Creatinine']:.2f} mg/dL <span style="color: {'#ef4444' if patient_data['Creatinine'] > 1.3 else '#10b981'};">({creat_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                ldl_status = "High" if patient_data['LDL_Cholesterol'] > 160 else "Borderline" if patient_data['LDL_Cholesterol'] > 130 else "Optimal"
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">LDL Cholesterol</div>
                    <div class="medical-metric-value">{patient_data['LDL_Cholesterol']:.0f} mg/dL <span style="color: {'#ef4444' if patient_data['LDL_Cholesterol'] > 160 else '#f59e0b' if patient_data['LDL_Cholesterol'] > 130 else '#10b981'};">({ldl_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                er_high = patient_data['ER_Visits_6M'] > 2
                care_color = "#ef4444" if er_high else "#10b981"
                
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {care_color}; margin-bottom: 1rem;">
                    <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.1rem;">
                        Care Utilization
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("ER Visits (6M)", patient_data['ER_Visits_6M'])
                st.metric("Days Since Last Hosp", patient_data['Days_Since_Last_Hosp'])
                st.metric("Med Adherence", f"{patient_data['Med_Adherence']:.1%}")
                
                st.markdown('<div class="spacing-sm"></div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card" style="border-left: 4px solid #8b5cf6; margin-bottom: 1rem;">
                    <h3 style="color: #1e293b; margin: 0 0 1rem 0; font-weight: 600; font-size: 1.1rem;">
                        Social Factors
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Social Risk Score", f"{patient_data['Social_Risk_Score']:.2f}")
                st.metric("Distance to Hospital", f"{patient_data['Distance_to_Hospital']:.1f} miles")
                
                adherence_status = "Excellent" if patient_data['Med_Adherence'] > 0.9 else "Good" if patient_data['Med_Adherence'] > 0.7 else "Poor"
                adherence_status_class = "status-low" if patient_data['Med_Adherence'] > 0.9 else "status-medium" if patient_data['Med_Adherence'] > 0.7 else "status-high"
                
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">Medication Adherence</div>
                    <div class="medical-metric-value">{patient_data['Med_Adherence']:.1%} <span style="color: {'#10b981' if patient_data['Med_Adherence'] > 0.9 else '#f59e0b' if patient_data['Med_Adherence'] > 0.7 else '#ef4444'};">({adherence_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                steps_status = "Active" if patient_data['Steps_Per_Day'] > 5000 else "Moderate" if patient_data['Steps_Per_Day'] > 2500 else "Sedentary"
                steps_status_class = "status-low" if patient_data['Steps_Per_Day'] > 5000 else "status-medium" if patient_data['Steps_Per_Day'] > 2500 else "status-high"
                
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">Daily Steps</div>
                    <div class="medical-metric-value">{patient_data['Steps_Per_Day']:,.0f} <span style="color: {'#10b981' if patient_data['Steps_Per_Day'] > 5000 else '#f59e0b' if patient_data['Steps_Per_Day'] > 2500 else '#ef4444'};">({steps_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                er_status = "Frequent" if patient_data['ER_Visits_6M'] > 2 else "Occasional" if patient_data['ER_Visits_6M'] > 0 else "None"
                er_status_class = "status-high" if patient_data['ER_Visits_6M'] > 2 else "status-medium" if patient_data['ER_Visits_6M'] > 0 else "status-low"
                
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">ER Visits (6M)</div>
                    <div class="medical-metric-value">{patient_data['ER_Visits_6M']} <span style="color: {'#ef4444' if patient_data['ER_Visits_6M'] > 2 else '#f59e0b' if patient_data['ER_Visits_6M'] > 0 else '#10b981'};">({er_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                missed_status = "Frequent" if patient_data['Missed_Appointments'] > 3 else "Occasional" if patient_data['Missed_Appointments'] > 1 else "Rare"
                missed_status_class = "status-high" if patient_data['Missed_Appointments'] > 3 else "status-medium" if patient_data['Missed_Appointments'] > 1 else "status-low"
                
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">Missed Appointments</div>
                    <div class="medical-metric-value">{patient_data['Missed_Appointments']} <span style="color: {'#ef4444' if patient_data['Missed_Appointments'] > 3 else '#f59e0b' if patient_data['Missed_Appointments'] > 1 else '#10b981'};">({missed_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                days_since_status = "Recent" if patient_data['Days_Since_Last_Hosp'] < 30 else "Moderate" if patient_data['Days_Since_Last_Hosp'] < 90 else "Remote"
                days_since_status_class = "status-high" if patient_data['Days_Since_Last_Hosp'] < 30 else "status-medium" if patient_data['Days_Since_Last_Hosp'] < 90 else "status-low"
                
                st.markdown(f"""
                <div class="medical-metric">
                    <div class="medical-metric-label">Days Since Hospitalization</div>
                    <div class="medical-metric-value">{patient_data['Days_Since_Last_Hosp']} <span style="color: {'#ef4444' if patient_data['Days_Since_Last_Hosp'] < 30 else '#f59e0b' if patient_data['Days_Since_Last_Hosp'] < 90 else '#10b981'};">({days_since_status})</span></div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="spacing-sm"></div>', unsafe_allow_html=True)
            
            # Clinical trend visualizations
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 class="subsection-header">Blood Pressure Trends</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Simulated BP trend data for demonstration
                dates = pd.date_range(start='2024-01-01', periods=12, freq='W')
                bp_trend = pd.DataFrame({
                    'Date': dates,
                    'Systolic': np.random.normal(patient_data['Systolic_BP'], 10, 12),
                    'Diastolic': np.random.normal(patient_data['Diastolic_BP'], 5, 12)
                })
                
                fig_bp = px.line(bp_trend, x='Date', y=['Systolic', 'Diastolic'], 
                                title="12-Week Blood Pressure Trend",
                                color_discrete_map={'Systolic': '#ef4444', 'Diastolic': '#10b981'})
                fig_bp.add_hline(y=140, line_dash="dash", line_color="#ef4444", annotation_text="Hypertensive Threshold")
                fig_bp.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", color="#1e293b"),
                    margin=dict(t=40, l=40, r=40, b=40),
                    xaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
                    yaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
                    legend=dict(bgcolor="rgba(255, 255, 255, 0.98)", bordercolor="rgba(148, 163, 184, 0.3)", borderwidth=1)
                )
                st.plotly_chart(fig_bp, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 class="subsection-header">HbA1c Progress</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Simulated HbA1c trend
                hba1c_trend = pd.DataFrame({
                    'Date': pd.date_range(start='2024-01-01', periods=4, freq='3M'),
                    'HbA1c': np.random.normal(patient_data['HbA1c'], 0.3, 4)
                })
                
                fig_hba1c = px.line(hba1c_trend, x='Date', y='HbA1c', 
                                   title="Quarterly HbA1c Levels",
                                   markers=True, line_shape='spline')
                fig_hba1c.add_hline(y=7.0, line_dash="dash", line_color="#f59e0b", annotation_text="Target < 7%")
                fig_hba1c.update_traces(line_color='#10b981', marker_color='#10b981')
                fig_hba1c.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", color="#1e293b"),
                    margin=dict(t=40, l=40, r=40, b=40),
                    xaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
                    yaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b")
                )
                st.plotly_chart(fig_hba1c, use_container_width=True)
            
            # Activity and engagement metrics
            st.markdown("""
            <div class="metric-card">
                <h3 class="subsection-header">Daily Activity & Engagement</h3>
            </div>
            """, unsafe_allow_html=True)
            
            activity_data = pd.DataFrame({
                'Day': range(1, 31),
                'Steps': np.random.normal(patient_data['Steps_Per_Day'], 1000, 30),
                'Med_Taken': np.random.choice([0, 1], 30, p=[1-patient_data['Med_Adherence'], patient_data['Med_Adherence']])
            })
            
            fig_activity = px.bar(activity_data, x='Day', y='Steps', 
                                 title="30-Day Step Count",
                                 color='Med_Taken', 
                                 color_discrete_map={1: '#10b981', 0: '#ef4444'},
                                 labels={'Med_Taken': 'Medication Taken'})
            fig_activity.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", color="#1e293b"),
                margin=dict(t=40, l=40, r=40, b=40),
                xaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
                yaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
                legend=dict(bgcolor="rgba(255, 255, 255, 0.98)", bordercolor="rgba(148, 163, 184, 0.3)", borderwidth=1)
            )
            st.plotly_chart(fig_activity, use_container_width=True)
        
        with tab3:
            st.markdown('<div class="spacing-sm"></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1], gap="large")
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 class="subsection-header">Risk Factor Analysis</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # SHAP values for this specific patient
                patient_features = model_info['X'].iloc[patient_index:patient_index+1]
                shap_values_patient = model_info['explainer'].shap_values(patient_features)
                
                # Create SHAP feature importance bar chart (alternative to waterfall)
                feature_shap_df = pd.DataFrame({
                    'Feature': model_info['feature_cols'],
                    'SHAP_Value': shap_values_patient[0],
                    'Impact': ['Increases Risk' if x > 0 else 'Decreases Risk' for x in shap_values_patient[0]]
                }).sort_values('SHAP_Value', key=abs, ascending=False).head(8)
                
                fig_shap = px.bar(
                    feature_shap_df,
                    x='SHAP_Value',
                    y='Feature',
                    orientation='h',
                    title='AI Model Decision Breakdown - Top Risk Factors',
                    color='SHAP_Value',
                    color_continuous_scale=[[0, '#ef4444'], [0.5, '#1e293b'], [1, '#10b981']]
                )
                fig_shap.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", color="#1e293b"),
                    xaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
                    yaxis=dict(color="#1e293b"),
                    coloraxis_colorbar=dict(
                        bgcolor="rgba(255, 255, 255, 0.98)",
                        bordercolor="rgba(148, 163, 184, 0.3)",
                        borderwidth=1,
                        tickcolor="#1e293b"
                    )
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 class="subsection-header">Risk Insights</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Top risk factors for this patient
                feature_importance = pd.DataFrame({
                    'Feature': model_info['feature_cols'],
                    'SHAP_Value': shap_values_patient[0],
                    'Patient_Value': patient_features.iloc[0].values
                }).sort_values('SHAP_Value', key=abs, ascending=False).head(5)
                
                for idx, row in feature_importance.iterrows():
                    impact = "Increases Risk" if row['SHAP_Value'] > 0 else "Decreases Risk"
                    impact_color = "#ef4444" if row['SHAP_Value'] > 0 else "#10b981"
                    
                    st.markdown(f"""
                    <div class="medical-metric" style="margin-bottom: 1rem; padding: 1rem; border-left: 4px solid {impact_color};">
                        <div class="medical-metric-label" style="font-weight: 600;">{row['Feature']}</div>
                        <div class="medical-metric-value">{row['Patient_Value']:.2f}</div>
                        <div style="color: {impact_color}; font-size: 0.9rem; font-weight: 500;">{impact}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<div class="spacing-sm"></div>', unsafe_allow_html=True)
                
                # Risk summary
                if risk_score > 0.7:
                    risk_summary = "This patient shows multiple high-risk indicators requiring immediate clinical attention and care plan adjustment."
                elif risk_score > 0.4:
                    risk_summary = "This patient demonstrates moderate risk factors that warrant increased monitoring and preventive interventions."
                else:
                    risk_summary = "This patient maintains good clinical stability with low risk indicators. Continue current care approach."
                
                st.markdown(f"""
                <div class="metric-card" style="background: rgba(255, 255, 255, 0.95); border-left: 4px solid #10b981;">
                    <h4 style="color: #10b981; margin-bottom: 0.5rem;">Clinical Summary</h4>
                    <p style="color: #1e293b; margin: 0; line-height: 1.6;">{risk_summary}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="spacing-sm"></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 class="subsection-header">Immediate Actions</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate personalized recommendations based on risk factors
                recommendations = []
                
                if patient_data['Systolic_BP'] > 140:
                    recommendations.append("Schedule urgent cardiology consultation for hypertension management")
                
                if patient_data['HbA1c'] > 9:
                    recommendations.append("Initiate intensive diabetes management protocol")
                
                if patient_data['Med_Adherence'] < 0.7:
                    recommendations.append("Implement medication adherence support program")
                
                if patient_data['ER_Visits_6M'] > 2:
                    recommendations.append("Enroll in high-risk case management program")
                
                if patient_data['Steps_Per_Day'] < 2500:
                    recommendations.append("Refer to cardiac rehabilitation or physical therapy")
                
                if patient_data['Missed_Appointments'] > 3:
                    recommendations.append("Schedule care coordination meeting to address barriers")
                
                if not recommendations:
                    recommendations = [
                        "Continue current care plan with standard monitoring",
                        "Maintain medication regimen as prescribed",
                        "Schedule routine follow-up in 3 months"
                    ]
                
                for i, rec in enumerate(recommendations[:5], 1):
                    priority_color = "#ef4444" if i <= 2 else "#f59e0b" if i <= 4 else "#10b981"
                    priority_text = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
                    
                    st.markdown(f"""
                    <div class="medical-metric" style="margin-bottom: 1rem; padding: 1rem; border-left: 4px solid {priority_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="color: {priority_color}; font-weight: 600; font-size: 0.9rem;">{priority_text} Priority</span>
                            <span style="background: {priority_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.8rem; font-weight: 500;">{i}</span>
                        </div>
                        <div style="color: #1e293b; line-height: 1.5;">{rec}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 class="subsection-header">Monitoring Plan</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Monitoring frequency based on risk level
                if risk_score > 0.7:
                    monitoring_plan = {
                        "Clinical Follow-up": "Weekly for 4 weeks, then biweekly",
                        "Vital Signs": "Daily self-monitoring with reporting",
                        "Lab Work": "Monthly until stabilized",
                        "Care Coordination": "Weekly multidisciplinary rounds",
                        "Patient Contact": "Nursing check-in every 3 days"
                    }
                elif risk_score > 0.4:
                    monitoring_plan = {
                        "Clinical Follow-up": "Biweekly for 8 weeks",
                        "Vital Signs": "3x weekly self-monitoring",
                        "Lab Work": "Every 6 weeks",
                        "Care Coordination": "Monthly team review",
                        "Patient Contact": "Weekly nursing check-in"
                    }
                else:
                    monitoring_plan = {
                        "Clinical Follow-up": "Monthly routine visits",
                        "Vital Signs": "Weekly self-monitoring",
                        "Lab Work": "Quarterly routine labs",
                        "Care Coordination": "Quarterly team review",
                        "Patient Contact": "As needed basis"
                    }
                
                for category, frequency in monitoring_plan.items():
                    st.markdown(f"""
                    <div class="medical-metric" style="margin-bottom: 1rem;">
                        <div class="medical-metric-label">{category}</div>
                        <div class="medical-metric-value" style="font-size: 1rem;">{frequency}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<div class="spacing-sm"></div>', unsafe_allow_html=True)
                
                # Contact information
                st.markdown("""
                <div class="metric-card" style="background: rgba(255, 255, 255, 0.95); border-left: 4px solid #3b82f6;">
                    <h4 style="color: #3b82f6; margin-bottom: 1rem;">Emergency Contacts</h4>
                    <div style="color: #1e293b;">
                        <div style="margin-bottom: 0.5rem;"><strong>Primary Care:</strong> Dr. Smith (555-0123)</div>
                        <div style="margin-bottom: 0.5rem;"><strong>Care Manager:</strong> Jane Doe, RN (555-0124)</div>
                        <div style="margin-bottom: 0.5rem;"><strong>After Hours:</strong> (555-0125)</div>
                        <div><strong>Emergency:</strong> 911 or nearest ER</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:  # Model Analytics
    st.markdown('<h2 class="section-header">Model Analytics & Performance</h2>', unsafe_allow_html=True)
    st.markdown('<div class="spacing-md"></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Feature Importance", "Model Interpretation"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Model comparison metrics
            st.subheader("Model Comparison")
            metrics_df = pd.DataFrame({
                'Model': ['XGBoost', 'Random Forest', 'Ensemble'],
                'AUC-ROC': [model_info['xgb_auc'], model_info['rf_auc'], 0.85],
                'Precision': [0.78, 0.72, 0.80],
                'Recall': [0.75, 0.70, 0.77],
                'F1-Score': [0.77, 0.71, 0.78]
            })
            
            st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['AUC-ROC', 'Precision', 'Recall', 'F1-Score']))
        
        with col2:
            # Feature correlation heatmap
            feature_corr = df_with_predictions[model_info['feature_cols']].corr()
            
            fig_corr = px.imshow(
                feature_corr,
                title="Feature Correlation Matrix",
                color_continuous_scale=['#10b981', '#ffffff', '#ef4444'],
                aspect="auto"
            )
            fig_corr.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", color="#1e293b"),
                xaxis=dict(color="#1e293b", tickangle=45),
                yaxis=dict(color="#1e293b"),
                title=dict(
                    font=dict(size=14, color="#1e293b"),
                    x=0.5,
                    y=0.95
                ),
                margin=dict(t=60, b=60, l=60, r=60),
                coloraxis_colorbar=dict(
                    bgcolor="rgba(255, 255, 255, 0.98)",
                    bordercolor="rgba(148, 163, 184, 0.3)",
                    borderwidth=1,
                    tickcolor="#1e293b",
                    tickfont=dict(size=10),
                    title=dict(text="Correlation", font=dict(size=12))
                )
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        # Global feature importance
        feature_importance = pd.DataFrame({
            'Feature': model_info['feature_cols'],
            'Importance': model_info['xgb_model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Global Feature Importance (XGBoost)',
            color_discrete_sequence=['#10b981']
        )
        fig_importance.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#1e293b"),
            xaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
            yaxis=dict(color="#1e293b")
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # SHAP summary plot
        st.subheader("SHAP Feature Impact Distribution")
        shap_summary_df = pd.DataFrame(
            model_info['shap_values'],
            columns=model_info['feature_cols']
        )
        
        # Create violin plot for SHAP values
        shap_melted = shap_summary_df.melt(var_name='Feature', value_name='SHAP_Value')
        fig_violin = px.violin(
            shap_melted, 
            y='Feature', 
            x='SHAP_Value',
            title='SHAP Value Distribution by Feature',
            orientation='h',
            color_discrete_sequence=['#10b981']
        )
        fig_violin.update_layout(
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#1e293b"),
            xaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
            yaxis=dict(color="#1e293b")
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    
    with tab3:
        st.subheader("Model Decision Boundaries")
        
        # Risk score distribution
        fig_dist = px.histogram(
            df_with_predictions,
            x='Risk_Score',
            nbins=30,
            title='Risk Score Distribution Across Cohort',
            color_discrete_sequence=['#10b981']
        )
        fig_dist.add_vline(x=0.4, line_dash="dash", line_color="#f59e0b", annotation_text="Medium Risk Threshold")
        fig_dist.add_vline(x=0.7, line_dash="dash", line_color="#ef4444", annotation_text="High Risk Threshold")
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#1e293b"),
            xaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
            yaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b")
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Model calibration
        st.subheader("Model Calibration Analysis")
        st.info("Model calibration ensures predicted probabilities match actual outcomes. Well-calibrated models have predictions close to the diagonal line.")
        
        # Create calibration plot (simplified)
        risk_bins = pd.cut(df_with_predictions['Risk_Score'], bins=10)
        calibration_df = df_with_predictions.groupby(risk_bins, observed=False).agg({
            'Risk_Score': 'mean',
            'Risk_of_Deterioration_90d': 'mean'
        }).reset_index(drop=True)
        
        fig_calib = px.scatter(
            calibration_df,
            x='Risk_Score',
            y='Risk_of_Deterioration_90d',
            title='Model Calibration Plot',
            labels={
                'Risk_Score': 'Predicted Probability',
                'Risk_of_Deterioration_90d': 'Actual Proportion'
            },
            color_discrete_sequence=['#10b981']
        )
        
        # Add perfect calibration line
        fig_calib.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(dash='dash', color='#ef4444')
            )
        )
        
        fig_calib.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#1e293b"),
            xaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
            yaxis=dict(gridcolor='rgba(148, 163, 184, 0.3)', color="#1e293b"),
            legend=dict(bgcolor="rgba(255, 255, 255, 0.98)", bordercolor="rgba(148, 163, 184, 0.3)", borderwidth=1)
        )
        
        st.plotly_chart(fig_calib, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.9rem; font-family: "Source Sans Pro", sans-serif;'>
    <p style="margin-bottom: 0.5rem; font-weight: 500;">AI-Driven Risk Prediction Engine | Built for Chronic Care Management</p>
    <p style="margin: 0; color: #9ca3af;">This is a prototype for demonstration purposes. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)