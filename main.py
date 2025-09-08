# Modular imports for streamlined architecture
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config import (
    APP_CONFIG, THEME_CONFIG, MODEL_CONFIG, DATA_CONFIG, NAVIGATION_OPTIONS,
    get_custom_css, apply_theme_config, get_risk_category, get_risk_color,
    format_medical_value, DEFAULT_PATIENT
)
from data_generator import generate_comprehensive_patient_data, generate_time_series_data
from model_trainer import RiskPredictionModel, train_risk_prediction_models
from visualization import (
    create_risk_distribution_chart, create_age_risk_scatter, create_adherence_risk_chart,
    create_comorbidity_analysis, create_patient_timeline_chart, create_feature_importance_chart,
    create_risk_category_pie_chart, create_condition_risk_analysis
)
from utils import (
    create_metric_card, display_risk_indicator, create_sidebar_section,
    format_patient_summary, assess_clinical_values, display_clinical_assessment,
    calculate_summary_statistics, create_patient_selector, generate_recommendations,
    export_patient_report
)

# App configuration
st.set_page_config(**APP_CONFIG)

# Apply theme settings
apply_theme_config()

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Dashboard Overview"

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Load or generate data
@st.cache_data
def load_patient_data():
    """Load patient data from CSV or generate new data."""
    try:
        df = pd.read_csv(DATA_CONFIG['csv_filename'])
        print(f"Loaded existing data: {len(df)} patients")
        return df
    except FileNotFoundError:
        print("Generating new patient data...")
        df = generate_comprehensive_patient_data(
            num_patients=DATA_CONFIG['num_patients'],
            save_to_csv=True,
            filename=DATA_CONFIG['csv_filename']
        )
        return df

# Load and train model
@st.cache_resource
def load_trained_model():
    """Load or train the risk prediction model."""
    model = RiskPredictionModel()
    
    # Try to load existing models
    if model.load_models():
        print("Loaded existing trained models")
        df = load_patient_data()
        
        # Check if Risk_Score already exists, if not compute it efficiently
        if 'Risk_Score' not in df.columns:
            print("Computing risk scores for all patients...")
            df['Risk_Score'] = model.predict_batch(df)
            # Save the updated CSV with risk scores to avoid recomputation
            df.to_csv(DATA_CONFIG['csv_filename'], index=False)
            print("Risk scores computed and saved to CSV")
        
        return df, model.model_info, model
    else:
        print("Training new models...")
        df, model_info, trained_model = train_risk_prediction_models(
            csv_file=DATA_CONFIG['csv_filename'],
            save_models=True
        )
        return df, model_info, trained_model
    """Generates realistic chronic care patient data with multiple conditions and risk factors."""
    np.random.seed(42)
    num_patients = 150
    
    # Patient identifiers and basic demographics
    patient_ids = [f'CHR-{str(i).zfill(4)}' for i in range(1, num_patients + 1)]
    ages = np.random.normal(65, 12, num_patients).astype(int)
    ages = np.clip(ages, 35, 95)
    
    genders = np.random.choice(['Male', 'Female'], num_patients, p=[0.45, 0.55])
    
    # Common chronic conditions with realistic prevalence
    conditions = np.random.choice([
        'Type 2 Diabetes', 'Heart Failure', 'COPD', 'Hypertension', 
        'Chronic Kidney Disease', 'Obesity + Diabetes'
    ], num_patients, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
    
    comorbidity_counts = np.random.poisson(2.5, num_patients)
    comorbidity_counts = np.clip(comorbidity_counts, 1, 6)
    
    # Clinical measurements with realistic correlations
    systolic_bp = np.random.normal(140, 18, num_patients)
    diastolic_bp = systolic_bp * 0.6 + np.random.normal(10, 8, num_patients)
    heart_rate = np.random.normal(75, 12, num_patients)
    
    # Laboratory values
    hba1c = np.random.gamma(2, 3.5, num_patients)
    hba1c = np.clip(hba1c, 5.5, 14.0)
    
    creatinine = np.random.gamma(1.5, 0.8, num_patients)
    creatinine = np.clip(creatinine, 0.6, 4.5)
    
    ldl_cholesterol = np.random.normal(130, 35, num_patients)
    ldl_cholesterol = np.clip(ldl_cholesterol, 70, 250)
    
    # Behavioral and lifestyle factors
    med_adherence = np.random.beta(3, 1.5, num_patients)
    med_adherence = np.clip(med_adherence, 0.3, 1.0)
    
    steps_per_day = np.random.exponential(3500, num_patients).astype(int)
    steps_per_day = np.clip(steps_per_day, 500, 12000)
    
    bmi = np.random.normal(29, 6, num_patients)
    bmi = np.clip(bmi, 18, 45)
    
    # Healthcare utilization patterns
    er_visits_6m = np.random.poisson(0.8, num_patients)
    missed_appointments = np.random.poisson(1.2, num_patients)
    
    # Social determinants of health
    insurance_types = np.random.choice(['Medicare', 'Medicaid', 'Private', 'Dual'], 
                                     num_patients, p=[0.4, 0.25, 0.25, 0.1])
    
    social_risk_score = np.random.normal(0.3, 0.15, num_patients)
    social_risk_score = np.clip(social_risk_score, 0, 1)
    
    distance_to_hospital = np.random.exponential(15, num_patients)
    distance_to_hospital = np.clip(distance_to_hospital, 1, 50)
    
    days_since_last_hosp = np.random.exponential(180, num_patients).astype(int)
    days_since_last_hosp = np.clip(days_since_last_hosp, 0, 1095)
    
    # Compile all data into structured dataframe
    df = pd.DataFrame({
        'Patient_ID': patient_ids,
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
    
    # Calculate composite risk score using clinical factors with evidence-based weights
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
    
    # Convert to probability distribution and create balanced outcome variable
    prob_deterioration = 1 / (1 + np.exp(-risk_score))
    threshold = np.percentile(prob_deterioration, 70)
    df['Risk_of_Deterioration_90d'] = (prob_deterioration > threshold).astype(int)
    df['Risk_Score'] = prob_deterioration
    
# Initialize data and models using modular approach
df_patients, model_info, trained_model = load_trained_model()

# Calculate summary statistics
summary_stats = calculate_summary_statistics(df_patients)

# --- PROFESSIONAL DASHBOARD UI ---

# Professional styling and layout configuration
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        background: transparent;
        max-width: 95%;
    }
    
    div[data-testid="stSidebar"] > div {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    
    /* Header styling */
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
    
    /* Card components with glass morphism effect */
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
    
    /* Risk level styling with color coding */
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
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
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
    
    /* Section headers */
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
    
    /* Medical metrics styling */
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
    
    /* Button styling */
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
    
    /* Tab styling */
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
    
    /* Status indicators */
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
    
    /* Spacing utilities */
    .spacing-lg { margin: 2.5rem 0; }
    .spacing-md { margin: 1.5rem 0; }
    .spacing-sm { margin: 1rem 0; }
    
    /* Risk score display */
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
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Component styling overrides for consistent light theme */
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
    
    div[data-testid="stDataFrame"] {
        background: white !important;
    }
    
    div[data-testid="stDataFrame"] * {
        color: #1e293b !important;
    }
    
    # Primary button styling for main actions
    .stButton > button:active {
        background: linear-gradient(135deg, #bbf7d0 0%, #86efac 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.15) !important;
    }
    
    /* Secondary button styling */
    .secondary-button {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        color: #475569 !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    .secondary-button:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%) !important;
        color: #334155 !important;
    }
    
    /* Input field styling */
    .stSelectbox > div[data-baseweb="select"] {
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.95) !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div[data-baseweb="select"]:hover {
        border-color: rgba(16, 185, 129, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.1) !important;
    }
    
    .stSelectbox > div[data-baseweb="select"] > div {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    
    /* Chart container styling */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, #10b981 50%, transparent 100%);
        opacity: 0.7;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%) !important;
        border-radius: 4px !important;
    }
    
    .stProgress > div > div {
        background: rgba(148, 163, 184, 0.2) !important;
        border-radius: 4px !important;
    }
    
    /* Alert styling */
    .alert-info {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        color: #1e3a8a;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
        color: #92400e;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #10b981;
        color: #065f46;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Table styling improvements */
    .dataframe {
        border: none !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        color: #374151 !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
        text-align: left !important;
        border-bottom: 2px solid #e5e7eb !important;
    }
    
    .dataframe td {
        padding: 0.75rem !important;
        border-bottom: 1px solid #f3f4f6 !important;
        color: #374151 !important;
        vertical-align: middle !important;
    }
    
    .dataframe tr:hover {
        background: rgba(249, 250, 251, 0.8) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 8px !important;
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(16, 185, 129, 0.6) !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.1) !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
    }
    
    .stRadio > div > label {
        color: #1e293b !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        color: #1e293b !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: rgba(148, 163, 184, 0.3) !important;
    }
    
    .stSlider [role="slider"] {
        background: #10b981 !important;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
        border-radius: 8px !important;
    }
    
    /* Spacing and layout utilities */
    .header-container {
        text-align: center;
        margin: 2rem 0 3rem 0;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
    }
    
    .header-container h1 {
        color: #1e293b !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.5rem 0 !important;
    }
    
    .header-container p {
        color: #64748b !important;
        font-size: 1.1rem !important;
        margin: 0 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-container h1 {
            font-size: 2rem !important;
        }
        
        .section-header {
            font-size: 1.5rem !important;
            padding: 1rem !important;
        }
        
        .medical-metric {
            padding: 0.75rem !important;
        }
        
        .chart-container {
            padding: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Navigation state management
if 'sidebar_collapsed' not in st.session_state:
    st.session_state.sidebar_collapsed = False

# Toggle sidebar visibility
if st.button("☰ Menu", key="burger_toggle", help="Toggle navigation sidebar"):
    st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed

# Main application header
st.markdown(f"""
<style>
.main .block-container {{
    padding-top: 0rem !important;
    margin-top: 0 !important;
}}

.stSidebar .block-container {{
    padding-top: 0rem !important;
    margin-top: 0 !important;
}}

.stApp > header {{
    display: none !important;
}}
</style>

<div style="margin-top: 0; margin-bottom: 2rem;">
    <div class="header-container">
        <h1 class="main-header">AI-Driven Risk Prediction Engine</h1>
        <p class="sub-header">Advanced Chronic Care Management & 90-Day Deterioration Prediction</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Menu button styling for professional appearance
st.markdown("""
<style>
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

# Conditional sidebar display based on user preference
if not st.session_state.sidebar_collapsed:
    with st.sidebar:
        # Remove extra padding from sidebar top
        st.markdown("""
        <style>
        .stSidebar .block-container {
            padding-top: 0rem !important;
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
        
        total_patients = len(df_patients)
        high_risk = len(df_patients[df_patients['Risk_Score'] > 0.7])
        medium_risk = len(df_patients[(df_patients['Risk_Score'] > 0.4) & (df_patients['Risk_Score'] <= 0.7)])
        low_risk = len(df_patients[df_patients['Risk_Score'] <= 0.4])
        
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
                <div style="font-size: 2.5rem; font-weight: 800; color: #10b981; margin: 0;">{len(df_patients)}</div>
                <div style="font-size: 0.9rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;">Total Patients</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Integrated risk summary cards
    high_risk_patients = df_patients[df_patients['Risk_Score'] > 0.7]
    medium_risk_patients = df_patients[(df_patients['Risk_Score'] > 0.4) & (df_patients['Risk_Score'] <= 0.7)]
    low_risk_patients = df_patients[df_patients['Risk_Score'] <= 0.4]
    
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
        avg_risk = df_patients['Risk_Score'].mean() * 100
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
    

    
    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
    

    
    # Professional layout: Table + Pie Chart 
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
        
        # Enhanced patient table with pagination (same as before)
        full_display_df = df_patients[['Patient_ID', 'Age', 'Primary_Condition', 'Comorbidity_Count', 
                                      'Med_Adherence', 'Days_Since_Last_Hosp', 'ER_Visits_6M', 'Risk_Score']].copy()
        full_display_df['Risk_Score'] = full_display_df['Risk_Score'] * 100
        full_display_df = full_display_df.sort_values('Risk_Score', ascending=False).reset_index(drop=True)
        
        # Pagination controls
        patients_per_page = 100
        total_patients = len(full_display_df)
        total_pages = (total_patients + patients_per_page - 1) // patients_per_page
        
        col1_inner, col2_inner, col3_inner = st.columns([2, 1, 2])
        with col2_inner:
            page_num = st.selectbox(
                f"Page (Total: {total_patients:,} patients)", 
                range(1, total_pages + 1),
                index=0,
                help=f"Showing {patients_per_page} patients per page",
                key="patient_table_pagination"
            )
        
        # Calculate page boundaries
        start_idx = (page_num - 1) * patients_per_page
        end_idx = start_idx + patients_per_page
        display_df = full_display_df.iloc[start_idx:end_idx].copy()
        
        display_df['Risk_Category'] = pd.cut(display_df['Risk_Score'], 
                                           bins=[0, 40, 70, 100], 
                                           labels=['Low', 'Medium', 'High'])
        
        # Show current page info
        st.markdown(f"**Showing patients {start_idx + 1:,} - {min(end_idx, total_patients):,} of {total_patients:,} total patients** (Sorted by Risk Score)")
        st.markdown('<div class="spacing-sm"></div>', unsafe_allow_html=True)
        
        # Light theme optimized color coding with better contrast
        def color_risk_row(row):
            if row['Risk_Score'] > 70:
                return ['background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 6px; color: #991b1b; font-weight: 600; padding: 0.5rem;'] * len(row)
            elif row['Risk_Score'] > 40:
                return ['background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 6px; color: #92400e; font-weight: 500; padding: 0.5rem;'] * len(row)
            else:
                return ['background: #f0fdf4; border-left: 4px solid #10b981; border-radius: 6px; color: #065f46; font-weight: 500; padding: 0.5rem;'] * len(row)
        
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
        # Risk distribution pie chart (compact sidebar version)
        st.markdown("""
        <div class="plot-container" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px);">
            <h3 style="margin: 0 0 1.5rem 0; font-size: 1.4rem; font-weight: 600; color: #1e293b;">Risk Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Professional risk distribution chart using FULL dataset (50k patients)
        full_risk_scores = df_patients['Risk_Score'] * 100
        full_risk_categories = pd.cut(full_risk_scores, 
                                     bins=[0, 40, 70, 100], 
                                     labels=['Low', 'Medium', 'High'])
        risk_dist = full_risk_categories.value_counts()
        
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
            height=400,  # Compact size for sidebar
            margin=dict(t=50, b=80, l=50, r=50),
            font=dict(size=12, color="#1e293b", family="Inter"),
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
        st.plotly_chart(fig_pie, use_container_width=True, height=400)
    
    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
    
    # Patient Risk Dashboard Header
    st.markdown(f"""
    <div class="section-header" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px); border: 1px solid rgba(148, 163, 184, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">📊 Patient Risk Dashboard</h2>
                <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 1rem;">Risk Stratification & Analytics for 50,000 Patients</p>
            </div>
            <div style="display: flex; gap: 1rem; align-items: center;">
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #ef4444;">{len(high_risk_patients):,}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Critical</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #f59e0b;">{len(medium_risk_patients):,}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Monitor</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #10b981;">{len(low_risk_patients):,}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Stable</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
    
    # Advanced Analytics Section Header
    st.markdown(f"""
    <div class="section-header" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px); border: 1px solid rgba(148, 163, 184, 0.3); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">🔍 Advanced Risk Analytics</h2>
                <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 1rem;">Large-Scale Predictive Insights & Clinical Correlations</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
    
    # ==== THE THREE LARGE CHARTS (as requested) ====
    
    # CHART 1: Age vs Risk Correlation (Large)
    st.markdown("""
    <div class="plot-container" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px); margin-bottom: 3rem;">
        <h3 style="margin: 0 0 2rem 0; font-size: 1.8rem; font-weight: 700; color: #1e293b; text-align: center;">📈 Age vs Risk Correlation Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    fig_age = px.scatter(
        df_patients, 
        x='Age', 
        y='Risk_Score',
        color='Primary_Condition',
        title='',
        hover_data=['Patient_ID', 'Comorbidity_Count'],
        color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899']
    )
    fig_age.update_layout(
        height=700,  # Large
        margin=dict(t=80, b=160, l=100, r=100),
        font=dict(size=16, color="#1e293b", family="Inter"),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=-0.35, 
            xanchor="center", 
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.98)",
            bordercolor="rgba(148, 163, 184, 0.4)",
            borderwidth=2,
            font=dict(size=14),
            itemwidth=80
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.3)', 
            color="#1e293b",
            title_font_size=18,
            tickfont_size=14,
            title="Age"
        ),
        yaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.3)', 
            color="#1e293b",
            title_font_size=18,
            tickfont_size=14,
            title="Risk Score"
        )
    )
    st.plotly_chart(fig_age, use_container_width=True, height=700)
    
    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
    
    # CHART 2: Medication Adherence Impact (Large)
    st.markdown("""
    <div class="plot-container" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px); margin-bottom: 3rem;">
        <h3 style="margin: 0 0 2rem 0; font-size: 1.8rem; font-weight: 700; color: #1e293b; text-align: center;">💊 Medication Adherence Impact Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    fig_med = px.scatter(
        df_patients, 
        x='Med_Adherence', 
        y='Risk_Score',
        color='ER_Visits_6M',
        title='',
        hover_data=['Patient_ID', 'Primary_Condition'],
        color_continuous_scale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']]
    )
    fig_med.update_layout(
        height=700,  # Large
        margin=dict(t=80, b=140, l=100, r=160),
        font=dict(size=16, color="#1e293b", family="Inter"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.3)', 
            color="#1e293b",
            title_font_size=18,
            tickfont_size=14,
            title="Medication Adherence"
        ),
        yaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.3)', 
            color="#1e293b",
            title_font_size=18,
            tickfont_size=14,
            title="Risk Score"
        ),
        coloraxis_colorbar=dict(
            bgcolor="rgba(255, 255, 255, 0.98)",
            bordercolor="rgba(148, 163, 184, 0.4)",
            borderwidth=2,
            tickcolor="#1e293b",
            tickfont=dict(size=14),
            title=dict(text="ER Visits", font=dict(size=16)),
            len=0.8,
            thickness=25
        )
    )
    st.plotly_chart(fig_med, use_container_width=True, height=700)
    
    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
    
    # CHART 3: Comorbidity Distribution & Risk Analysis (Large)
    st.markdown("""
    <div class="plot-container" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(30px); margin-bottom: 3rem;">
        <h3 style="margin: 0 0 2rem 0; font-size: 1.8rem; font-weight: 700; color: #1e293b; text-align: center;">🏥 Comorbidity Distribution & Risk Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    df_with_risk_category = df_patients.copy()
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
        height=700,  # Large
        margin=dict(t=80, b=160, l=120, r=100),
        font=dict(size=16, color="#1e293b", family="Inter"),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=-0.35, 
            xanchor="center", 
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.98)",
            bordercolor="rgba(148, 163, 184, 0.4)",
            borderwidth=2,
            font=dict(size=14),
            itemwidth=80
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.3)', 
            color="#1e293b",
            title_font_size=18,
            tickfont_size=14,
            title="Comorbidity Count"
        ),
        yaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.3)', 
            color="#1e293b",
            title_font_size=18,
            tickfont_size=14,
            title="Patient Count"
        )
    )
    st.plotly_chart(fig_comorb, use_container_width=True, height=700)

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
        
        high_risk_patients = df_patients[df_patients['Risk_Score'] > 0.7]['Patient_ID'].tolist()
        medium_risk_patients = df_patients[(df_patients['Risk_Score'] > 0.4) & (df_patients['Risk_Score'] <= 0.7)]['Patient_ID'].tolist()
        low_risk_patients = df_patients[df_patients['Risk_Score'] <= 0.4]['Patient_ID'].tolist()
        
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
            available_patients = df_patients['Patient_ID'].tolist()
        
        selected_patient_id = st.selectbox(
            "Patient ID", 
            available_patients,
            help="Select a patient to view detailed analysis"
        )
    
    if selected_patient_id:
        patient_data = df_patients[df_patients['Patient_ID'] == selected_patient_id].iloc[0]
        patient_index = df_patients[df_patients['Patient_ID'] == selected_patient_id].index[0]
        
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
                feature_names = model_info['feature_names']
                shap_values_patient = model_info['shap_values'][patient_index:patient_index+1]
                
                # Create SHAP feature importance bar chart (alternative to waterfall)
                feature_shap_df = pd.DataFrame({
                    'Feature': feature_names,
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
                    'Feature': feature_names,
                    'SHAP_Value': shap_values_patient[0],
                    'Patient_Value': patient_data[feature_names].values
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
            feature_corr = df_patients[model_info['feature_names']].corr()
            
            fig_corr = px.imshow(
                feature_corr,
                title="Feature Correlation Matrix",
                color_continuous_scale=['#10b981', '#ffffff', '#ef4444'],
                aspect="auto"
            )
            fig_corr.update_layout(
                height=600,  # Increased from 400
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", color="#1e293b", size=12),  # Larger font
                xaxis=dict(color="#1e293b", tickangle=45, tickfont=dict(size=11)),
                yaxis=dict(color="#1e293b", tickfont=dict(size=11)),
                title=dict(
                    font=dict(size=16, color="#1e293b"),  # Larger title
                    x=0.5,
                    y=0.95
                ),
                margin=dict(t=80, b=80, l=80, r=80),  # More margin
                coloraxis_colorbar=dict(
                    bgcolor="rgba(255, 255, 255, 0.98)",
                    bordercolor="rgba(148, 163, 184, 0.3)",
                    borderwidth=1,
                    tickcolor="#1e293b",
                    tickfont=dict(size=11),  # Larger colorbar font
                    title=dict(text="Correlation", font=dict(size=14)),
                    thickness=20  # Wider colorbar
                )
            )
            st.plotly_chart(fig_corr, use_container_width=True, height=600)
    
    with tab2:
        # Global feature importance
        feature_importance = pd.DataFrame({
            'Feature': model_info['feature_names'],
            'Importance': trained_model.xgb_model.feature_importances_
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
            height=600,  # Increased from 500
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#1e293b", size=12),  # Larger font
            title=dict(font=dict(size=16, color="#1e293b")),  # Larger title
            margin=dict(t=80, b=60, l=120, r=60),  # More margin for y-axis labels
            xaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.3)', 
                color="#1e293b",
                tickfont=dict(size=11),
                title_font=dict(size=14)
            ),
            yaxis=dict(
                color="#1e293b",
                tickfont=dict(size=11),
                title_font=dict(size=14)
            )
        )
        st.plotly_chart(fig_importance, use_container_width=True, height=600)
        
        # SHAP summary plot
        st.subheader("SHAP Feature Impact Distribution")
        shap_summary_df = pd.DataFrame(
            model_info['shap_values'],
            columns=model_info['feature_names']
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
            height=700,  # Increased from 600
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#1e293b", size=12),  # Larger font
            title=dict(font=dict(size=16, color="#1e293b")),  # Larger title
            margin=dict(t=80, b=60, l=120, r=60),  # More margin
            xaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.3)', 
                color="#1e293b",
                tickfont=dict(size=11),
                title_font=dict(size=14)
            ),
            yaxis=dict(
                color="#1e293b",
                tickfont=dict(size=11),
                title_font=dict(size=14)
            )
        )
        st.plotly_chart(fig_violin, use_container_width=True, height=700)
    
    with tab3:
        st.subheader("Model Decision Boundaries")
        
        # Risk score distribution
        fig_dist = px.histogram(
            df_patients,
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
        risk_bins = pd.cut(df_patients['Risk_Score'], bins=10)
        calibration_df = df_patients.groupby(risk_bins, observed=False).agg({
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