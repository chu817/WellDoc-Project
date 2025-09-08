"""
Configuration Module for Chronic Care Risk Prediction Dashboard
Contains all configuration settings, constants, and styling
"""

import streamlit as st

# App Configuration
APP_CONFIG = {
    'page_title': "AI Risk Prediction Engine - Chronic Care",
    'page_icon': "⚕️",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Theme Settings
THEME_CONFIG = {
    'base': 'light',
    'backgroundColor': '#ffffff',
    'secondaryBackgroundColor': '#f0f2f6',
    'textColor': '#262730'
}

# Model Configuration
MODEL_CONFIG = {
    'feature_columns': [
        'Age', 'Comorbidity_Count', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate',
        'HbA1c', 'Creatinine', 'LDL_Cholesterol', 'Med_Adherence', 'Steps_Per_Day',
        'BMI', 'ER_Visits_6M', 'Missed_Appointments', 'Days_Since_Last_Hosp'
    ],
    'risk_thresholds': {
        'low': 0.3,
        'medium': 0.6,
        'high': 1.0
    },
    'ensemble_weights': {
        'xgb': 0.7,
        'rf': 0.3
    }
}

# Data generation configuration
DATA_CONFIG = {
    'num_patients': 50000,
    'csv_filename': 'patient_data.csv',
    'time_series_days': 90
}

# Medical Reference Values
MEDICAL_REFERENCES = {
    'blood_pressure': {
        'systolic_normal': 120,
        'systolic_high': 140,
        'diastolic_normal': 80,
        'diastolic_high': 90
    },
    'hba1c': {
        'target': 7.0,
        'good_control': 7.5,
        'poor_control': 9.0
    },
    'steps': {
        'target': 10000,
        'sedentary': 5000
    },
    'bmi': {
        'normal': 25,
        'overweight': 30,
        'obese': 35
    }
}

# Color Schemes
COLORS = {
    'primary': '#10b981',
    'secondary': '#3b82f6',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'success': '#10b981',
    'info': '#3b82f6',
    'light': '#f8fafc',
    'dark': '#1e293b',
    'risk_low': '#10b981',
    'risk_medium': '#f59e0b',
    'risk_high': '#ef4444'
}

# CSS Styling
def get_custom_css():
    """
    Returns the complete CSS styling for the application.
    
    Returns:
        str: CSS styling string
    """
    return """
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 0rem !important;
        margin-top: 0 !important;
        max-width: 100% !important;
    }
    
    .stSidebar .block-container {
        padding-top: 0rem !important;
        margin-top: 0 !important;
    }
    
    .stApp > header {
        display: none !important;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.75rem;
        font-weight: 700;
        color: #1e293b;
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
    
    .metric-card::before, .plot-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, #10b981 50%, transparent 100%);
        opacity: 0.6;
    }
    
    .metric-card:hover, .plot-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
        background: rgba(255, 255, 255, 0.95);
        border-color: rgba(16, 185, 129, 0.4);
    }
    
    /* Risk indicator styling */
    .risk-indicator {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .risk-indicator::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    
    .high-risk {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
        color: #991b1b !important;
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    .high-risk::before {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 50%, #b91c1c 100%) !important;
    }
    
    .medium-risk {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.05) 100%);
        color: #92400e !important;
        border-color: rgba(245, 158, 11, 0.3);
    }
    
    .medium-risk::before {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 50%, #b45309 100%) !important;
    }
    
    .low-risk {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        color: #065f46 !important;
        border-color: rgba(16, 185, 129, 0.3);
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
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.1) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%) !important;
        color: #14532d !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(34, 197, 94, 0.2) !important;
        border-color: #86efac !important;
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
    
    /* Header container styling */
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
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-container h1 {
            font-size: 2rem !important;
        }
        
        .main-header {
            font-size: 2rem !important;
        }
        
        .metric-card, .plot-container {
            padding: 1rem !important;
        }
    }
    </style>
    """

# Navigation Options
NAVIGATION_OPTIONS = [
    "📊 Dashboard Overview",
    "👥 Patient Management", 
    "🔍 Individual Assessment",
    "📈 Analytics & Insights",
    "⚙️ Model Performance"
]

# Default Patient for Demo
DEFAULT_PATIENT = {
    'Age': 68,
    'Comorbidity_Count': 3,
    'Systolic_BP': 145.0,
    'Diastolic_BP': 88.0,
    'Heart_Rate': 72,
    'HbA1c': 8.2,
    'Creatinine': 1.4,
    'LDL_Cholesterol': 145,
    'Med_Adherence': 0.75,
    'Steps_Per_Day': 3500,
    'BMI': 31.2,
    'ER_Visits_6M': 2,
    'Missed_Appointments': 1,
    'Days_Since_Last_Hosp': 45
}

def apply_theme_config():
    """Apply theme configuration to Streamlit app."""
    st._config.set_option('theme.base', THEME_CONFIG['base'])
    st._config.set_option('theme.backgroundColor', THEME_CONFIG['backgroundColor'])
    st._config.set_option('theme.secondaryBackgroundColor', THEME_CONFIG['secondaryBackgroundColor'])
    st._config.set_option('theme.textColor', THEME_CONFIG['textColor'])

def get_risk_category(risk_score):
    """
    Determine risk category based on score.
    
    Args:
        risk_score (float): Risk probability score
        
    Returns:
        str: Risk category ('Low', 'Medium', 'High')
    """
    if risk_score < MODEL_CONFIG['risk_thresholds']['low']:
        return 'Low'
    elif risk_score < MODEL_CONFIG['risk_thresholds']['medium']:
        return 'Medium'
    else:
        return 'High'

def get_risk_color(risk_category):
    """
    Get color for risk category.
    
    Args:
        risk_category (str): Risk category
        
    Returns:
        str: Color hex code
    """
    color_map = {
        'Low': COLORS['risk_low'],
        'Medium': COLORS['risk_medium'],
        'High': COLORS['risk_high']
    }
    return color_map.get(risk_category, COLORS['primary'])

def format_medical_value(value, value_type):
    """
    Format medical values with appropriate units and precision.
    
    Args:
        value: The value to format
        value_type (str): Type of medical value
        
    Returns:
        str: Formatted value with units
    """
    formatters = {
        'bp': lambda x: f"{x:.0f} mmHg",
        'hba1c': lambda x: f"{x:.1f}%",
        'creatinine': lambda x: f"{x:.2f} mg/dL",
        'cholesterol': lambda x: f"{x:.0f} mg/dL",
        'adherence': lambda x: f"{x:.1%}",
        'bmi': lambda x: f"{x:.1f} kg/m²",
        'steps': lambda x: f"{x:,} steps",
        'days': lambda x: f"{x:.0f} days",
        'percentage': lambda x: f"{x:.1%}",
        'score': lambda x: f"{x:.3f}"
    }
    
    formatter = formatters.get(value_type, lambda x: str(x))
    return formatter(value)
