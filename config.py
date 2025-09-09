"""
Configuration Module for Chronic Care Risk Prediction Dashboard
Contains essential configuration settings
"""

import streamlit as st

# App Configuration
APP_CONFIG = {
    'page_title': "RiskWise",
    'layout': "wide",
    'initial_sidebar_state': "expanded",
    'menu_items': {
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
}

# Data generation configuration
DATA_CONFIG = {
    'num_patients': 50000,
    'csv_filename': 'patient_data.csv',
    'time_series_days': 90
}

def apply_theme_config():
    """Apply theme configuration to Streamlit app."""
    st._config.set_option('theme.base', 'light')
    st._config.set_option('theme.backgroundColor', '#ffffff')
    st._config.set_option('theme.secondaryBackgroundColor', '#f0f2f6')
    st._config.set_option('theme.textColor', '#262730')
