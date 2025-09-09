"""
Utility Functions for Chronic Care Risk Prediction Dashboard
Contains essential helper functions for data processing
"""

import pandas as pd
import numpy as np

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for the patient dataset.
    
    Args:
        df (pd.DataFrame): Patient dataset
        
    Returns:
        dict: Summary statistics
    """
    stats = {}
    
    # Basic demographics
    stats['total_patients'] = len(df)
    stats['avg_age'] = df['Age'].mean()
    stats['age_range'] = (df['Age'].min(), df['Age'].max())
    stats['gender_distribution'] = df['Gender'].value_counts().to_dict() if 'Gender' in df.columns else {}
    
    # Risk distribution
    stats['high_risk_count'] = sum(df['Risk_Score'] >= 0.6)
    stats['medium_risk_count'] = sum((df['Risk_Score'] >= 0.3) & (df['Risk_Score'] < 0.6))
    stats['low_risk_count'] = sum(df['Risk_Score'] < 0.3)
    stats['avg_risk_score'] = df['Risk_Score'].mean()
    
    # Clinical metrics
    stats['avg_systolic_bp'] = df['Systolic_BP'].mean()
    stats['avg_diastolic_bp'] = df['Diastolic_BP'].mean()
    stats['avg_hba1c'] = df['HbA1c'].mean()
    stats['avg_bmi'] = df['BMI'].mean()
    stats['avg_adherence'] = df['Med_Adherence'].mean()
    
    # Healthcare utilization
    stats['avg_er_visits'] = df['ER_Visits_6M'].mean()
    stats['avg_missed_appointments'] = df['Missed_Appointments'].mean()
    stats['avg_comorbidities'] = df['Comorbidity_Count'].mean()
    
    return stats
