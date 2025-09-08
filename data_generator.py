"""
Data Generator Module for Chronic Care Risk Prediction
Generates realistic patient data and saves to CSV for model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_comprehensive_patient_data(num_patients=150, save_to_csv=True, filename='patient_data.csv'):
    """
    Generates realistic chronic care patient data with multiple conditions and risk factors.
    
    Args:
        num_patients (int): Number of patients to generate
        save_to_csv (bool): Whether to save the data to CSV
        filename (str): Name of the CSV file to save
        
    Returns:
        pd.DataFrame: Generated patient data
    """
    np.random.seed(42)
    
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
    
    # Calculate risk of deterioration based on clinical factors
    risk_factors = (
        (ages - 35) / 60 * 0.15 +  # Age factor
        comorbidity_counts / 6 * 0.20 +  # Comorbidity burden
        np.clip((systolic_bp - 120) / 60, 0, 1) * 0.15 +  # Blood pressure
        np.clip((hba1c - 7) / 7, 0, 1) * 0.15 +  # Diabetes control
        (1 - med_adherence) * 0.20 +  # Medication adherence
        er_visits_6m / 5 * 0.10 +  # Healthcare utilization
        social_risk_score * 0.05  # Social determinants
    )
    
    # Add some randomness and create binary outcome
    risk_prob = np.clip(risk_factors + np.random.normal(0, 0.1, num_patients), 0, 1)
    risk_of_deterioration = (risk_prob > 0.4).astype(int)
    
    # Compile all data into structured dataframe
    df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Age': ages,
        'Gender': genders,
        'Primary_Condition': conditions,
        'Comorbidity_Count': comorbidity_counts,
        'Systolic_BP': systolic_bp.round(1),
        'Diastolic_BP': diastolic_bp.round(1),
        'Heart_Rate': heart_rate.round(0).astype(int),
        'HbA1c': hba1c.round(1),
        'Creatinine': creatinine.round(2),
        'LDL_Cholesterol': ldl_cholesterol.round(0).astype(int),
        'Med_Adherence': med_adherence.round(3),
        'Steps_Per_Day': steps_per_day,
        'BMI': bmi.round(1),
        'ER_Visits_6M': er_visits_6m,
        'Missed_Appointments': missed_appointments,
        'Insurance_Type': insurance_types,
        'Social_Risk_Score': social_risk_score.round(3),
        'Distance_to_Hospital': distance_to_hospital.round(1),
        'Days_Since_Last_Hosp': days_since_last_hosp,
        'Risk_of_Deterioration_90d': risk_of_deterioration
    })
    
    # Save to CSV if requested
    if save_to_csv:
        df.to_csv(filename, index=False)
        print(f"Patient data saved to {filename}")
        print(f"Generated {num_patients} patient records")
        print(f"Risk distribution: {risk_of_deterioration.sum()} high-risk, {num_patients - risk_of_deterioration.sum()} low-risk")
    
    return df

def generate_time_series_data(patient_id, metric_type='bp', days=84):
    """
    Generate time series data for patient monitoring charts.
    
    Args:
        patient_id (str): Patient identifier
        metric_type (str): Type of metric ('bp', 'hba1c', 'steps')
        days (int): Number of days of data to generate
        
    Returns:
        pd.DataFrame: Time series data
    """
    np.random.seed(hash(patient_id) % 2**32)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    if metric_type == 'bp':
        # Blood pressure with some trend and noise
        trend = np.linspace(0, -5, days)  # Slight improvement over time
        systolic = 145 + trend + np.random.normal(0, 8, days)
        diastolic = 85 + trend * 0.6 + np.random.normal(0, 5, days)
        
        return pd.DataFrame({
            'Date': dates,
            'Systolic': systolic.round(0).astype(int),
            'Diastolic': diastolic.round(0).astype(int)
        })
    
    elif metric_type == 'hba1c':
        # Quarterly HbA1c measurements
        quarterly_dates = pd.date_range(end=datetime.now(), periods=4, freq='3M')
        hba1c_values = 8.2 + np.cumsum(np.random.normal(-0.1, 0.3, 4))
        hba1c_values = np.clip(hba1c_values, 6.0, 12.0)
        
        return pd.DataFrame({
            'Date': quarterly_dates,
            'HbA1c': hba1c_values.round(1)
        })
    
    elif metric_type == 'steps':
        # Daily step count with weekly patterns
        base_steps = 4500
        weekly_pattern = np.tile([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 0.7], days // 7 + 1)[:days]
        steps = base_steps * weekly_pattern + np.random.normal(0, 500, days)
        steps = np.clip(steps, 1000, 12000).astype(int)
        
        return pd.DataFrame({
            'Date': dates,
            'Steps': steps
        })
    
    return pd.DataFrame()

if __name__ == "__main__":
    # Generate and save patient data
    print("Generating patient dataset...")
    df = generate_comprehensive_patient_data(num_patients=150, save_to_csv=True)
    print("Dataset generation complete!")
