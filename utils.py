"""
Utility Functions for Chronic Care Risk Prediction Dashboard
Contains helper functions for data processing and UI components
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from config import MEDICAL_REFERENCES, get_risk_category, get_risk_color, format_medical_value

def create_metric_card(title, value, delta=None, delta_color="normal", help_text=None):
    """
    Create a styled metric card for displaying key metrics.
    
    Args:
        title (str): Metric title
        value (str): Metric value
        delta (str, optional): Change indicator
        delta_color (str): Color for delta ("normal", "inverse")
        help_text (str, optional): Help tooltip text
        
    Returns:
        None: Displays metric using Streamlit
    """
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )

def display_risk_indicator(risk_score, patient_id=None):
    """
    Display a styled risk indicator based on risk score.
    
    Args:
        risk_score (float): Risk probability score (0-1)
        patient_id (str, optional): Patient identifier
        
    Returns:
        None: Displays risk indicator using Streamlit
    """
    risk_category = get_risk_category(risk_score)
    risk_color = get_risk_color(risk_category)
    
    # Determine risk level styling
    if risk_category == 'High':
        risk_class = "high-risk"
        icon = "🔴"
    elif risk_category == 'Medium':
        risk_class = "medium-risk"
        icon = "🟡"
    else:
        risk_class = "low-risk"
        icon = "🟢"
    
    patient_text = f" - {patient_id}" if patient_id else ""
    
    st.markdown(f"""
    <div class="risk-indicator {risk_class}">
        <h3 style="margin: 0; color: inherit;">
            {icon} {risk_category} Risk{patient_text}
        </h3>
        <h2 style="margin: 0.5rem 0 0 0; color: inherit;">
            {risk_score:.1%}
        </h2>
        <p style="margin: 0; opacity: 0.8; color: inherit;">
            90-Day Deterioration Probability
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar_section(title, content_func):
    """
    Create a styled sidebar section.
    
    Args:
        title (str): Section title
        content_func (callable): Function to render section content
        
    Returns:
        None: Displays sidebar section using Streamlit
    """
    st.markdown(f"""
    <div class="sidebar-content">
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        content_func()

def format_patient_summary(patient_data):
    """
    Format patient data into a readable summary.
    
    Args:
        patient_data (dict or pd.Series): Patient information
        
    Returns:
        dict: Formatted patient summary
    """
    summary = {}
    
    # Basic demographics
    summary['Age'] = f"{patient_data['Age']} years"
    summary['Gender'] = patient_data.get('Gender', 'Not specified')
    summary['Primary Condition'] = patient_data.get('Primary_Condition', 'Not specified')
    
    # Clinical metrics
    summary['Blood Pressure'] = f"{patient_data['Systolic_BP']:.0f}/{patient_data['Diastolic_BP']:.0f} mmHg"
    summary['Heart Rate'] = f"{patient_data['Heart_Rate']:.0f} bpm"
    summary['HbA1c'] = format_medical_value(patient_data['HbA1c'], 'hba1c')
    summary['Creatinine'] = format_medical_value(patient_data['Creatinine'], 'creatinine')
    summary['LDL Cholesterol'] = format_medical_value(patient_data['LDL_Cholesterol'], 'cholesterol')
    summary['BMI'] = format_medical_value(patient_data['BMI'], 'bmi')
    
    # Behavioral factors
    summary['Medication Adherence'] = format_medical_value(patient_data['Med_Adherence'], 'adherence')
    summary['Daily Steps'] = format_medical_value(patient_data['Steps_Per_Day'], 'steps')
    
    # Healthcare utilization
    summary['ER Visits (6 months)'] = f"{patient_data['ER_Visits_6M']:.0f} visits"
    summary['Missed Appointments'] = f"{patient_data['Missed_Appointments']:.0f} appointments"
    summary['Days Since Hospitalization'] = format_medical_value(patient_data['Days_Since_Last_Hosp'], 'days')
    summary['Comorbidity Count'] = f"{patient_data['Comorbidity_Count']:.0f} conditions"
    
    return summary

def assess_clinical_values(patient_data):
    """
    Assess clinical values against reference ranges.
    
    Args:
        patient_data (dict or pd.Series): Patient clinical data
        
    Returns:
        dict: Assessment of each clinical parameter
    """
    assessments = {}
    
    # Blood pressure assessment
    systolic = patient_data['Systolic_BP']
    diastolic = patient_data['Diastolic_BP']
    
    if systolic >= MEDICAL_REFERENCES['blood_pressure']['systolic_high'] or \
       diastolic >= MEDICAL_REFERENCES['blood_pressure']['diastolic_high']:
        assessments['Blood Pressure'] = {
            'status': 'High',
            'color': 'status-high',
            'recommendation': 'Monitor closely, consider medication adjustment'
        }
    elif systolic >= 130 or diastolic >= 80:
        assessments['Blood Pressure'] = {
            'status': 'Elevated',
            'color': 'status-medium',
            'recommendation': 'Lifestyle modifications recommended'
        }
    else:
        assessments['Blood Pressure'] = {
            'status': 'Normal',
            'color': 'status-low',
            'recommendation': 'Continue current management'
        }
    
    # HbA1c assessment
    hba1c = patient_data['HbA1c']
    if hba1c >= MEDICAL_REFERENCES['hba1c']['poor_control']:
        assessments['HbA1c'] = {
            'status': 'Poor Control',
            'color': 'status-high',
            'recommendation': 'Urgent diabetes management review needed'
        }
    elif hba1c >= MEDICAL_REFERENCES['hba1c']['target']:
        assessments['HbA1c'] = {
            'status': 'Needs Improvement',
            'color': 'status-medium',
            'recommendation': 'Intensify diabetes management'
        }
    else:
        assessments['HbA1c'] = {
            'status': 'Good Control',
            'color': 'status-low',
            'recommendation': 'Maintain current diabetes management'
        }
    
    # BMI assessment
    bmi = patient_data['BMI']
    if bmi >= MEDICAL_REFERENCES['bmi']['obese']:
        assessments['BMI'] = {
            'status': 'Obese',
            'color': 'status-high',
            'recommendation': 'Weight management program recommended'
        }
    elif bmi >= MEDICAL_REFERENCES['bmi']['overweight']:
        assessments['BMI'] = {
            'status': 'Overweight',
            'color': 'status-medium',
            'recommendation': 'Lifestyle modifications for weight loss'
        }
    else:
        assessments['BMI'] = {
            'status': 'Normal',
            'color': 'status-low',
            'recommendation': 'Maintain healthy weight'
        }
    
    # Medication adherence assessment
    adherence = patient_data['Med_Adherence']
    if adherence < 0.8:
        assessments['Medication Adherence'] = {
            'status': 'Poor',
            'color': 'status-high',
            'recommendation': 'Adherence counseling and support needed'
        }
    elif adherence < 0.9:
        assessments['Medication Adherence'] = {
            'status': 'Moderate',
            'color': 'status-medium',
            'recommendation': 'Encourage better adherence'
        }
    else:
        assessments['Medication Adherence'] = {
            'status': 'Good',
            'color': 'status-low',
            'recommendation': 'Continue current adherence patterns'
        }
    
    # Activity level assessment
    steps = patient_data['Steps_Per_Day']
    if steps < MEDICAL_REFERENCES['steps']['sedentary']:
        assessments['Physical Activity'] = {
            'status': 'Sedentary',
            'color': 'status-high',
            'recommendation': 'Increase physical activity gradually'
        }
    elif steps < MEDICAL_REFERENCES['steps']['target']:
        assessments['Physical Activity'] = {
            'status': 'Moderate',
            'color': 'status-medium',
            'recommendation': 'Aim for 10,000 steps daily'
        }
    else:
        assessments['Physical Activity'] = {
            'status': 'Active',
            'color': 'status-low',
            'recommendation': 'Maintain current activity level'
        }
    
    return assessments

def display_clinical_assessment(assessments):
    """
    Display clinical assessments in a formatted table.
    
    Args:
        assessments (dict): Clinical parameter assessments
        
    Returns:
        None: Displays assessment table using Streamlit
    """
    st.markdown("### Clinical Parameter Assessment")
    
    for parameter, assessment in assessments.items():
        status = assessment['status']
        color_class = assessment['color']
        recommendation = assessment['recommendation']
        
        st.markdown(f"""
        <div style="margin: 0.5rem 0; padding: 0.75rem; border-radius: 8px; background: rgba(255,255,255,0.9); border: 1px solid rgba(148,163,184,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <strong>{parameter}</strong>
                <span class="{color_class}">{status}</span>
            </div>
            <div style="font-size: 0.9rem; color: #64748b;">
                {recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)

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

def create_patient_selector(df):
    """
    Create a patient selection widget.
    
    Args:
        df (pd.DataFrame): Patient dataset
        
    Returns:
        str: Selected patient ID
    """
    # Create patient options with risk info
    patient_options = []
    for _, patient in df.iterrows():
        risk_category = get_risk_category(patient['Risk_Score'])
        risk_icon = "🔴" if risk_category == "High" else "🟡" if risk_category == "Medium" else "🟢"
        patient_options.append(f"{risk_icon} {patient['Patient_ID']} - {risk_category} Risk")
    
    selected_option = st.selectbox(
        "Select Patient for Detailed Analysis",
        patient_options,
        help="Choose a patient to view detailed risk assessment and recommendations"
    )
    
    # Extract patient ID from selection
    patient_id = selected_option.split(' ')[1]
    return patient_id

def generate_recommendations(patient_data, risk_score):
    """
    Generate personalized care recommendations based on patient data and risk score.
    
    Args:
        patient_data (dict or pd.Series): Patient information
        risk_score (float): Risk probability score
        
    Returns:
        list: List of personalized recommendations
    """
    recommendations = []
    risk_category = get_risk_category(risk_score)
    
    # Risk-based recommendations
    if risk_category == "High":
        recommendations.append("🚨 **Immediate Action Required**: Schedule urgent clinical review within 7 days")
        recommendations.append("📞 **Enhanced Monitoring**: Implement daily check-ins or remote monitoring")
        recommendations.append("💊 **Medication Review**: Evaluate current medications and adherence barriers")
    elif risk_category == "Medium":
        recommendations.append("⚠️ **Increased Monitoring**: Schedule follow-up within 2-3 weeks")
        recommendations.append("📋 **Care Plan Review**: Update care plan to address identified risk factors")
    else:
        recommendations.append("✅ **Routine Monitoring**: Continue current care plan with scheduled follow-ups")
    
    # Clinical parameter-based recommendations
    if patient_data['Systolic_BP'] >= 140 or patient_data['Diastolic_BP'] >= 90:
        recommendations.append("🩺 **Blood Pressure**: Monitor BP closely, consider antihypertensive adjustment")
    
    if patient_data['HbA1c'] >= 8.0:
        recommendations.append("🍯 **Diabetes Management**: HbA1c above target - intensify diabetes therapy")
    
    if patient_data['Med_Adherence'] < 0.8:
        recommendations.append("💊 **Adherence Support**: Implement adherence aids (pill organizers, reminders)")
    
    if patient_data['Steps_Per_Day'] < 5000:
        recommendations.append("🚶 **Physical Activity**: Initiate supervised exercise program")
    
    if patient_data['BMI'] >= 30:
        recommendations.append("⚖️ **Weight Management**: Refer to nutritionist and weight management program")
    
    if patient_data['ER_Visits_6M'] >= 2:
        recommendations.append("🏥 **Care Coordination**: Implement care transitions program to reduce ER visits")
    
    # Social determinants
    if 'Social_Risk_Score' in patient_data and patient_data['Social_Risk_Score'] > 0.5:
        recommendations.append("🏠 **Social Support**: Connect with social services for additional support")
    
    return recommendations

def export_patient_report(patient_data, risk_score, recommendations):
    """
    Generate a downloadable patient report.
    
    Args:
        patient_data (dict or pd.Series): Patient information
        risk_score (float): Risk probability score
        recommendations (list): Care recommendations
        
    Returns:
        str: Formatted report text
    """
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    risk_category = get_risk_category(risk_score)
    
    report = f"""
CHRONIC CARE RISK ASSESSMENT REPORT
Generated: {report_date}

PATIENT INFORMATION
==================
Patient ID: {patient_data.get('Patient_ID', 'N/A')}
Age: {patient_data['Age']} years
Gender: {patient_data.get('Gender', 'Not specified')}
Primary Condition: {patient_data.get('Primary_Condition', 'Not specified')}

RISK ASSESSMENT
==============
90-Day Deterioration Risk: {risk_score:.1%}
Risk Category: {risk_category}

CLINICAL PARAMETERS
==================
Blood Pressure: {patient_data['Systolic_BP']:.0f}/{patient_data['Diastolic_BP']:.0f} mmHg
Heart Rate: {patient_data['Heart_Rate']:.0f} bpm
HbA1c: {patient_data['HbA1c']:.1f}%
Creatinine: {patient_data['Creatinine']:.2f} mg/dL
LDL Cholesterol: {patient_data['LDL_Cholesterol']:.0f} mg/dL
BMI: {patient_data['BMI']:.1f} kg/m²

BEHAVIORAL FACTORS
=================
Medication Adherence: {patient_data['Med_Adherence']:.1%}
Daily Steps: {patient_data['Steps_Per_Day']:,}
Comorbidity Count: {patient_data['Comorbidity_Count']:.0f}

HEALTHCARE UTILIZATION
=====================
ER Visits (6 months): {patient_data['ER_Visits_6M']:.0f}
Missed Appointments: {patient_data['Missed_Appointments']:.0f}
Days Since Last Hospitalization: {patient_data['Days_Since_Last_Hosp']:.0f}

CARE RECOMMENDATIONS
===================
"""
    
    for i, rec in enumerate(recommendations, 1):
        # Remove markdown formatting for plain text report
        clean_rec = rec.replace('**', '').replace('🚨', '').replace('⚠️', '').replace('✅', '')
        clean_rec = clean_rec.replace('🩺', '').replace('🍯', '').replace('💊', '')
        clean_rec = clean_rec.replace('🚶', '').replace('⚖️', '').replace('🏥', '').replace('🏠', '')
        report += f"{i}. {clean_rec.strip()}\n"
    
    report += f"""

DISCLAIMER
==========
This report is generated by an AI-powered risk prediction model and should be used
as a clinical decision support tool. All recommendations should be reviewed by 
qualified healthcare professionals before implementation.

Report generated by AI Risk Prediction Engine v1.0
"""
    
    return report
