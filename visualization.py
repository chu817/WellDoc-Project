"""
Visualization Module for Chronic Care Risk Prediction Dashboard
Contains all chart generation functions for the Streamlit app
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

# Set consistent plotly theme
import plotly.io as pio
pio.templates.default = "plotly_white"

def create_risk_distribution_chart(df):
    """
    Create risk score distribution histogram.
    
    Args:
        df (pd.DataFrame): Patient data with Risk_Score column
        
    Returns:
        plotly.graph_objects.Figure: Risk distribution chart
    """
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=df['Risk_Score'],
        nbinsx=30,
        name='Risk Distribution',
        marker_color='rgba(16, 185, 129, 0.7)',
        marker_line=dict(color='rgb(16, 185, 129)', width=1)
    ))
    
    # Add threshold lines
    fig.add_vline(x=0.3, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk", annotation_position="top left")
    fig.add_vline(x=0.6, line_dash="dash", line_color="red", 
                  annotation_text="High Risk", annotation_position="top left")
    
    fig.update_layout(
        title="Patient Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Number of Patients",
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_age_risk_scatter(df):
    """
    Create age vs risk score scatter plot.
    
    Args:
        df (pd.DataFrame): Patient data
        
    Returns:
        plotly.graph_objects.Figure: Age vs risk scatter plot
    """
    # Create risk categories for color coding
    df['Risk_Category'] = pd.cut(df['Risk_Score'], 
                                bins=[0, 0.3, 0.6, 1.0], 
                                labels=['Low', 'Medium', 'High'])
    
    colors = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'}
    
    fig = go.Figure()
    
    for category in ['Low', 'Medium', 'High']:
        category_data = df[df['Risk_Category'] == category]
        fig.add_trace(go.Scatter(
            x=category_data['Age'],
            y=category_data['Risk_Score'],
            mode='markers',
            name=f'{category} Risk',
            marker=dict(
                color=colors[category],
                size=8,
                opacity=0.7,
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>Age:</b> %{x}<br><b>Risk Score:</b> %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Age vs Risk Score Distribution",
        xaxis_title="Age",
        yaxis_title="Risk Score",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_adherence_risk_chart(df):
    """
    Create medication adherence vs risk chart.
    
    Args:
        df (pd.DataFrame): Patient data
        
    Returns:
        plotly.graph_objects.Figure: Adherence vs risk chart
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Medication Adherence vs Risk', 'ER Visits Distribution'),
        specs=[[{"secondary_y": False}, {"type": "bar"}]]
    )
    
    # Adherence scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['Med_Adherence'],
            y=df['Risk_Score'],
            mode='markers',
            name='Patients',
            marker=dict(
                color=df['Risk_Score'],
                colorscale='RdYlGn_r',
                size=6,
                opacity=0.7,
                showscale=True,
                colorbar=dict(title="Risk Score", x=0.45)
            ),
            hovertemplate='<b>Adherence:</b> %{x:.1%}<br><b>Risk:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # ER visits bar chart
    er_counts = df['ER_Visits_6M'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=er_counts.index,
            y=er_counts.values,
            name='ER Visits',
            marker_color='rgba(239, 68, 68, 0.7)',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Medication Adherence", row=1, col=1)
    fig.update_yaxes(title_text="Risk Score", row=1, col=2)
    fig.update_xaxes(title_text="ER Visits (6 months)", row=1, col=2)
    fig.update_yaxes(title_text="Patient Count", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_comorbidity_analysis(df):
    """
    Create comorbidity count analysis chart.
    
    Args:
        df (pd.DataFrame): Patient data
        
    Returns:
        plotly.graph_objects.Figure: Comorbidity analysis chart
    """
    # Group by comorbidity count and calculate mean risk
    comorbidity_risk = df.groupby('Comorbidity_Count').agg({
        'Risk_Score': 'mean',
        'Patient_ID': 'count'
    }).reset_index()
    comorbidity_risk.rename(columns={'Patient_ID': 'Patient_Count'}, inplace=True)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar chart for patient count
    fig.add_trace(
        go.Bar(
            x=comorbidity_risk['Comorbidity_Count'],
            y=comorbidity_risk['Patient_Count'],
            name='Patient Count',
            marker_color='rgba(59, 130, 246, 0.7)',
            yaxis='y2'
        )
    )
    
    # Line chart for mean risk
    fig.add_trace(
        go.Scatter(
            x=comorbidity_risk['Comorbidity_Count'],
            y=comorbidity_risk['Risk_Score'],
            mode='lines+markers',
            name='Mean Risk Score',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=8, color='#ef4444')
        )
    )
    
    fig.update_xaxes(title_text="Comorbidity Count")
    fig.update_yaxes(title_text="Mean Risk Score", secondary_y=False)
    fig.update_yaxes(title_text="Patient Count", secondary_y=True)
    
    fig.update_layout(
        title="Comorbidity Burden vs Risk Assessment",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_patient_timeline_chart(patient_data, metric_type='bp'):
    """
    Create patient timeline monitoring chart.
    
    Args:
        patient_data (pd.DataFrame): Time series data for patient
        metric_type (str): Type of metric ('bp', 'hba1c', 'steps')
        
    Returns:
        plotly.graph_objects.Figure: Patient timeline chart
    """
    if metric_type == 'bp':
        fig = go.Figure()
        
        # Systolic BP
        fig.add_trace(go.Scatter(
            x=patient_data['Date'],
            y=patient_data['Systolic'],
            mode='lines+markers',
            name='Systolic BP',
            line=dict(color='#ef4444', width=2),
            marker=dict(size=4)
        ))
        
        # Diastolic BP
        fig.add_trace(go.Scatter(
            x=patient_data['Date'],
            y=patient_data['Diastolic'],
            mode='lines+markers',
            name='Diastolic BP',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=4)
        ))
        
        # Target ranges
        fig.add_hline(y=140, line_dash="dash", line_color="red", opacity=0.5,
                      annotation_text="Systolic Target")
        fig.add_hline(y=90, line_dash="dash", line_color="blue", opacity=0.5,
                      annotation_text="Diastolic Target")
        
        fig.update_layout(
            title="12-Week Blood Pressure Trend",
            xaxis_title="Date",
            yaxis_title="Blood Pressure (mmHg)",
            height=350
        )
    
    elif metric_type == 'hba1c':
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=patient_data['Date'],
            y=patient_data['HbA1c'],
            mode='lines+markers',
            name='HbA1c',
            line=dict(color='#8b5cf6', width=3),
            marker=dict(size=8, color='#8b5cf6')
        ))
        
        # Target line
        fig.add_hline(y=7.0, line_dash="dash", line_color="green",
                      annotation_text="Target HbA1c <7%")
        
        fig.update_layout(
            title="Quarterly HbA1c Levels",
            xaxis_title="Date",
            yaxis_title="HbA1c (%)",
            height=350
        )
    
    elif metric_type == 'steps':
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=patient_data['Date'],
            y=patient_data['Steps'],
            mode='lines',
            name='Daily Steps',
            line=dict(color='#10b981', width=2),
            fill='tonexty'
        ))
        
        # Target line
        fig.add_hline(y=10000, line_dash="dash", line_color="orange",
                      annotation_text="Target: 10,000 steps")
        
        fig.update_layout(
            title="30-Day Step Count",
            xaxis_title="Date",
            yaxis_title="Steps per Day",
            height=350
        )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    """
    Create feature importance horizontal bar chart.
    
    Args:
        feature_importance (dict): Feature importance scores
        
    Returns:
        plotly.graph_objects.Figure: Feature importance chart
    """
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_features)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker=dict(
            color=importances,
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Feature Importance in Risk Prediction",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_risk_category_pie_chart(df):
    """
    Create pie chart showing risk category distribution.
    
    Args:
        df (pd.DataFrame): Patient data with Risk_Score
        
    Returns:
        plotly.graph_objects.Figure: Risk category pie chart
    """
    # Create risk categories
    df['Risk_Category'] = pd.cut(df['Risk_Score'], 
                                bins=[0, 0.3, 0.6, 1.0], 
                                labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    risk_counts = df['Risk_Category'].value_counts()
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker=dict(
            colors=['#10b981', '#f59e0b', '#ef4444'],
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textfont_size=12,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Patient Risk Category Distribution",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_condition_risk_analysis(df):
    """
    Create analysis of risk by primary condition.
    
    Args:
        df (pd.DataFrame): Patient data
        
    Returns:
        plotly.graph_objects.Figure: Condition risk analysis
    """
    condition_stats = df.groupby('Primary_Condition').agg({
        'Risk_Score': ['mean', 'std', 'count']
    }).round(3)
    
    condition_stats.columns = ['Mean_Risk', 'Risk_Std', 'Patient_Count']
    condition_stats = condition_stats.reset_index()
    
    fig = go.Figure()
    
    # Add bar chart with error bars
    fig.add_trace(go.Bar(
        x=condition_stats['Primary_Condition'],
        y=condition_stats['Mean_Risk'],
        error_y=dict(
            type='data',
            array=condition_stats['Risk_Std'],
            visible=True,
            color='rgba(0,0,0,0.3)'
        ),
        marker=dict(
            color=condition_stats['Mean_Risk'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Mean Risk")
        ),
        text=condition_stats['Patient_Count'],
        texttemplate='n=%{text}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Mean Risk: %{y:.3f}<br>Patients: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Risk Assessment by Primary Condition",
        xaxis_title="Primary Condition",
        yaxis_title="Mean Risk Score",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(tickangle=45)
    )
    
    return fig
