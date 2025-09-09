# RiskWise: An AI Risk Prediction Engine

An AI-driven risk prediction dashboard for chronic care management, built with Streamlit and featuring sophisticated machine learning models for 90-day deterioration prediction.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-green?style=for-the-badge)

![Image](https://github.com/user-attachments/assets/a0f13059-ad1b-45b2-b2e1-533f29c382c1)

## 🌟 Features

### 📊 **Cohort Overview Dashboard**
- **Real-time Risk Stratification**: Interactive patient cohort analytics with immediate risk classification
- **Professional Light Theme**: Clean, medical-grade interface optimized for clinical environments
- **Advanced Visualizations**: Risk distribution charts, correlation analysis, and predictive insights
- **Smart Table Rendering**: Dynamic patient tables with risk-based color coding and filtering

### 👤 **Patient Deep Dive Analysis**  
- **Individual Risk Profiles**: Comprehensive patient-specific risk assessment and clinical summary
- **Temporal Trends**: Blood pressure, HbA1c, and activity monitoring over time
- **AI-Powered Insights**: SHAP-based explainable AI showing key risk factors for each patient
- **Care Recommendations**: Personalized monitoring plans and intervention strategies

### 🤖 **Model Analytics & Performance**
- **Ensemble ML Models**: XGBoost + Random Forest with 94.2% accuracy
- **Feature Importance Analysis**: Global and patient-specific feature impact visualization
- **Model Calibration**: Real-time performance metrics and calibration analysis
- **Explainable AI**: SHAP values for transparent decision-making

### 🎨 **Professional UI/UX**
- **Responsive Design**: Collapsible sidebar with smart navigation controls
- **Glass Morphism**: Modern design with backdrop filters and subtle animations
- **Accessible Interface**: High contrast ratios and clear typography for medical professionals
- **Theme Consistency**: Light theme optimized for clinical workflows

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chu817/WellDoc-Project.git
   cd WellDoc-Project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:8501` to access the dashboard

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=1.7.0
shap>=0.42.0
```

## 🏗️ Architecture

### Data Pipeline
- **Synthetic Patient Data**: Realistic chronic care patient simulation with clinical parameters
- **Feature Engineering**: Multi-dimensional risk factors including demographics, vitals, labs, and social determinants
- **Real-time Processing**: Dynamic risk score calculation with immediate dashboard updates

### Machine Learning Stack
- **Primary Model**: XGBoost Classifier (AUC: 0.85+)
- **Secondary Model**: Random Forest Classifier  
- **Ensemble Method**: Weighted averaging for improved prediction accuracy
- **Explainability**: SHAP TreeExplainer for feature importance and decision transparency

### Frontend Architecture
- **Framework**: Streamlit with custom CSS/HTML components
- **Styling**: Professional medical theme with glass morphism effects
- **Interactivity**: Real-time filtering, collapsible navigation, and responsive charts
- **Performance**: Optimized rendering with smart caching and background processing

## 📊 Dashboard Components

### 1. Navigation System
- **Collapsible Sidebar**: Model performance metrics and patient cohort overview
- **Quick Navigation**: Streamlined selectbox when sidebar is collapsed
- **Smart Toggle**: Burger menu with visual state indicators

### 2. Risk Analytics
- **Risk Cards**: High/Medium/Low risk patient counts with actionable insights
- **Interactive Tables**: Sortable, filterable patient data with risk-based styling
- **Correlation Analysis**: Feature correlation heatmaps with light theme optimization

### 3. Patient Management
- **Risk Filtering**: Filter patients by risk level for targeted interventions
- **Clinical Summary**: Demographics, vitals, labs, and care utilization metrics
- **Trend Analysis**: Time-series visualizations for key health indicators
- **Care Planning**: Personalized monitoring schedules and intervention recommendations

## 🔧 Configuration

### Theme Customization
The application uses a professional light theme optimized for medical environments. Key color scheme:
- **Primary**: `#10b981` (Medical Green)
- **Warning**: `#f59e0b` (Amber)
- **Danger**: `#ef4444` (Red)
- **Background**: `#ffffff` (Pure White)
- **Text**: `#1e293b` (Dark Gray)

### Model Parameters
- **Risk Thresholds**: Low (<40%), Medium (40-70%), High (>70%)
- **Feature Count**: 14 clinical and demographic features
- **Training Split**: 70/30 train-test split with stratification
- **Update Frequency**: Real-time risk score recalculation

## 🚑 Clinical Use Cases

1. **Risk Stratification**: Identify high-risk patients requiring immediate intervention
2. **Care Coordination**: Optimize resource allocation based on predicted deterioration risk
3. **Population Health**: Monitor cohort-level trends and outcomes
4. **Clinical Decision Support**: Evidence-based recommendations for care planning
5. **Quality Improvement**: Track intervention effectiveness and model performance

## 🔬 Technical Details

### Performance Optimizations
- **Caching**: Streamlit caching for data generation and model training
- **Lazy Loading**: Progressive chart rendering to improve initial load times
- **Memory Management**: Efficient data structures and garbage collection
- **CSS Optimization**: Minimized styling rules and optimized selectors

### Security & Privacy
- **Synthetic Data**: No real patient information used in demonstration
- **Local Processing**: All computations performed client-side
- **HIPAA Considerations**: Architecture designed for compliance with healthcare data requirements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This is a prototype for demonstration purposes only. Not intended for clinical use.** 

This application is designed to showcase AI-driven healthcare analytics capabilities and should not be used for actual patient care decisions without proper clinical validation, regulatory approval, and integration with certified medical systems.

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Review the documentation and examples

---

**Built with ❤️ for healthcare innovation**
