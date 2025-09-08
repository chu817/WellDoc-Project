"""
Model Training Module for Chronic Care Risk Prediction
Handles ML model training, evaluation, and SHAP analysis
"""

import pandas as pd
import numpy as np
import xgboost
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RiskPredictionModel:
    """
    Ensemble model for chronic care risk prediction with interpretability features.
    """
    
    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.feature_cols = [
            'Age', 'Comorbidity_Count', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate',
            'HbA1c', 'Creatinine', 'LDL_Cholesterol', 'Med_Adherence', 'Steps_Per_Day',
            'BMI', 'ER_Visits_6M', 'Missed_Appointments', 'Days_Since_Last_Hosp'
        ]
        self.shap_explainer = None
        self.model_info = {}
    
    def load_data(self, csv_file='patient_data.csv'):
        """
        Load patient data from CSV file.
        
        Args:
            csv_file (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded patient data
        """
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} patient records from {csv_file}")
            return df
        except FileNotFoundError:
            print(f"Error: {csv_file} not found. Please run data_generator.py first.")
            return None
    
    def train_models(self, df):
        """
        Train XGBoost and Random Forest models for ensemble prediction.
        
        Args:
            df (pd.DataFrame): Patient data
            
        Returns:
            dict: Model performance metrics and information
        """
        # Prepare features and target
        X = df[self.feature_cols]
        y = df['Risk_of_Deterioration_90d']
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features for better model performance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_scaled = self.scaler.transform(X)
        
        # Primary model: XGBoost with optimized hyperparameters
        print("Training XGBoost model...")
        self.xgb_model = xgboost.XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self.xgb_model.fit(X_train_scaled, y_train)
        
        # Secondary model: Random Forest for ensemble comparison
        print("Training Random Forest model...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Generate probability predictions from both models
        xgb_pred_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
        rf_pred_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        
        # Create weighted ensemble prediction
        ensemble_pred = 0.7 * xgb_pred_proba + 0.3 * rf_pred_proba
        df['Risk_Score'] = ensemble_pred
        
        # Performance evaluation metrics
        xgb_test_pred = self.xgb_model.predict_proba(X_test_scaled)[:, 1]
        rf_test_pred = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        ensemble_test_pred = 0.7 * xgb_test_pred + 0.3 * rf_test_pred
        
        xgb_auc = roc_auc_score(y_test, xgb_test_pred)
        rf_auc = roc_auc_score(y_test, rf_test_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_test_pred)
        
        # SHAP values for model interpretability
        print("Generating SHAP explanations...")
        self.shap_explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = self.shap_explainer.shap_values(X_scaled)
        
        # Feature importance from both models
        xgb_feature_importance = dict(zip(self.feature_cols, self.xgb_model.feature_importances_))
        rf_feature_importance = dict(zip(self.feature_cols, self.rf_model.feature_importances_))
        
        # Store model information
        self.model_info = {
            'xgb_auc': xgb_auc,
            'rf_auc': rf_auc,
            'ensemble_auc': ensemble_auc,
            'xgb_feature_importance': xgb_feature_importance,
            'rf_feature_importance': rf_feature_importance,
            'shap_values': shap_values,
            'feature_names': self.feature_cols,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'positive_rate': y.mean()
        }
        
        print(f"Model training complete!")
        print(f"XGBoost AUC: {xgb_auc:.3f}")
        print(f"Random Forest AUC: {rf_auc:.3f}")
        print(f"Ensemble AUC: {ensemble_auc:.3f}")
        
        return df, self.model_info
    
    def predict_risk(self, patient_data):
        """
        Predict risk for new patient data.
        
        Args:
            patient_data (dict, pd.Series, or pd.DataFrame): Patient features
            
        Returns:
            float or array: Risk probability
        """
        if isinstance(patient_data, dict):
            # Convert dict to DataFrame
            patient_df = pd.DataFrame([patient_data])
        elif isinstance(patient_data, pd.Series):
            # Convert Series to DataFrame
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Ensure all required features are present
        for col in self.feature_cols:
            if col not in patient_df.columns:
                print(f"Warning: Missing feature {col}")
                return None
        
        # Scale features
        X_scaled = self.scaler.transform(patient_df[self.feature_cols])
        
        # Generate ensemble prediction
        xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
        rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
        ensemble_pred = 0.7 * xgb_pred + 0.3 * rf_pred
        
        return ensemble_pred[0] if len(ensemble_pred) == 1 else ensemble_pred
    
    def predict_batch(self, df):
        """
        Efficient batch prediction for large datasets.
        
        Args:
            df (pd.DataFrame): DataFrame with patient features
            
        Returns:
            np.array: Risk probabilities for all patients
        """
        # Ensure all required features are present
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing features {missing_cols}")
            return None
        
        # Scale features in batch
        X_scaled = self.scaler.transform(df[self.feature_cols])
        
        # Generate ensemble predictions in batch
        xgb_pred = self.xgb_model.predict_proba(X_scaled)[:, 1]
        rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
        ensemble_pred = 0.7 * xgb_pred + 0.3 * rf_pred
        
        return ensemble_pred
    
    def get_feature_importance(self, model_type='ensemble'):
        """
        Get feature importance scores.
        
        Args:
            model_type (str): 'xgb', 'rf', or 'ensemble'
            
        Returns:
            dict: Feature importance scores
        """
        if model_type == 'xgb':
            return self.model_info['xgb_feature_importance']
        elif model_type == 'rf':
            return self.model_info['rf_feature_importance']
        else:
            # Average of both models
            xgb_imp = self.model_info['xgb_feature_importance']
            rf_imp = self.model_info['rf_feature_importance']
            return {
                feature: (xgb_imp[feature] + rf_imp[feature]) / 2
                for feature in self.feature_cols
            }
    
    def get_shap_explanation(self, patient_index=None):
        """
        Get SHAP explanations for model predictions.
        
        Args:
            patient_index (int): Index of patient to explain (None for all)
            
        Returns:
            numpy.ndarray: SHAP values
        """
        if self.shap_explainer is None:
            print("SHAP explainer not available. Train model first.")
            return None
        
        shap_values = self.model_info['shap_values']
        
        if patient_index is not None:
            return shap_values[patient_index]
        return shap_values
    
    def save_models(self, model_dir='models'):
        """
        Save trained models to disk.
        
        Args:
            model_dir (str): Directory to save models
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.xgb_model, f'{model_dir}/xgb_model.pkl')
        joblib.dump(self.rf_model, f'{model_dir}/rf_model.pkl')
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        joblib.dump(self.model_info, f'{model_dir}/model_info.pkl')
        
        print(f"Models saved to {model_dir}/")
    
    def load_models(self, model_dir='models'):
        """
        Load trained models from disk.
        
        Args:
            model_dir (str): Directory containing saved models
        """
        try:
            self.xgb_model = joblib.load(f'{model_dir}/xgb_model.pkl')
            self.rf_model = joblib.load(f'{model_dir}/rf_model.pkl')
            self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
            self.model_info = joblib.load(f'{model_dir}/model_info.pkl')
            
            # Recreate SHAP explainer
            self.shap_explainer = shap.TreeExplainer(self.xgb_model)
            
            print(f"Models loaded from {model_dir}/")
            return True
        except FileNotFoundError:
            print(f"Error: Model files not found in {model_dir}/")
            return False

def train_risk_prediction_models(csv_file='patient_data.csv', save_models=True):
    """
    Main function to train risk prediction models.
    
    Args:
        csv_file (str): Path to patient data CSV
        save_models (bool): Whether to save trained models
        
    Returns:
        tuple: (DataFrame with predictions, model info, trained model object)
    """
    # Initialize model trainer
    model_trainer = RiskPredictionModel()
    
    # Load data
    df = model_trainer.load_data(csv_file)
    if df is None:
        return None, None, None
    
    # Train models
    df_with_predictions, model_info = model_trainer.train_models(df)
    
    # Save models if requested
    if save_models:
        model_trainer.save_models()
    
    return df_with_predictions, model_info, model_trainer

if __name__ == "__main__":
    print("Training risk prediction models...")
    df, info, model = train_risk_prediction_models()
    
    if df is not None:
        print(f"\nModel Performance Summary:")
        print(f"Ensemble AUC: {info['ensemble_auc']:.3f}")
        print(f"Training samples: {info['n_train_samples']}")
        print(f"Test samples: {info['n_test_samples']}")
        print(f"Positive rate: {info['positive_rate']:.3f}")
        
        # Display top features
        feature_importance = model.get_feature_importance()
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 Features:")
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.3f}")
    else:
        print("Training failed. Please check data file.")
