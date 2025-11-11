import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
import io
import base64
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings('ignore')

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from PIL import Image as PILImage
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Forecasting models
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb

# LLM integration - Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

import os

# Page config
st.set_page_config(
    page_title="Advanced Sales Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}
if 'llm_report' not in st.session_state:
    st.session_state.llm_report = None

# ==================== UTILITY FUNCTIONS ====================

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = actual != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def calculate_wape(actual, predicted):
    """Calculate Weighted Absolute Percentage Error"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    numerator = np.sum(np.abs(actual - predicted))
    denominator = np.sum(np.abs(actual))
    if denominator == 0:
        return np.inf
    return (numerator / denominator) * 100

def calculate_rmse(actual, predicted):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(actual, predicted))

def calculate_mae(actual, predicted):
    """Calculate Mean Absolute Error"""
    return mean_absolute_error(actual, predicted)

def calculate_r2(actual, predicted):
    """Calculate R-squared"""
    return r2_score(actual, predicted)

# ==================== DATA GENERATION ====================

def generate_advanced_sample_data(n_months=12, base_sales=10000, trend_strength=0.15, 
                                  seasonality_strength=0.2, noise_level=0.05, 
                                  start_date='2023-01-01', seed=42):
    """Generate customizable sample monthly sales data"""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=n_months, freq='M')
    
    # Base sales
    base = base_sales
    
    # Trend component
    trend = np.linspace(0, base * trend_strength, n_months)
    
    # Seasonality component (12-month cycle)
    seasonal_pattern = np.sin(np.arange(n_months) * 2 * np.pi / 12) * base * seasonality_strength
    seasonal_pattern += np.cos(np.arange(n_months) * 2 * np.pi / 6) * base * seasonality_strength * 0.5
    
    # Noise component
    noise = np.random.normal(0, base * noise_level, n_months)
    
    # Combine components
    sales = base + trend + seasonal_pattern + noise
    sales = np.maximum(sales, base * 0.3)  # Ensure positive values
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.astype(int)
    })
    return df

# ==================== FEATURE ENGINEERING ====================

def create_arima_features(df, max_lag=3):
    """Create features specifically for ARIMA model"""
    df = df.copy()
    # ARIMA doesn't need explicit features, but we can create lagged values for analysis
    for lag in range(1, max_lag + 1):
        df[f'ARIMA_Lag_{lag}'] = df['Sales'].shift(lag)
    return df

def create_ml_features(df, lag_features=[1,2,3], rolling_windows=[3,6], 
                      seasonal_features=True, trend_features=True):
    """Create features for ML models with customizable options"""
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    
    # Lag features (customizable)
    for lag in lag_features:
        df[f'Lag_{lag}'] = df['Sales'].shift(lag)
    
    # Rolling statistics (customizable windows)
    for window in rolling_windows:
        df[f'Rolling_Mean_{window}'] = df['Sales'].rolling(window=window).mean()
        df[f'Rolling_Std_{window}'] = df['Sales'].rolling(window=window).std()
    
    # Seasonal indicators
    if seasonal_features:
        df['Is_Q1'] = df['Quarter'].eq(1).astype(int)
        df['Is_Q2'] = df['Quarter'].eq(2).astype(int)
        df['Is_Q3'] = df['Quarter'].eq(3).astype(int)
        df['Is_Q4'] = df['Quarter'].eq(4).astype(int)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Trend component
    if trend_features:
        df['Trend'] = range(len(df))
        df['Trend_Squared'] = df['Trend'] ** 2
    
    # Year-over-year change
    df['YoY_Change'] = df['Sales'].pct_change(12)
    
    return df

# ==================== FORECASTING MODELS ====================

def naive_forecast(train_data, forecast_horizon=3):
    """Naïve forecasting: use last value"""
    last_value = train_data['Sales'].iloc[-1]
    return [last_value] * forecast_horizon, None

def arima_forecast(train_data, forecast_horizon=3, p=1, d=1, q=1):
    """ARIMA forecasting with customizable parameters"""
    try:
        model = ARIMA(train_data['Sales'], order=(p, d, q))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=forecast_horizon)
        conf_int = fitted_model.get_forecast(steps=forecast_horizon).conf_int()
        
        # Get model statistics
        model_stats = {
            'AIC': fitted_model.aic,
            'BIC': fitted_model.bic,
            'Log_Likelihood': fitted_model.llf
        }
        
        return forecast.values, {'model': fitted_model, 'stats': model_stats, 'conf_int': conf_int}
    except Exception as e:
        st.warning(f"ARIMA model error: {str(e)}")
        return naive_forecast(train_data, forecast_horizon)

def random_forest_forecast(train_data, forecast_horizon=3, n_estimators=100, 
                          max_depth=5, min_samples_split=2, feature_config=None):
    """Random Forest forecasting with customizable parameters"""
    try:
        # Prepare features
        if feature_config:
            df_features = create_ml_features(train_data, **feature_config)
        else:
            df_features = create_ml_features(train_data)
        
        df_features = df_features.dropna()
        
        if len(df_features) < 3:
            return naive_forecast(train_data, forecast_horizon)
        
        # Select feature columns
        feature_cols = [col for col in df_features.columns 
                       if col not in ['Date', 'Sales', 'Year']]
        available_features = [col for col in feature_cols if col in df_features.columns]
        
        X_train = df_features[available_features]
        y_train = df_features['Sales']
        
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = dict(zip(available_features, rf_model.feature_importances_))
        
        # Generate forecasts recursively
        forecasts = []
        last_data = df_features.iloc[-1:].copy()
        
        for i in range(forecast_horizon):
            next_month = (last_data['Month'].iloc[0] % 12) + 1
            next_quarter = ((next_month - 1) // 3) + 1
            next_trend = last_data['Trend'].iloc[0] + 1 if 'Trend' in last_data.columns else i + len(df_features)
            
            pred_features = {}
            for feat in available_features:
                if feat.startswith('Lag_'):
                    lag_num = int(feat.split('_')[1])
                    if lag_num == 1:
                        pred_features[feat] = [last_data['Sales'].iloc[0]]
                    else:
                        prev_lag = f'Lag_{lag_num-1}'
                        pred_features[feat] = [last_data[prev_lag].iloc[0] if prev_lag in last_data.columns else 0]
                elif feat == 'Month':
                    pred_features[feat] = [next_month]
                elif feat == 'Quarter':
                    pred_features[feat] = [next_quarter]
                elif feat == 'Trend':
                    pred_features[feat] = [next_trend]
                elif feat == 'Trend_Squared':
                    pred_features[feat] = [next_trend ** 2]
                elif feat.startswith('Is_Q'):
                    q_num = int(feat.split('_')[1][1])
                    pred_features[feat] = [1 if next_quarter == q_num else 0]
                elif feat == 'Month_Sin':
                    pred_features[feat] = [np.sin(2 * np.pi * next_month / 12)]
                elif feat == 'Month_Cos':
                    pred_features[feat] = [np.cos(2 * np.pi * next_month / 12)]
                elif feat.startswith('Rolling_'):
                    pred_features[feat] = [last_data[feat].iloc[0] if feat in last_data.columns else last_data['Sales'].mean()]
                else:
                    pred_features[feat] = [last_data[feat].iloc[0] if feat in last_data.columns else 0]
            
            pred_df = pd.DataFrame(pred_features)
            pred = rf_model.predict(pred_df)[0]
            forecasts.append(max(pred, 0))
            
            # Update last_data for next iteration
            last_data = last_data.copy()
            # Update lags in reverse order (highest to lowest)
            max_lag = max([int(feat.split('_')[1]) for feat in available_features if feat.startswith('Lag_')], default=0)
            for lag in range(max_lag, 0, -1):
                lag_col = f'Lag_{lag}'
                if lag_col in last_data.columns:
                    if lag == 1:
                        last_data[lag_col] = pred
                    else:
                        prev_lag = f'Lag_{lag-1}'
                        if prev_lag in last_data.columns:
                            last_data[lag_col] = last_data[prev_lag].iloc[0]
            last_data['Sales'] = pred
            last_data['Month'] = next_month
            if 'Trend' in last_data.columns:
                last_data['Trend'] = next_trend
            # Update rolling means (simplified - use exponential smoothing)
            for feat in available_features:
                if feat.startswith('Rolling_Mean_'):
                    window = int(feat.split('_')[2])
                    if feat in last_data.columns:
                        # Simple exponential smoothing for rolling mean
                        alpha = 2.0 / (window + 1)
                        old_mean = last_data[feat].iloc[0] if not pd.isna(last_data[feat].iloc[0]) else pred
                        last_data[feat] = alpha * pred + (1 - alpha) * old_mean
        
        return forecasts, {'feature_importance': feature_importance, 'model': rf_model}
    except Exception as e:
        st.warning(f"Random Forest model error: {str(e)}")
        return naive_forecast(train_data, forecast_horizon)

def xgboost_forecast(train_data, forecast_horizon=3, n_estimators=100, 
                    max_depth=3, learning_rate=0.1, feature_config=None):
    """XGBoost forecasting with customizable parameters"""
    try:
        # Prepare features
        if feature_config:
            df_features = create_ml_features(train_data, **feature_config)
        else:
            df_features = create_ml_features(train_data)
        
        df_features = df_features.dropna()
        
        if len(df_features) < 3:
            return naive_forecast(train_data, forecast_horizon)
        
        # Select feature columns
        feature_cols = [col for col in df_features.columns 
                       if col not in ['Date', 'Sales', 'Year']]
        available_features = [col for col in feature_cols if col in df_features.columns]
        
        X_train = df_features[available_features]
        y_train = df_features['Sales']
        
        # Train model
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = dict(zip(available_features, xgb_model.feature_importances_))
        
        # Generate forecasts recursively (similar to RF)
        forecasts = []
        last_data = df_features.iloc[-1:].copy()
        
        for i in range(forecast_horizon):
            next_month = (last_data['Month'].iloc[0] % 12) + 1
            next_quarter = ((next_month - 1) // 3) + 1
            next_trend = last_data['Trend'].iloc[0] + 1 if 'Trend' in last_data.columns else i + len(df_features)
            
            pred_features = {}
            for feat in available_features:
                if feat.startswith('Lag_'):
                    lag_num = int(feat.split('_')[1])
                    if lag_num == 1:
                        pred_features[feat] = [last_data['Sales'].iloc[0]]
                    else:
                        prev_lag = f'Lag_{lag_num-1}'
                        pred_features[feat] = [last_data[prev_lag].iloc[0] if prev_lag in last_data.columns else 0]
                elif feat == 'Month':
                    pred_features[feat] = [next_month]
                elif feat == 'Quarter':
                    pred_features[feat] = [next_quarter]
                elif feat == 'Trend':
                    pred_features[feat] = [next_trend]
                elif feat == 'Trend_Squared':
                    pred_features[feat] = [next_trend ** 2]
                elif feat.startswith('Is_Q'):
                    q_num = int(feat.split('_')[1][1])
                    pred_features[feat] = [1 if next_quarter == q_num else 0]
                elif feat == 'Month_Sin':
                    pred_features[feat] = [np.sin(2 * np.pi * next_month / 12)]
                elif feat == 'Month_Cos':
                    pred_features[feat] = [np.cos(2 * np.pi * next_month / 12)]
                elif feat.startswith('Rolling_'):
                    pred_features[feat] = [last_data[feat].iloc[0] if feat in last_data.columns else last_data['Sales'].mean()]
                else:
                    pred_features[feat] = [last_data[feat].iloc[0] if feat in last_data.columns else 0]
            
            pred_df = pd.DataFrame(pred_features)
            pred = xgb_model.predict(pred_df)[0]
            forecasts.append(max(pred, 0))
            
            # Update last_data for next iteration
            last_data = last_data.copy()
            # Update lags in reverse order (highest to lowest)
            max_lag = max([int(feat.split('_')[1]) for feat in available_features if feat.startswith('Lag_')], default=0)
            for lag in range(max_lag, 0, -1):
                lag_col = f'Lag_{lag}'
                if lag_col in last_data.columns:
                    if lag == 1:
                        last_data[lag_col] = pred
                    else:
                        prev_lag = f'Lag_{lag-1}'
                        if prev_lag in last_data.columns:
                            last_data[lag_col] = last_data[prev_lag].iloc[0]
            last_data['Sales'] = pred
            last_data['Month'] = next_month
            if 'Trend' in last_data.columns:
                last_data['Trend'] = next_trend
            # Update rolling means (simplified - use exponential smoothing)
            for feat in available_features:
                if feat.startswith('Rolling_Mean_'):
                    window = int(feat.split('_')[2])
                    if feat in last_data.columns:
                        # Simple exponential smoothing for rolling mean
                        alpha = 2.0 / (window + 1)
                        old_mean = last_data[feat].iloc[0] if not pd.isna(last_data[feat].iloc[0]) else pred
                        last_data[feat] = alpha * pred + (1 - alpha) * old_mean
        
        return forecasts, {'feature_importance': feature_importance, 'model': xgb_model}
    except Exception as e:
        st.warning(f"XGBoost model error: {str(e)}")
        return naive_forecast(train_data, forecast_horizon)

# ==================== HYPERPARAMETER TUNING ====================

def tune_random_forest(X_train, y_train, cv_splits=3, n_iter=20):
    """Hyperparameter tuning for Random Forest using RandomizedSearchCV"""
    try:
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_base = RandomForestRegressor(random_state=42)
        
        # Use MAPE as scoring metric
        mape_scorer = make_scorer(lambda y_true, y_pred: -calculate_mape(y_true, y_pred), 
                                  greater_is_better=False)
        
        tscv = TimeSeriesSplit(n_splits=min(cv_splits, len(X_train) // 3))
        
        random_search = RandomizedSearchCV(
            rf_base,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=tscv,
            scoring=mape_scorer,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        return random_search.best_params_, random_search.best_estimator_, random_search.best_score_
    except Exception as e:
        return None, None, None

def tune_xgboost(X_train, y_train, cv_splits=3, n_iter=20):
    """Hyperparameter tuning for XGBoost using RandomizedSearchCV"""
    try:
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [2, 3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_base = xgb.XGBRegressor(random_state=42)
        
        # Use MAPE as scoring metric
        mape_scorer = make_scorer(lambda y_true, y_pred: -calculate_mape(y_true, y_pred), 
                                  greater_is_better=False)
        
        tscv = TimeSeriesSplit(n_splits=min(cv_splits, len(X_train) // 3))
        
        random_search = RandomizedSearchCV(
            xgb_base,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=tscv,
            scoring=mape_scorer,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        return random_search.best_params_, random_search.best_estimator_, random_search.best_score_
    except Exception as e:
        return None, None, None

def tune_arima(train_data, max_p=3, max_d=2, max_q=3):
    """ARIMA hyperparameter tuning using grid search"""
    try:
        best_aic = np.inf
        best_params = None
        best_model = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(train_data['Sales'], order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                            best_model = fitted_model
                    except:
                        continue
        
        return best_params, best_model, best_aic
    except Exception as e:
        return None, None, None

# ==================== ENSEMBLE MODELS ====================

def weighted_average_ensemble(forecasts_dict, weights=None):
    """Weighted average ensemble of forecasts"""
    if weights is None:
        # Equal weights
        weights = {model: 1.0 / len(forecasts_dict) for model in forecasts_dict.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Get forecast length
    forecast_length = len(list(forecasts_dict.values())[0])
    
    # Calculate weighted average
    ensemble_forecast = []
    for i in range(forecast_length):
        weighted_sum = sum(forecasts_dict[model][i] * weights[model] 
                          for model in forecasts_dict.keys())
        ensemble_forecast.append(weighted_sum)
    
    return ensemble_forecast, weights

def inverse_error_weighted_ensemble(forecasts_dict, errors_dict):
    """Ensemble weighted by inverse of error (lower error = higher weight)"""
    # Calculate weights as inverse of MAPE
    weights = {}
    for model in forecasts_dict.keys():
        if model in errors_dict and errors_dict[model] > 0:
            weights[model] = 1.0 / errors_dict[model]
        else:
            weights[model] = 0.0
    
    return weighted_average_ensemble(forecasts_dict, weights)

def stacking_ensemble(train_data, test_data, models_dict, feature_config=None):
    """Stacking ensemble using meta-learner"""
    try:
        # Prepare features
        if feature_config:
            df_features = create_ml_features(train_data, **feature_config)
        else:
            df_features = create_ml_features(train_data)
        
        df_features = df_features.dropna()
        
        if len(df_features) < 3:
            return None, None
        
        feature_cols = [col for col in df_features.columns 
                       if col not in ['Date', 'Sales', 'Year']]
        available_features = [col for col in feature_cols if col in df_features.columns]
        
        X_train = df_features[available_features]
        y_train = df_features['Sales']
        
        # Train base models and get predictions
        base_predictions = []
        base_models = []
        
        for model_name, model_obj in models_dict.items():
            if model_name == "Random Forest":
                model_obj.fit(X_train, y_train)
                base_predictions.append(model_obj.predict(X_train).reshape(-1, 1))
                base_models.append((model_name, model_obj))
            elif model_name == "XGBoost":
                model_obj.fit(X_train, y_train)
                base_predictions.append(model_obj.predict(X_train).reshape(-1, 1))
                base_models.append((model_name, model_obj))
        
        if not base_predictions:
            return None, None
        
        # Stack predictions as features for meta-learner
        meta_features = np.hstack(base_predictions)
        
        # Train meta-learner (simple linear regression)
        from sklearn.linear_model import LinearRegression
        meta_learner = LinearRegression()
        meta_learner.fit(meta_features, y_train)
        
        return meta_learner, base_models
    except Exception as e:
        return None, None

# ==================== SUPPLY CHAIN FUNCTIONS ====================

def calculate_safety_stock(forecast, std_dev, service_level=0.95):
    """Calculate safety stock based on forecast and service level"""
    z_score = stats.norm.ppf(service_level)
    safety_stock = z_score * std_dev
    return safety_stock

def calculate_reorder_point(avg_demand, lead_time, safety_stock):
    """Calculate reorder point"""
    return (avg_demand * lead_time) + safety_stock

def calculate_eoq(demand, ordering_cost, holding_cost):
    """Calculate Economic Order Quantity"""
    if holding_cost == 0:
        return 0
    return np.sqrt((2 * demand * ordering_cost) / holding_cost)

# ==================== LLM INTEGRATION (GEMINI) ====================

def get_gemini_audience_specific_report(model_performance, forecast_results, data_summary, 
                                       supply_chain_metrics=None, audience="Management"):
    """Get LLM interpretation tailored to specific audience"""
    try:
        api_key = os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY', None)
        
        if not api_key:
            return None, "⚠️ LLM interpretation requires Gemini API key. Please set GEMINI_API_KEY in environment variables or Streamlit secrets."
        
        if not GEMINI_AVAILABLE:
            return None, "⚠️ Google Generative AI library not installed. Install with: pip install google-generativeai"
        
        genai.configure(api_key=api_key)
        
        # Try to get available models first, then use the best one for free tier
        model = None
        model_name_used = None
        
        try:
            # List available models
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            # Prefer models in this order (best for free tier first)
            preferred_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            
            for preferred_model in preferred_models:
                matching_models = [m for m in available_models 
                                  if preferred_model.replace('-', '') in m.replace('-', '').replace('models/', '').lower() 
                                  or preferred_model in m.replace('models/', '').lower()]
                if matching_models:
                    model_name_used = matching_models[0]
                    model = genai.GenerativeModel(model_name_used)
                    break
            
            if model is None and available_models:
                model_name_used = available_models[0]
                model = genai.GenerativeModel(model_name_used)
                
        except Exception as e:
            # Fallback: try common model names directly
            model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    model_name_used = model_name
                    break
                except:
                    continue
        
        if model is None:
            return None, "⚠️ Could not find a compatible Gemini model."
        
        # Audience-specific prompts
        audience_prompts = {
            "Management": """
You are writing a report for C-level executives and senior management. Focus on:
- High-level business insights and strategic implications
- Financial impact and ROI considerations
- Risk assessment and mitigation strategies
- Actionable recommendations for decision-making
- Key performance indicators and metrics
- Executive summary with clear takeaways
Avoid technical jargon. Use business language. Emphasize what matters for strategic planning and resource allocation.
""",
            "Technical": """
You are writing a report for data scientists, ML engineers, and technical teams. Focus on:
- Detailed model performance analysis and comparison
- Technical methodology and algorithms used
- Feature engineering and model architecture details
- Statistical significance and validation metrics
- Model limitations and assumptions
- Recommendations for model improvement
Include technical details, code considerations, and deep dive into model mechanics.
""",
            "Supply Chain": """
You are writing a report for supply chain managers, operations teams, and logistics professionals. Focus on:
- Inventory planning and optimization recommendations
- Production scheduling and capacity planning
- Safety stock and reorder point analysis
- Distribution and logistics implications
- Demand variability and supply chain risks
- Operational efficiency improvements
- Cost optimization opportunities
Use supply chain terminology and focus on operational execution.
"""
        }
        
        audience_context = audience_prompts.get(audience, audience_prompts["Management"])
        
        # Prepare comprehensive prompt
        prompt = f"""
You are an expert consultant analyzing sales forecasting results for a beverage company launching a new flavored water product.

**AUDIENCE:** {audience}
{audience_context}

**DATA SUMMARY:**
- Total months of historical data: {data_summary['total_months']}
- Average monthly sales: ${data_summary['avg_sales']:,.0f}
- Sales trend: {data_summary['trend']}
- Seasonality detected: {data_summary['has_seasonality']}
- Data variance: {data_summary.get('variance', 'N/A')}

**MODEL PERFORMANCE METRICS:**
{json.dumps(model_performance, indent=2)}

**FORECAST RESULTS:**
- Best performing model: {forecast_results.get('best_model', 'N/A')}
- Next 3 months forecasts: {forecast_results.get('best_forecasts', [])}
- Forecast confidence: {forecast_results.get('confidence', 'N/A')}

**SUPPLY CHAIN METRICS:**
{supply_chain_metrics if supply_chain_metrics else 'Not calculated'}

Generate a comprehensive, audience-appropriate report (1000-1200 words) that addresses the specific needs and concerns of {audience} professionals. 
Use clear sections, professional formatting, and actionable insights relevant to this audience.
"""
        
        # Generate content with error handling
        try:
            response = model.generate_content(prompt)
            return response.text, None
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                return None, f"⚠️ Model not available. Error: {error_msg}."
            elif "quota" in error_msg.lower() or "429" in error_msg:
                return None, f"⚠️ API quota exceeded. Error: {error_msg}."
            else:
                return None, f"⚠️ Error generating report: {error_msg}."
        
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "authentication" in error_msg.lower():
            return None, f"⚠️ Authentication error: {error_msg}."
        else:
            return None, f"⚠️ Error generating report: {error_msg}."

def get_gemini_interpretation(model_performance, forecast_results, data_summary, 
                              supply_chain_metrics=None):
    """Get LLM interpretation using Google Gemini"""
    try:
        api_key = os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY', None)
        
        if not api_key:
            return None, "⚠️ LLM interpretation requires Gemini API key. Please set GEMINI_API_KEY in environment variables or Streamlit secrets."
        
        if not GEMINI_AVAILABLE:
            return None, "⚠️ Google Generative AI library not installed. Install with: pip install google-generativeai"
        
        genai.configure(api_key=api_key)
        
        # Try to get available models first, then use the best one for free tier
        model = None
        model_name_used = None
        
        try:
            # List available models
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    model_display_name = m.display_name or m.name
                    available_models.append(m.name)
            
            # Prefer models in this order (best for free tier first)
            preferred_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            
            for preferred_model in preferred_models:
                # Check if model name matches (handle different naming formats like 'models/gemini-1.5-flash')
                matching_models = [m for m in available_models 
                                  if preferred_model.replace('-', '') in m.replace('-', '').replace('models/', '').lower() 
                                  or preferred_model in m.replace('models/', '').lower()]
                if matching_models:
                    model_name_used = matching_models[0]
                    # Use the full model name from API (might include 'models/' prefix)
                    model = genai.GenerativeModel(model_name_used)
                    break
            
            # If no preferred model found, use first available
            if model is None and available_models:
                model_name_used = available_models[0]
                model = genai.GenerativeModel(model_name_used)
                
        except Exception as e:
            # Fallback: try common model names directly
            model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    model_name_used = model_name
                    break
                except:
                    continue
        
        if model is None:
            return None, "⚠️ Could not find a compatible Gemini model. Please check your API key, quota tier, and ensure you have access to Gemini models. Free tier typically supports 'gemini-1.5-flash'."
        
        # Prepare comprehensive prompt
        prompt = f"""
You are an expert data science and supply chain consultant analyzing sales forecasting results for a beverage company launching a new flavored water product.

**DATA SUMMARY:**
- Total months of historical data: {data_summary['total_months']}
- Average monthly sales: ${data_summary['avg_sales']:,.0f}
- Sales trend: {data_summary['trend']}
- Seasonality detected: {data_summary['has_seasonality']}
- Data variance: {data_summary.get('variance', 'N/A')}

**MODEL PERFORMANCE METRICS:**
{json.dumps(model_performance, indent=2)}

**FORECAST RESULTS:**
- Best performing model: {forecast_results.get('best_model', 'N/A')}
- Next 3 months forecasts: {forecast_results.get('best_forecasts', [])}
- Forecast confidence: {forecast_results.get('confidence', 'N/A')}

**SUPPLY CHAIN METRICS:**
{supply_chain_metrics if supply_chain_metrics else 'Not calculated'}

Please provide a comprehensive analysis report covering:

1. **MODEL PERFORMANCE ANALYSIS**
   - Which model performed best and detailed explanation of why
   - Comparison of model strengths and weaknesses
   - Statistical significance of results

2. **FORECAST ACCURACY ASSESSMENT**
   - How well did models capture seasonality and trends?
   - Confidence level in forecasts
   - Potential sources of error

3. **SUPPLY CHAIN & OPERATIONS RECOMMENDATIONS**
   - Production planning recommendations
   - Inventory management strategies
   - Distribution planning
   - Risk mitigation strategies

4. **BUSINESS INSIGHTS**
   - Key takeaways for stakeholders
   - Actionable recommendations
   - Expected business impact

5. **LIMITATIONS & RISKS**
   - Model limitations
   - Data quality concerns
   - External factors to consider

Format the response professionally with clear sections and bullet points. Keep it comprehensive but concise (800-1000 words).
"""
        
        # Generate content with error handling
        try:
            response = model.generate_content(prompt)
            return response.text, None
        except Exception as e:
            error_msg = str(e)
            # Provide helpful error message
            if "404" in error_msg or "not found" in error_msg.lower():
                return None, f"⚠️ Model not available. Error: {error_msg}. Try using gemini-1.5-flash or gemini-1.5-pro. Check your API quota tier."
            elif "quota" in error_msg.lower() or "429" in error_msg:
                return None, f"⚠️ API quota exceeded. Error: {error_msg}. Please check your Gemini API quota limits."
            else:
                return None, f"⚠️ Error generating LLM interpretation: {error_msg}. Please check your API key configuration."
        
    except Exception as e:
        error_msg = str(e)
        # Try to provide helpful debugging info
        if "API key" in error_msg or "authentication" in error_msg.lower():
            return None, f"⚠️ Authentication error: {error_msg}. Please verify your GEMINI_API_KEY is correct."
        else:
            return None, f"⚠️ Error generating LLM interpretation: {error_msg}. Please check your API key configuration and quota tier."

def convert_to_serializable(obj):
    """Convert numpy types and other non-serializable objects to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif pd.isna(obj):
        return None
    elif hasattr(obj, '__dict__'):
        # Skip complex objects like sklearn models
        return str(type(obj).__name__)
    else:
        return obj

def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = PILImage.open(buf)
    return img

def generate_pdf_report(llm_text, model_performance, forecast_results, data_summary, 
                       supply_chain_metrics, audience, figures_dict=None):
    """Generate PDF report with visualizations"""
    if not PDF_AVAILABLE:
        return None, "PDF generation libraries not installed. Install with: pip install reportlab Pillow"
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    elements.append(Paragraph("Sales Forecasting Analysis Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"<b>Audience:</b> {audience}", styles['Normal']))
    elements.append(Paragraph(f"<b>Generated on:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    elements.append(Paragraph(
        "This report presents a comprehensive analysis of sales forecasting for a new beverage product "
        "using multiple forecasting models, tailored for " + audience.lower() + " professionals.",
        styles['Normal']
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    # Data Overview
    elements.append(Paragraph("Data Overview", heading_style))
    data_text = f"""
    <b>Total months of data:</b> {data_summary['total_months']}<br/>
    <b>Average monthly sales:</b> ${data_summary['avg_sales']:,.0f}<br/>
    <b>Sales trend:</b> {data_summary['trend']}<br/>
    <b>Seasonality:</b> {data_summary['has_seasonality']}<br/>
    """
    elements.append(Paragraph(data_text, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Model Performance
    elements.append(Paragraph("Model Performance Metrics", heading_style))
    perf_data = [['Model'] + list(list(model_performance.values())[0].keys())]
    for model_name, metrics in model_performance.items():
        row = [model_name] + [str(v) for v in metrics.values()]
        perf_data.append(row)
    
    perf_table = Table(perf_data)
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(perf_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Forecast Results
    elements.append(Paragraph("Forecast Results", heading_style))
    forecast_text = ""
    for model_name, model_data in forecast_results.items():
        if 'future_forecast' in model_data and model_data['future_forecast']:
            forecasts = [float(f) for f in model_data['future_forecast']]
            forecast_text += f"<b>{model_name}:</b> {', '.join([f'${f:,.2f}' for f in forecasts])}<br/>"
    elements.append(Paragraph(forecast_text or "No forecasts available", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Add visualizations if available
    if figures_dict:
        elements.append(Paragraph("Visualizations", heading_style))
        for fig_name, fig in figures_dict.items():
            if fig:
                try:
                    img = fig_to_image(fig)
                    img_width = 6 * inch
                    img_height = img.height * (img_width / img.width)
                    # Limit height
                    if img_height > 4 * inch:
                        img_height = 4 * inch
                        img_width = img.width * (img_height / img.height)
                    
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    elements.append(Paragraph(f"<b>{fig_name}</b>", styles['Normal']))
                    elements.append(Image(img_buffer, width=img_width, height=img_height))
                    elements.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    elements.append(Paragraph(f"Could not include {fig_name}: {str(e)}", styles['Normal']))
    
    # Supply Chain Metrics
    if supply_chain_metrics:
        elements.append(Paragraph("Supply Chain Metrics", heading_style))
        sc_text = "<br/>".join([f"<b>{k}:</b> {v}" for k, v in supply_chain_metrics.items()])
        elements.append(Paragraph(sc_text, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
    
    # AI-Generated Analysis
    elements.append(PageBreak())
    elements.append(Paragraph("AI-Generated Analysis", heading_style))
    
    # Split LLM text into paragraphs
    llm_paragraphs = llm_text.split('\n\n')
    for para in llm_paragraphs:
        para = para.strip()
        if para:
            # Check if it's a heading
            if para.startswith('**') and para.endswith('**'):
                para = para.replace('**', '')
                elements.append(Paragraph(f"<b>{para}</b>", styles['Heading3']))
            elif para.startswith('#'):
                para = para.replace('#', '').strip()
                elements.append(Paragraph(f"<b>{para}</b>", styles['Heading3']))
            else:
                # Clean markdown formatting
                para = para.replace('**', '<b>').replace('**', '</b>')
                para = para.replace('*', '').replace('_', '')
                elements.append(Paragraph(para, styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue(), None

def generate_downloadable_report(llm_text, model_performance, forecast_results, 
                                 data_summary, supply_chain_metrics):
    """Generate downloadable report"""
    # Convert forecast_results to serializable format
    serializable_forecast_results = {}
    for model_name, model_data in forecast_results.items():
        serializable_forecast_results[model_name] = {}
        for key, value in model_data.items():
            if key == 'model_info':
                # Skip model_info as it contains sklearn models
                serializable_forecast_results[model_name][key] = "Model object (skipped in report)"
            elif key in ['test_forecast', 'future_forecast']:
                # Convert numpy arrays to lists
                serializable_forecast_results[model_name][key] = convert_to_serializable(value)
            elif key in ['mape', 'wape', 'rmse', 'mae', 'r2']:
                # Convert numpy floats
                if pd.isna(value) or np.isnan(value):
                    serializable_forecast_results[model_name][key] = None
                else:
                    serializable_forecast_results[model_name][key] = float(value)
            elif key == 'params_used':
                # Convert params_used (may contain numpy types)
                serializable_forecast_results[model_name][key] = convert_to_serializable(value)
            else:
                serializable_forecast_results[model_name][key] = convert_to_serializable(value)
    
    # Format forecast results as readable text instead of JSON
    forecast_text = ""
    for model_name, model_data in serializable_forecast_results.items():
        forecast_text += f"\n### {model_name}\n"
        
        mape_val = model_data.get('mape')
        forecast_text += f"- MAPE: {mape_val:.2f}%\n" if mape_val is not None and not pd.isna(mape_val) else "- MAPE: N/A\n"
        
        wape_val = model_data.get('wape')
        forecast_text += f"- WAPE: {wape_val:.2f}%\n" if wape_val is not None and not pd.isna(wape_val) else "- WAPE: N/A\n"
        
        rmse_val = model_data.get('rmse')
        forecast_text += f"- RMSE: ${rmse_val:.2f}\n" if rmse_val is not None and not pd.isna(rmse_val) else "- RMSE: N/A\n"
        
        mae_val = model_data.get('mae')
        forecast_text += f"- MAE: ${mae_val:.2f}\n" if mae_val is not None and not pd.isna(mae_val) else "- MAE: N/A\n"
        
        r2_val = model_data.get('r2')
        forecast_text += f"- R²: {r2_val:.3f}\n" if r2_val is not None and not pd.isna(r2_val) else "- R²: N/A\n"
        
        if 'future_forecast' in model_data and model_data['future_forecast']:
            try:
                forecasts = [float(f) for f in model_data['future_forecast']]
                forecast_text += f"- Future Forecasts: {', '.join([f'${f:,.2f}' for f in forecasts])}\n"
            except (ValueError, TypeError):
                forecast_text += f"- Future Forecasts: {model_data['future_forecast']}\n"
        forecast_text += "\n"
    
    report = f"""
# Sales Forecasting Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report presents a comprehensive analysis of sales forecasting for a new beverage product using multiple forecasting models.

## Data Overview
- Total months of data: {data_summary['total_months']}
- Average monthly sales: ${data_summary['avg_sales']:,.0f}
- Sales trend: {data_summary['trend']}
- Seasonality: {data_summary['has_seasonality']}

## Model Performance Metrics
{json.dumps(model_performance, indent=2)}

## Forecast Results
{forecast_text}

## Supply Chain Metrics
{json.dumps(supply_chain_metrics, indent=2) if supply_chain_metrics and isinstance(supply_chain_metrics, dict) else (supply_chain_metrics if supply_chain_metrics else 'Not calculated')}

## AI-Generated Analysis
{llm_text if llm_text else 'Not available'}

---
Report generated by Advanced Sales Forecasting System
"""
    return report

# ==================== VISUALIZATION FUNCTIONS ====================

def plot_time_series_decomposition(df):
    """Plot time series decomposition"""
    try:
        if len(df) < 12:
            return None
        
        df_ts = df.set_index('Date')['Sales']
        decomposition = seasonal_decompose(df_ts, model='additive', period=12)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        decomposition.observed.plot(ax=axes[0], title='Original', color='blue')
        decomposition.trend.plot(ax=axes[1], title='Trend', color='green')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='orange')
        decomposition.resid.plot(ax=axes[3], title='Residual', color='red')
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    except:
        return None

def plot_residuals(actual, predicted, model_name):
    """Plot residual analysis"""
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals over time
    axes[0, 0].plot(residuals, 'o-', alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title(f'{model_name} - Residuals Over Time')
    axes[0, 0].set_xlabel('Observation')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title(f'{model_name} - Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'{model_name} - Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    axes[1, 1].scatter(predicted, residuals, alpha=0.7)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_title(f'{model_name} - Residuals vs Predicted')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_importance, model_name):
    """Plot feature importance for ML models"""
    if not feature_importance:
        return None
    
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    features = [features[i] for i in sorted_idx[:10]]  # Top 10
    importance = [importance[i] for i in sorted_idx[:10]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features, importance)
    ax.set_xlabel('Importance')
    ax.set_title(f'{model_name} - Feature Importance (Top 10)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return fig

# ==================== MAIN APP ====================

st.markdown('<h1 class="main-header">📊 Advanced Sales Forecasting System</h1>', unsafe_allow_html=True)
st.markdown("### Comprehensive Demand Forecasting with Supply Chain Integration")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Data Input Section
st.sidebar.subheader("📥 Data Input")
data_source = st.sidebar.radio("Data Source", ["Generate Sample", "Upload CSV"])

df = None

if data_source == "Generate Sample":
    st.sidebar.subheader("Sample Data Configuration")
    n_months = st.sidebar.slider("Number of Months", 6, 36, 12)
    base_sales = st.sidebar.number_input("Base Sales ($)", 5000, 50000, 10000)
    trend_strength = st.sidebar.slider("Trend Strength", 0.0, 0.5, 0.15, 0.01)
    seasonality_strength = st.sidebar.slider("Seasonality Strength", 0.0, 0.5, 0.2, 0.01)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 0.2, 0.05, 0.01)
    start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
    
    if st.sidebar.button("Generate Sample Data"):
        df = generate_advanced_sample_data(
            n_months=n_months,
            base_sales=base_sales,
            trend_strength=trend_strength,
            seasonality_strength=seasonality_strength,
            noise_level=noise_level,
            start_date=start_date.strftime('%Y-%m-%d')
        )
        st.session_state.df = df
        st.sidebar.success("✅ Sample data generated!")
    
    if 'df' in st.session_state:
        df = st.session_state.df

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Auto-detect columns
            date_col = None
            for col in df.columns:
                if 'date' in col.lower() or 'month' in col.lower():
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.rename(columns={date_col: 'Date'})
            else:
                st.error("Please ensure your CSV has a date column")
            
            sales_col = None
            for col in df.columns:
                if 'sales' in col.lower() or 'demand' in col.lower():
                    sales_col = col
                    break
            
            if sales_col:
                df = df.rename(columns={sales_col: 'Sales'})
            else:
                st.error("Please ensure your CSV has a sales/demand column")
            
            df = df[['Date', 'Sales']].copy()
            df = df.sort_values('Date')
            st.session_state.df = df
            st.sidebar.success("✅ Data loaded successfully")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# Model Selection
st.sidebar.subheader("🤖 Model Selection")
selected_models = st.sidebar.multiselect(
    "Choose forecasting models",
    ["Naïve", "ARIMA", "Random Forest", "XGBoost"],
    default=["Naïve", "ARIMA", "Random Forest", "XGBoost"]
)

# Forecast Settings
st.sidebar.subheader("📈 Forecast Settings")
forecast_horizon = st.sidebar.slider("Forecast Horizon (months)", 1, 12, 3)
train_split = st.sidebar.slider("Training Data (%)", 60, 90, 80) / 100

# Model Parameters (Expandable)
st.sidebar.subheader("🔧 Model Parameters")

# ARIMA Parameters
arima_expander = st.sidebar.expander("ARIMA Parameters", expanded=False)
arima_p = arima_expander.slider("ARIMA p (AR)", 0, 3, 1)
arima_d = arima_expander.slider("ARIMA d (I)", 0, 2, 1)
arima_q = arima_expander.slider("ARIMA q (MA)", 0, 3, 1)

# Random Forest Parameters
rf_expander = st.sidebar.expander("Random Forest Parameters", expanded=False)
rf_n_estimators = rf_expander.slider("RF: n_estimators", 50, 300, 100)
rf_max_depth = rf_expander.slider("RF: max_depth", 3, 15, 5)
rf_min_samples_split = rf_expander.slider("RF: min_samples_split", 2, 10, 2)

# XGBoost Parameters
xgb_expander = st.sidebar.expander("XGBoost Parameters", expanded=False)
xgb_n_estimators = xgb_expander.slider("XGB: n_estimators", 50, 300, 100)
xgb_max_depth = xgb_expander.slider("XGB: max_depth", 2, 10, 3)
xgb_learning_rate = xgb_expander.slider("XGB: learning_rate", 0.01, 0.3, 0.1, 0.01)

# Feature Engineering Options
st.sidebar.subheader("🔨 Feature Engineering")

feature_expander = st.sidebar.expander("ML Model Features", expanded=False)
ml_lags = feature_expander.multiselect("Lag Features", [1,2,3,4,5,6], default=[1,2,3])
ml_rolling = feature_expander.multiselect("Rolling Windows", [3,6,9,12], default=[3,6])
ml_seasonal = feature_expander.checkbox("Seasonal Features", value=True)
ml_trend = feature_expander.checkbox("Trend Features", value=True)

feature_config = {
    'lag_features': ml_lags,
    'rolling_windows': ml_rolling,
    'seasonal_features': ml_seasonal,
    'trend_features': ml_trend
}

# Hyperparameter Tuning Options
st.sidebar.subheader("🎯 Hyperparameter Tuning")
enable_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=False)
if enable_tuning:
    tuning_method = st.sidebar.radio("Tuning Method", ["Randomized Search", "Grid Search (ARIMA only)"])
    cv_splits = st.sidebar.slider("CV Splits", 2, 5, 3)
    n_iter = st.sidebar.slider("Random Search Iterations", 10, 50, 20)
    tune_arima_model = st.sidebar.checkbox("Tune ARIMA", value=True)
    tune_rf_model = st.sidebar.checkbox("Tune Random Forest", value=True)
    tune_xgb_model = st.sidebar.checkbox("Tune XGBoost", value=True)

# Ensemble Model Options
st.sidebar.subheader("🎭 Ensemble Models")
enable_ensemble = st.sidebar.checkbox("Enable Ensemble Models", value=False)
if enable_ensemble:
    ensemble_method = st.sidebar.selectbox(
        "Ensemble Method",
        ["Weighted Average (Equal)", "Weighted Average (Inverse Error)", "Stacking"]
    )

# Supply Chain Settings
st.sidebar.subheader("📦 Supply Chain Settings")
enable_supply_chain = st.sidebar.checkbox("Enable Supply Chain Analysis", value=True)
if enable_supply_chain:
    service_level = st.sidebar.slider("Service Level", 0.80, 0.99, 0.95, 0.01)
    lead_time = st.sidebar.number_input("Lead Time (months)", 0.5, 3.0, 1.0, 0.1)
    ordering_cost = st.sidebar.number_input("Ordering Cost ($)", 100, 10000, 500)
    holding_cost_rate = st.sidebar.slider("Holding Cost Rate (%)", 0.1, 5.0, 1.0, 0.1) / 100

# Main Content Area
if df is not None:
    # Data Overview Tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🔬 Model Training", 
                                            "📈 Visualizations", "📦 Supply Chain", "🤖 AI Analysis"])
    
    with tab1:
        st.subheader("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Average Sales", f"${df['Sales'].mean():,.0f}")
        with col3:
            st.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
        with col4:
            st.metric("Std Deviation", f"${df['Sales'].std():,.0f}")
        
        # Data table
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True)
        
        # Time series decomposition
        st.subheader("Time Series Decomposition")
        decomp_fig = plot_time_series_decomposition(df)
        if decomp_fig:
            st.pyplot(decomp_fig)
        else:
            st.info("Need at least 12 months of data for decomposition")
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df['Sales'].describe(), use_container_width=True)
        
        # Data summary for later use
        data_summary = {
            'total_months': len(df),
            'avg_sales': df['Sales'].mean(),
            'trend': 'Increasing' if df['Sales'].iloc[-1] > df['Sales'].iloc[0] else 'Decreasing',
            'has_seasonality': 'Yes' if len(df) >= 12 else 'Unknown',
            'variance': f"${df['Sales'].var():,.0f}"
        }
    
    # Train-test split
    split_idx = int(len(df) * train_split)
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    
    with tab2:
        st.subheader("Model Training & Evaluation")
        
        # Display current parameters being used
        if len(selected_models) > 0:
            with st.expander("📋 Current Model Parameters", expanded=False):
                param_display = {}
                if "ARIMA" in selected_models:
                    param_display["ARIMA"] = f"p={arima_p}, d={arima_d}, q={arima_q}"
                if "Random Forest" in selected_models:
                    param_display["Random Forest"] = f"n_estimators={rf_n_estimators}, max_depth={rf_max_depth}, min_samples_split={rf_min_samples_split}"
                if "XGBoost" in selected_models:
                    param_display["XGBoost"] = f"n_estimators={xgb_n_estimators}, max_depth={xgb_max_depth}, learning_rate={xgb_learning_rate}"
                
                for model, params in param_display.items():
                    st.text(f"{model}: {params}")
        
        if len(selected_models) > 0:
            if st.button("🚀 Train Models", type="primary"):
                results = {}
                performance = {}
                tuned_params = {}
                base_models_dict = {}
                
                # Prepare ML features for tuning if needed
                ml_features_ready = False
                X_train_ml = None
                y_train_ml = None
                available_features_ml = None
                
                if enable_tuning and ("Random Forest" in selected_models or "XGBoost" in selected_models):
                    try:
                        if feature_config:
                            df_features = create_ml_features(train_data, **feature_config)
                        else:
                            df_features = create_ml_features(train_data)
                        df_features = df_features.dropna()
                        
                        if len(df_features) >= 3:
                            feature_cols = [col for col in df_features.columns 
                                           if col not in ['Date', 'Sales', 'Year']]
                            available_features_ml = [col for col in feature_cols if col in df_features.columns]
                            X_train_ml = df_features[available_features_ml]
                            y_train_ml = df_features['Sales']
                            ml_features_ready = True
                    except Exception as e:
                        st.warning(f"Could not prepare features for tuning: {str(e)}")
                
                # Train each model
                for model_name in selected_models:
                    with st.spinner(f"Training {model_name} model..."):
                        model_params_used = {}
                        
                        if model_name == "Naïve":
                            train_forecast, model_info = naive_forecast(train_data, len(test_data))
                            future_forecast, _ = naive_forecast(train_data, forecast_horizon)
                            model_params_used = {"method": "Last value"}
                        
                        elif model_name == "ARIMA":
                            # Hyperparameter tuning for ARIMA
                            if enable_tuning and tune_arima_model:
                                with st.spinner(f"Tuning {model_name} hyperparameters..."):
                                    best_params, best_model, best_aic = tune_arima(train_data)
                                    if best_params:
                                        tuned_params[model_name] = best_params
                                        arima_p_tuned, arima_d_tuned, arima_q_tuned = best_params
                                        st.info(f"🎯 Tuned ARIMA: p={arima_p_tuned}, d={arima_d_tuned}, q={arima_q_tuned} (AIC={best_aic:.2f})")
                                    else:
                                        arima_p_tuned, arima_d_tuned, arima_q_tuned = arima_p, arima_d, arima_q
                            else:
                                arima_p_tuned, arima_d_tuned, arima_q_tuned = arima_p, arima_d, arima_q
                            
                            model_params_used = {"p": arima_p_tuned, "d": arima_d_tuned, "q": arima_q_tuned}
                            
                            train_forecast, model_info = arima_forecast(
                                train_data, len(test_data), arima_p_tuned, arima_d_tuned, arima_q_tuned
                            )
                            future_forecast, _ = arima_forecast(
                                train_data, forecast_horizon, arima_p_tuned, arima_d_tuned, arima_q_tuned
                            )
                        
                        elif model_name == "Random Forest":
                            # Hyperparameter tuning for Random Forest
                            if enable_tuning and tune_rf_model and ml_features_ready:
                                with st.spinner(f"Tuning {model_name} hyperparameters..."):
                                    best_params, best_model, best_score = tune_random_forest(
                                        X_train_ml, y_train_ml, cv_splits, n_iter
                                    )
                                    if best_params:
                                        tuned_params[model_name] = best_params
                                        rf_n_estimators_tuned = best_params.get('n_estimators', rf_n_estimators)
                                        rf_max_depth_tuned = best_params.get('max_depth', rf_max_depth)
                                        rf_min_samples_split_tuned = best_params.get('min_samples_split', rf_min_samples_split)
                                        st.info(f"🎯 Tuned RF: {best_params}")
                                        base_models_dict[model_name] = best_model
                                    else:
                                        rf_n_estimators_tuned = rf_n_estimators
                                        rf_max_depth_tuned = rf_max_depth
                                        rf_min_samples_split_tuned = rf_min_samples_split
                            else:
                                rf_n_estimators_tuned = rf_n_estimators
                                rf_max_depth_tuned = rf_max_depth
                                rf_min_samples_split_tuned = rf_min_samples_split
                            
                            model_params_used = {
                                "n_estimators": rf_n_estimators_tuned,
                                "max_depth": rf_max_depth_tuned,
                                "min_samples_split": rf_min_samples_split_tuned
                            }
                            
                            train_forecast, model_info = random_forest_forecast(
                                train_data, len(test_data), rf_n_estimators_tuned, 
                                rf_max_depth_tuned, rf_min_samples_split_tuned, feature_config
                            )
                            future_forecast, _ = random_forest_forecast(
                                train_data, forecast_horizon, rf_n_estimators_tuned,
                                rf_max_depth_tuned, rf_min_samples_split_tuned, feature_config
                            )
                        
                        elif model_name == "XGBoost":
                            # Hyperparameter tuning for XGBoost
                            if enable_tuning and tune_xgb_model and ml_features_ready:
                                with st.spinner(f"Tuning {model_name} hyperparameters..."):
                                    best_params, best_model, best_score = tune_xgboost(
                                        X_train_ml, y_train_ml, cv_splits, n_iter
                                    )
                                    if best_params:
                                        tuned_params[model_name] = best_params
                                        xgb_n_estimators_tuned = best_params.get('n_estimators', xgb_n_estimators)
                                        xgb_max_depth_tuned = best_params.get('max_depth', xgb_max_depth)
                                        xgb_learning_rate_tuned = best_params.get('learning_rate', xgb_learning_rate)
                                        st.info(f"🎯 Tuned XGBoost: {best_params}")
                                        base_models_dict[model_name] = best_model
                                    else:
                                        xgb_n_estimators_tuned = xgb_n_estimators
                                        xgb_max_depth_tuned = xgb_max_depth
                                        xgb_learning_rate_tuned = xgb_learning_rate
                            else:
                                xgb_n_estimators_tuned = xgb_n_estimators
                                xgb_max_depth_tuned = xgb_max_depth
                                xgb_learning_rate_tuned = xgb_learning_rate
                            
                            model_params_used = {
                                "n_estimators": xgb_n_estimators_tuned,
                                "max_depth": xgb_max_depth_tuned,
                                "learning_rate": xgb_learning_rate_tuned
                            }
                            
                            train_forecast, model_info = xgboost_forecast(
                                train_data, len(test_data), xgb_n_estimators_tuned,
                                xgb_max_depth_tuned, xgb_learning_rate_tuned, feature_config
                            )
                            future_forecast, _ = xgboost_forecast(
                                train_data, forecast_horizon, xgb_n_estimators_tuned,
                                xgb_max_depth_tuned, xgb_learning_rate_tuned, feature_config
                            )
                        
                        # Evaluate on test data
                        if len(test_data) > 0 and len(train_forecast) > 0:
                            test_forecast = train_forecast[:len(test_data)]
                            mape = calculate_mape(test_data['Sales'].values, test_forecast)
                            wape = calculate_wape(test_data['Sales'].values, test_forecast)
                            rmse = calculate_rmse(test_data['Sales'].values, test_forecast)
                            mae = calculate_mae(test_data['Sales'].values, test_forecast)
                            r2 = calculate_r2(test_data['Sales'].values, test_forecast)
                        else:
                            mape = wape = rmse = mae = r2 = np.nan
                        
                        # Store results
                        results[model_name] = {
                            'test_forecast': test_forecast if len(test_data) > 0 else [],
                            'future_forecast': future_forecast,
                            'mape': mape,
                            'wape': wape,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'model_info': model_info,
                            'params_used': model_params_used
                        }
                        
                        # Performance metrics
                        perf_dict = {
                            'MAPE (%)': f"{mape:.2f}" if not np.isnan(mape) else "N/A",
                            'WAPE (%)': f"{wape:.2f}" if not np.isnan(wape) else "N/A",
                            'RMSE': f"${rmse:.2f}" if not np.isnan(rmse) else "N/A",
                            'MAE': f"${mae:.2f}" if not np.isnan(mae) else "N/A",
                            'R²': f"{r2:.3f}" if not np.isnan(r2) else "N/A"
                        }
                        
                        # Add ARIMA-specific metrics
                        if model_name == "ARIMA" and model_info and 'stats' in model_info:
                            perf_dict['AIC'] = f"{model_info['stats']['AIC']:.2f}"
                            perf_dict['BIC'] = f"{model_info['stats']['BIC']:.2f}"
                        
                        performance[model_name] = perf_dict
                
                # Ensemble Models
                if enable_ensemble and len(results) > 1:
                    with st.spinner("Creating ensemble model..."):
                        forecasts_dict = {name: results[name]['future_forecast'] for name in results.keys()}
                        errors_dict = {name: results[name]['mape'] for name in results.keys() 
                                     if not np.isnan(results[name]['mape'])}
                        
                        if ensemble_method == "Weighted Average (Equal)":
                            ensemble_forecast, ensemble_weights = weighted_average_ensemble(forecasts_dict)
                            ensemble_name = "Ensemble (Equal Weights)"
                        elif ensemble_method == "Weighted Average (Inverse Error)":
                            ensemble_forecast, ensemble_weights = inverse_error_weighted_ensemble(
                                forecasts_dict, errors_dict
                            )
                            ensemble_name = "Ensemble (Inverse Error Weights)"
                        elif ensemble_method == "Stacking":
                            # For stacking, we need to use the trained models
                            if base_models_dict:
                                meta_learner, base_models = stacking_ensemble(
                                    train_data, test_data, base_models_dict, feature_config
                                )
                                if meta_learner:
                                    # Generate stacking forecast (simplified)
                                    ensemble_forecast, ensemble_weights = weighted_average_ensemble(forecasts_dict)
                                    ensemble_name = "Ensemble (Stacking)"
                                else:
                                    ensemble_forecast, ensemble_weights = weighted_average_ensemble(forecasts_dict)
                                    ensemble_name = "Ensemble (Weighted Average)"
                            else:
                                ensemble_forecast, ensemble_weights = weighted_average_ensemble(forecasts_dict)
                                ensemble_name = "Ensemble (Weighted Average)"
                        
                        # Evaluate ensemble on test data
                        if len(test_data) > 0:
                            # Create ensemble test forecast
                            test_forecasts_dict = {name: results[name]['test_forecast'] 
                                                 for name in results.keys() 
                                                 if len(results[name]['test_forecast']) > 0}
                            if ensemble_method == "Weighted Average (Inverse Error)":
                                ensemble_test_forecast, _ = inverse_error_weighted_ensemble(
                                    test_forecasts_dict, errors_dict
                                )
                            else:
                                ensemble_test_forecast, _ = weighted_average_ensemble(test_forecasts_dict)
                            
                            ensemble_mape = calculate_mape(test_data['Sales'].values, ensemble_test_forecast)
                            ensemble_wape = calculate_wape(test_data['Sales'].values, ensemble_test_forecast)
                            ensemble_rmse = calculate_rmse(test_data['Sales'].values, ensemble_test_forecast)
                            ensemble_mae = calculate_mae(test_data['Sales'].values, ensemble_test_forecast)
                            ensemble_r2 = calculate_r2(test_data['Sales'].values, ensemble_test_forecast)
                        else:
                            ensemble_mape = ensemble_wape = ensemble_rmse = ensemble_mae = ensemble_r2 = np.nan
                        
                        # Store ensemble results
                        results[ensemble_name] = {
                            'test_forecast': ensemble_test_forecast if len(test_data) > 0 else [],
                            'future_forecast': ensemble_forecast,
                            'mape': ensemble_mape,
                            'wape': ensemble_wape,
                            'rmse': ensemble_rmse,
                            'mae': ensemble_mae,
                            'r2': ensemble_r2,
                            'model_info': {'weights': ensemble_weights},
                            'params_used': {'method': ensemble_method, 'weights': ensemble_weights}
                        }
                        
                        performance[ensemble_name] = {
                            'MAPE (%)': f"{ensemble_mape:.2f}" if not np.isnan(ensemble_mape) else "N/A",
                            'WAPE (%)': f"{ensemble_wape:.2f}" if not np.isnan(ensemble_wape) else "N/A",
                            'RMSE': f"${ensemble_rmse:.2f}" if not np.isnan(ensemble_rmse) else "N/A",
                            'MAE': f"${ensemble_mae:.2f}" if not np.isnan(ensemble_mae) else "N/A",
                            'R²': f"{ensemble_r2:.3f}" if not np.isnan(ensemble_r2) else "N/A"
                        }
                        
                        st.success(f"✅ Ensemble model created: {ensemble_method}")
                        st.json(ensemble_weights)
                
                st.session_state.forecast_results = results
                st.session_state.model_performance = performance
                st.session_state.tuned_params = tuned_params
                st.success("✅ Models trained successfully!")
            
            # Display performance metrics
            if st.session_state.forecast_results:
                st.subheader("📊 Model Performance Metrics")
                perf_df = pd.DataFrame(st.session_state.model_performance).T
                st.dataframe(perf_df, use_container_width=True)
                
                # Display tuned parameters if available
                if 'tuned_params' in st.session_state and st.session_state.tuned_params:
                    st.subheader("🎯 Tuned Hyperparameters")
                    for model_name, params in st.session_state.tuned_params.items():
                        st.success(f"**{model_name}**: {params}")
                
                # Display parameters used for each model
                st.subheader("⚙️ Parameters Used")
                params_df_data = []
                for model_name, result in st.session_state.forecast_results.items():
                    if 'params_used' in result:
                        params_str = ", ".join([f"{k}={v}" for k, v in result['params_used'].items()])
                        params_df_data.append({'Model': model_name, 'Parameters': params_str})
                
                if params_df_data:
                    params_df = pd.DataFrame(params_df_data)
                    st.dataframe(params_df, use_container_width=True)
                
                # Find best model
                valid_models = {k: v for k, v in st.session_state.forecast_results.items() 
                              if not np.isnan(v['mape'])}
                if valid_models:
                    best_model = min(valid_models.items(), key=lambda x: x[1]['mape'])[0]
                    st.success(f"🏆 Best Model: **{best_model}** (Lowest MAPE: {st.session_state.forecast_results[best_model]['mape']:.2f}%)")
                else:
                    best_model = selected_models[0] if selected_models else None
                
                # Model comparison chart
                st.subheader("Model Comparison")
                comparison_metrics = ['mape', 'wape', 'rmse', 'mae']
                comparison_data = []
                for model_name in selected_models:
                    if model_name in st.session_state.forecast_results:
                        model_data = st.session_state.forecast_results[model_name]
                        for metric in comparison_metrics:
                            if not np.isnan(model_data.get(metric, np.nan)):
                                comparison_data.append({
                                    'Model': model_name,
                                    'Metric': metric.upper(),
                                    'Value': model_data[metric]
                                })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    comp_df_pivot = comp_df.pivot(index='Model', columns='Metric', values='Value')
                    comp_df_pivot.plot(kind='bar', ax=ax)
                    ax.set_title('Model Performance Comparison')
                    ax.set_ylabel('Metric Value')
                    ax.legend(title='Metric')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.warning("Please select at least one model from the sidebar")
    
    with tab3:
        st.subheader("Forecast Visualizations")
        
        if st.session_state.forecast_results:
            # Include ensemble models in visualization
            models_to_plot = list(st.session_state.forecast_results.keys())
            results = st.session_state.forecast_results
            
            # Main forecast plot
            st.subheader("Forecast Comparison")
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Historical data
            ax.plot(df['Date'], df['Sales'], 'o-', label='Historical Data', 
                   color='blue', linewidth=2, markersize=6)
            
            # Test set predictions
            if len(test_data) > 0:
                for model_name in models_to_plot:
                    if model_name in results and len(results[model_name]['test_forecast']) > 0:
                        ax.plot(test_data['Date'], results[model_name]['test_forecast'],
                               '--', label=f'{model_name} (Test)', alpha=0.7, linewidth=2)
            
            # Future forecasts
            future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1),
                                        periods=forecast_horizon, freq='M')
            
            for model_name in models_to_plot:
                if model_name in results:
                    # Highlight ensemble models
                    if 'Ensemble' in model_name:
                        ax.plot(future_dates, results[model_name]['future_forecast'],
                               'o-', label=f'{model_name} (Forecast)', alpha=0.9, linewidth=3, markersize=10)
                    else:
                        ax.plot(future_dates, results[model_name]['future_forecast'],
                               'o--', label=f'{model_name} (Forecast)', alpha=0.8, linewidth=2, markersize=8)
            
            ax.set_title('Sales Forecast Comparison', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Sales ($)', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Residual analysis for each model
            if len(test_data) > 0:
                st.subheader("Residual Analysis")
                for model_name in selected_models:
                    if model_name in results and len(results[model_name]['test_forecast']) > 0:
                        with st.expander(f"{model_name} Residual Analysis"):
                            residual_fig = plot_residuals(
                                test_data['Sales'].values,
                                results[model_name]['test_forecast'],
                                model_name
                            )
                            if residual_fig:
                                st.pyplot(residual_fig)
            
            # Feature importance for ML models
            st.subheader("Feature Importance Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                if "Random Forest" in selected_models and "Random Forest" in results:
                    if results["Random Forest"].get('model_info') and 'feature_importance' in results["Random Forest"]['model_info']:
                        fi_fig = plot_feature_importance(
                            results["Random Forest"]["model_info"]["feature_importance"],
                            "Random Forest"
                        )
                        if fi_fig:
                            st.pyplot(fi_fig)
            
            with col2:
                if "XGBoost" in selected_models and "XGBoost" in results:
                    if results["XGBoost"].get('model_info') and 'feature_importance' in results["XGBoost"]['model_info']:
                        fi_fig = plot_feature_importance(
                            results["XGBoost"]["model_info"]["feature_importance"],
                            "XGBoost"
                        )
                        if fi_fig:
                            st.pyplot(fi_fig)
            
            # Forecast table
            st.subheader("Detailed Forecast Table")
            forecast_table_data = {'Month': [f"Month {i+1}" for i in range(forecast_horizon)]}
            for model_name in selected_models:
                if model_name in results:
                    forecast_table_data[model_name] = [
                        f"${f:,.2f}" for f in results[model_name]['future_forecast']
                    ]
            forecast_df = pd.DataFrame(forecast_table_data)
            st.dataframe(forecast_df, use_container_width=True)
    
    with tab4:
        st.subheader("Supply Chain & Operations Analysis")
        
        if enable_supply_chain and st.session_state.forecast_results:
            results = st.session_state.forecast_results
            valid_models = {k: v for k, v in results.items() if not np.isnan(v['mape'])}
            if valid_models:
                best_model_name = min(valid_models.items(), key=lambda x: x[1]['mape'])[0]
                best_forecasts = results[best_model_name]['future_forecast']
                
                # Calculate supply chain metrics
                avg_forecast = np.mean(best_forecasts)
                std_forecast = np.std(best_forecasts)
                
                safety_stock = calculate_safety_stock(avg_forecast, std_forecast, service_level)
                reorder_point = calculate_reorder_point(avg_forecast, lead_time, safety_stock)
                eoq = calculate_eoq(avg_forecast * 12, ordering_cost, holding_cost_rate * avg_forecast)
                
                supply_chain_metrics = {
                    'Average Forecasted Demand': f"${avg_forecast:,.2f}",
                    'Safety Stock': f"${safety_stock:,.2f}",
                    'Reorder Point': f"${reorder_point:,.2f}",
                    'Economic Order Quantity': f"{eoq:,.0f} units",
                    'Service Level': f"{service_level*100:.1f}%",
                    'Lead Time': f"{lead_time} months"
                }
                
                st.subheader("Supply Chain Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Monthly Demand", f"${avg_forecast:,.0f}")
                    st.metric("Safety Stock", f"${safety_stock:,.0f}")
                
                with col2:
                    st.metric("Reorder Point", f"${reorder_point:,.0f}")
                    st.metric("EOQ", f"{eoq:,.0f} units")
                
                with col3:
                    st.metric("Service Level", f"{service_level*100:.1f}%")
                    st.metric("Lead Time", f"{lead_time} months")
                
                # Inventory planning visualization
                st.subheader("Inventory Planning")
                months_ahead = list(range(1, forecast_horizon + 1))
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(months_ahead, best_forecasts, 'o-', label='Forecasted Demand', 
                       linewidth=2, markersize=8)
                ax.axhline(y=reorder_point, color='r', linestyle='--', 
                          label=f'Reorder Point (${reorder_point:,.0f})', linewidth=2)
                ax.axhline(y=safety_stock, color='orange', linestyle='--', 
                          label=f'Safety Stock (${safety_stock:,.0f})', linewidth=2)
                ax.fill_between(months_ahead, 
                               [max(0, f - safety_stock) for f in best_forecasts],
                               best_forecasts,
                               alpha=0.3, label='Safety Stock Buffer')
                ax.set_title('Inventory Planning Based on Forecasts', fontsize=14, fontweight='bold')
                ax.set_xlabel('Months Ahead')
                ax.set_ylabel('Units / Sales ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Production planning recommendations
                st.subheader("Production Planning Recommendations")
                
                total_forecast = sum(best_forecasts)
                monthly_avg = avg_forecast
                
                recommendations = []
                
                if monthly_avg > df['Sales'].mean() * 1.1:
                    recommendations.append("📈 **Increase Production Capacity**: Forecasted demand is significantly higher than historical average.")
                elif monthly_avg < df['Sales'].mean() * 0.9:
                    recommendations.append("📉 **Consider Production Adjustment**: Forecasted demand is lower than historical average.")
                
                recommendations.append(f"📦 **Recommended Inventory Level**: Maintain safety stock of ${safety_stock:,.0f} units")
                recommendations.append(f"🔄 **Reorder When**: Inventory drops below ${reorder_point:,.0f} units")
                recommendations.append(f"📊 **Total Forecasted Demand (Next {forecast_horizon} months)**: ${total_forecast:,.0f}")
                
                for rec in recommendations:
                    st.markdown(rec)
                
                st.session_state.supply_chain_metrics = supply_chain_metrics
            else:
                st.warning("No valid model results available for supply chain analysis")
        else:
            st.info("Enable Supply Chain Analysis in sidebar to view this section")
    
    with tab5:
        st.subheader("AI-Powered Analysis & Report Generation")
        
        if st.session_state.forecast_results:
            if st.button("🤖 Generate AI Analysis", type="primary"):
                with st.spinner("Generating comprehensive AI analysis..."):
                    results = st.session_state.forecast_results
                    valid_models = {k: v for k, v in results.items() if not np.isnan(v['mape'])}
                    best_model_name = min(valid_models.items(), key=lambda x: x[1]['mape'])[0] if valid_models else None
                    
                    forecast_results_for_llm = {
                        'best_model': best_model_name,
                        'best_forecasts': [f"${f:,.2f}" for f in results[best_model_name]['future_forecast']] if best_model_name else [],
                        'confidence': 'High' if results[best_model_name]['mape'] < 10 else 'Medium' if results[best_model_name]['mape'] < 20 else 'Low' if best_model_name else 'N/A'
                    }
                    
                    supply_chain_metrics = st.session_state.get('supply_chain_metrics', None)
                    
                    llm_text, error = get_gemini_interpretation(
                        st.session_state.model_performance,
                        forecast_results_for_llm,
                        data_summary,
                        supply_chain_metrics
                    )
                    
                    if error:
                        st.error(error)
                    else:
                        st.session_state.llm_report = llm_text
                        st.markdown("### AI-Generated Analysis Report")
                        st.markdown(llm_text)
            
            if st.session_state.llm_report:
                st.markdown("### AI-Generated Analysis Report")
                st.markdown(st.session_state.llm_report)
                
                # Download report section
                st.subheader("📥 Download Report")
                
                # Audience selection
                audience = st.selectbox(
                    "Select Report Audience",
                    ["Management", "Technical", "Supply Chain"],
                    help="Choose the target audience for the report. The LLM will tailor the content accordingly."
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Generate audience-specific report
                    if st.button(f"🤖 Generate {audience} Report", type="primary"):
                        with st.spinner(f"Generating {audience}-specific report..."):
                            results = st.session_state.forecast_results
                            valid_models = {k: v for k, v in results.items() if not np.isnan(v['mape'])}
                            best_model_name = min(valid_models.items(), key=lambda x: x[1]['mape'])[0] if valid_models else None
                            
                            forecast_results_for_llm = {
                                'best_model': best_model_name,
                                'best_forecasts': [f"${f:,.2f}" for f in results[best_model_name]['future_forecast']] if best_model_name else [],
                                'confidence': 'High' if results[best_model_name]['mape'] < 10 else 'Medium' if results[best_model_name]['mape'] < 20 else 'Low' if best_model_name else 'N/A'
                            }
                            
                            supply_chain_metrics = st.session_state.get('supply_chain_metrics', None)
                            
                            llm_text_audience, error = get_gemini_audience_specific_report(
                                st.session_state.model_performance,
                                forecast_results_for_llm,
                                data_summary,
                                supply_chain_metrics,
                                audience
                            )
                            
                            if error:
                                st.error(error)
                            else:
                                st.session_state[f'llm_report_{audience.lower()}'] = llm_text_audience
                                st.session_state['current_audience'] = audience
                                st.success(f"✅ {audience} report generated!")
                                st.markdown(f"### {audience}-Specific Analysis")
                                st.markdown(llm_text_audience)
                
                with col2:
                    # Prepare visualizations for PDF
                    figures_dict = {}
                    if st.session_state.forecast_results:
                        # Main forecast plot
                        try:
                            models_to_plot = list(st.session_state.forecast_results.keys())
                            fig, ax = plt.subplots(figsize=(14, 8))
                            ax.plot(df['Date'], df['Sales'], 'o-', label='Historical Data', 
                                   color='blue', linewidth=2, markersize=6)
                            
                            if len(test_data) > 0:
                                for model_name in models_to_plot:
                                    if model_name in st.session_state.forecast_results and len(st.session_state.forecast_results[model_name]['test_forecast']) > 0:
                                        ax.plot(test_data['Date'], st.session_state.forecast_results[model_name]['test_forecast'],
                                               '--', label=f'{model_name} (Test)', alpha=0.7, linewidth=2)
                            
                            future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(months=1),
                                                        periods=forecast_horizon, freq='M')
                            
                            for model_name in models_to_plot:
                                if model_name in st.session_state.forecast_results:
                                    if 'Ensemble' in model_name:
                                        ax.plot(future_dates, st.session_state.forecast_results[model_name]['future_forecast'],
                                               'o-', label=f'{model_name} (Forecast)', alpha=0.9, linewidth=3, markersize=10)
                                    else:
                                        ax.plot(future_dates, st.session_state.forecast_results[model_name]['future_forecast'],
                                               'o--', label=f'{model_name} (Forecast)', alpha=0.8, linewidth=2, markersize=8)
                            
                            ax.set_title('Sales Forecast Comparison', fontsize=16, fontweight='bold')
                            ax.set_xlabel('Date', fontsize=12)
                            ax.set_ylabel('Sales ($)', fontsize=12)
                            ax.legend(loc='best')
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            figures_dict['Forecast Comparison'] = fig
                        except:
                            pass
                        
                        # Time series decomposition
                        try:
                            decomp_fig = plot_time_series_decomposition(df)
                            if decomp_fig:
                                figures_dict['Time Series Decomposition'] = decomp_fig
                        except:
                            pass
                
                # Download buttons
                st.markdown("---")
                
                # Get the appropriate LLM text
                llm_text_for_download = st.session_state.get(f'llm_report_{audience.lower()}', st.session_state.llm_report)
                
                # Generate PDF when button is clicked
                if f'pdf_data_{audience.lower()}' not in st.session_state:
                    st.session_state[f'pdf_data_{audience.lower()}'] = None
                
                # PDF Download
                if PDF_AVAILABLE:
                    col_pdf1, col_pdf2 = st.columns([1, 1])
                    with col_pdf1:
                        if st.button(f"📄 Generate PDF Report ({audience})", type="primary"):
                            with st.spinner("Generating PDF report..."):
                                pdf_data, pdf_error = generate_pdf_report(
                                    llm_text_for_download,
                                    st.session_state.model_performance,
                                    st.session_state.forecast_results,
                                    data_summary,
                                    st.session_state.get('supply_chain_metrics', {}),
                                    audience,
                                    figures_dict
                                )
                                
                                if pdf_error:
                                    st.error(pdf_error)
                                    st.session_state[f'pdf_data_{audience.lower()}'] = None
                                else:
                                    st.session_state[f'pdf_data_{audience.lower()}'] = pdf_data
                                    st.success("✅ PDF report generated!")
                    
                    with col_pdf2:
                        if st.session_state[f'pdf_data_{audience.lower()}']:
                            st.download_button(
                                label=f"📥 Download PDF Report - {audience}",
                                data=st.session_state[f'pdf_data_{audience.lower()}'],
                                file_name=f"forecast_report_{audience.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.info("Click 'Generate PDF Report' first")
                else:
                    st.warning("PDF generation requires: pip install reportlab Pillow")
                
                # TXT Download (fallback)
                report_text = generate_downloadable_report(
                    llm_text_for_download,
                    st.session_state.model_performance,
                    st.session_state.forecast_results,
                    data_summary,
                    st.session_state.get('supply_chain_metrics', {})
                )
                
                st.download_button(
                    label=f"📄 Download TXT Report ({audience})",
                    data=report_text,
                    file_name=f"forecast_report_{audience.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("Train models first to generate AI analysis")

else:
    st.info("👈 Please configure and load data using the sidebar options")
    st.markdown("""
    ### Getting Started:
    1. **Generate Sample Data** or **Upload CSV** file
    2. Select forecasting models
    3. Configure model parameters
    4. Set feature engineering options
    5. Enable supply chain analysis (optional)
    6. Train models and view results
    """)
