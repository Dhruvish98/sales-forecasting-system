# Monthly Sales Forecasting for Beverage Product

A comprehensive Streamlit web application for forecasting monthly sales of a new beverage product using multiple forecasting models with LLM-powered interpretation.

## Features

### Core Forecasting
- **Multiple Forecasting Models**: Naïve, ARIMA, Random Forest, and XGBoost
- **Customizable Model Parameters**: Adjust ARIMA (p,d,q), RF/XGBoost hyperparameters
- **Advanced Feature Engineering**: Configurable lag features, rolling windows, seasonal indicators
- **Comprehensive Metrics**: MAPE, WAPE, RMSE, MAE, R², AIC, BIC

### Advanced Analytics
- **Time Series Decomposition**: Trend, seasonal, and residual analysis
- **Residual Analysis**: Q-Q plots, residual distributions, diagnostic charts
- **Feature Importance**: Visualize important features for ML models
- **Model Comparison**: Side-by-side performance comparison

### Supply Chain Integration
- **Inventory Planning**: Safety stock, reorder point calculations
- **Production Planning**: Demand-based production recommendations
- **Economic Order Quantity (EOQ)**: Optimal ordering calculations
- **Service Level Analysis**: Configurable service level parameters

### Data Generation
- **Advanced Sample Data Generator**: Control trend, seasonality, noise levels
- **Customizable Parameters**: Number of months, base sales, feature strengths
- **Realistic Patterns**: Generate data with realistic business patterns

### AI-Powered Insights
- **Gemini Integration**: Google Gemini AI for comprehensive analysis
- **Automated Report Generation**: Downloadable analysis reports
- **Business Recommendations**: Actionable insights for stakeholders
- **Model Interpretation**: Detailed explanation of model performance

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Development

Run the Streamlit app locally:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Data Format

Your CSV file should have:
- A date column (named 'Date', 'date', 'Month', or 'month')
- A sales column (named 'Sales', 'sales', 'Demand', or 'demand')

Example format:
```csv
Date,Sales
2023-01-31,10000
2023-02-28,12000
2023-03-31,11000
...
```

### LLM Integration (Optional - Gemini)

To enable LLM-powered interpretation:

1. Get a Google Gemini API key from https://makersuite.google.com/app/apikey
2. Set it as an environment variable:
   - **Windows**: `set GEMINI_API_KEY=your_key_here`
   - **Linux/Mac**: `export GEMINI_API_KEY=your_key_here`
3. Or add it to Streamlit secrets (for deployment):
   - Create `.streamlit/secrets.toml` file:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file path to `app.py`
7. Add secrets (if using LLM):
   - Go to Settings → Secrets
   - Add: `OPENAI_API_KEY = "your_key_here"`
8. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/           # Streamlit configuration (optional)
    └── secrets.toml      # API keys (not in git)
```

## Models Implemented

1. **Naïve**: Uses the last observed value as forecast
2. **ARIMA**: AutoRegressive Integrated Moving Average model
3. **Random Forest**: Ensemble learning with lag features
4. **XGBoost**: Gradient boosting with feature engineering

## Evaluation Metrics

- **MAPE**: Mean Absolute Percentage Error - measures average percentage error
- **WAPE**: Weighted Absolute Percentage Error - accounts for scale differences

## Features Created

- Lag features (1, 2, 3 months)
- Rolling averages (3 and 6 months)
- Seasonal indicators (Quarterly)
- Trend component

## Learning Outcomes

- Model selection and comparison
- Feature engineering for time series
- Forecasting metrics interpretation
- Business insights communication
- LLM integration for automated analysis

## Notes

- Sample data is provided for demonstration
- Minimum 6 months of data recommended for best results
- ARIMA may require more data points for optimal performance
- LLM interpretation requires OpenAI API key (optional feature)

## License

This project is for educational purposes as part of a graded lab assignment.

