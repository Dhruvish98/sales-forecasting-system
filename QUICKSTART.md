# Quick Start Guide

Get your Sales Forecasting app running in 5 minutes!

## Local Setup (2 minutes)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the app:**
```bash
streamlit run app.py
```

3. **Open your browser:**
The app will automatically open at `http://localhost:8501`

## Using the App (3 minutes)

1. **Load Data:**
   - Check "Use Sample Data" in the sidebar (for testing)
   - OR upload your own CSV file with Date and Sales columns

2. **Select Models:**
   - Choose which forecasting models to compare
   - Default: All models selected

3. **Configure Settings:**
   - Adjust forecast horizon (default: 3 months)
   - Set training data percentage (default: 80%)

4. **View Results:**
   - See performance metrics (MAPE, WAPE)
   - Check visualizations
   - Read AI-powered insights (if API key configured)

## Sample Data Format

Your CSV should look like this:

```csv
Date,Sales
2023-01-31,10000
2023-02-28,12000
2023-03-31,11000
```

## LLM Feature (Optional - Gemini)

To enable AI interpretation:

1. Get Gemini API key from https://makersuite.google.com/app/apikey
2. Set environment variable:
   ```bash
   # Windows
   set GEMINI_API_KEY=your_key_here
   
   # Mac/Linux
   export GEMINI_API_KEY=your_key_here
   ```
3. Restart the app

## Deployment

See `DEPLOYMENT.md` for Streamlit Cloud deployment instructions.

## Need Help?

- Check `README.md` for detailed documentation
- Review `DEPLOYMENT.md` for deployment issues
- Ensure all dependencies are installed correctly

