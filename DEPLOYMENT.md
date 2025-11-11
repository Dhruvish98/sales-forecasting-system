# Deployment Guide for Streamlit Cloud

This guide will help you deploy your Monthly Sales Forecasting app on Streamlit Cloud.

## Prerequisites

1. GitHub account
2. Streamlit Cloud account (free) - sign up at https://share.streamlit.io/

## Step-by-Step Deployment

### 1. Push Code to GitHub

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Sales forecasting app"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "Sign in" and authorize with GitHub
3. Click "New app"
4. Fill in the form:
   - **Repository**: Select your repository
   - **Branch**: Select `main` (or your default branch)
   - **Main file path**: Enter `app.py`
5. Click "Deploy"

### 3. Configure Secrets (Optional - for LLM)

If you want to use the LLM interpretation feature with Gemini:

1. Get a Gemini API key from https://makersuite.google.com/app/apikey
2. In your Streamlit Cloud app dashboard, go to "Settings" (⚙️ icon)
3. Scroll down to "Secrets"
4. Click "Edit secrets"
5. Add:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```
6. Click "Save"

### 4. Access Your App

Your app will be live at:
```
https://YOUR_APP_NAME.streamlit.app
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are in `requirements.txt`
2. **API Key Not Working**: Check that secrets are saved correctly
3. **App Not Loading**: Check the logs in Streamlit Cloud dashboard

### Updating Your App

Simply push new changes to GitHub:
```bash
git add .
git commit -m "Update app"
git push
```

Streamlit Cloud will automatically redeploy your app.

## Local Testing Before Deployment

Test locally first:
```bash
streamlit run app.py
```

Make sure everything works before deploying!

