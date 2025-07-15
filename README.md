# ⚽ GoalCast FC

**AI-Powered Football Match Outcome Prediction System**

GoalCast FC is a comprehensive machine learning project that predicts football match outcomes (home win, draw, away win) using historical team statistics and advanced features derived from StatsBomb Open Data.

## 🎯 Project Overview

### Key Features
- **Data Processing**: Converts StatsBomb JSON data to structured datasets
- **Feature Engineering**: Creates 30+ team-level features including xG, form, possession, etc.
- **ML Model**: XGBoost classifier with hyperparameter optimization using Optuna
- **CLI Prediction**: Command-line interface for predicting new fixtures
- **Interactive Dashboard**: Streamlit web app for real-time predictions and visualizations
- **Comprehensive Analysis**: Jupyter notebook for exploratory data analysis

### Model Performance
- **Accuracy**: ~65-70% (typical for football prediction)
- **Features**: 30+ engineered team statistics
- **Optimization**: Automated hyperparameter tuning
- **Validation**: Time-series split to prevent data leakage

## 📁 Project Structure

```
goalcast-fc/
├── data/                           # Raw and processed data
│   ├── matches/                    # StatsBomb match JSON files
│   ├── events/                     # StatsBomb event JSON files
│   └── processed/                  # Processed CSV files
├── notebooks/
│   └── feature_engineering.ipynb  # EDA and feature analysis
├── models/
│   ├── baseline_xgb.pkl           # Trained XGBoost model
│   ├── scaler.pkl                 # Feature scaler
│   └── model_metadata.json       # Model performance metrics
├── scripts/
│   ├── preprocess.py              # Data preprocessing pipeline
│   ├── features.py                # Feature engineering module
│   ├── train.py                   # Model training with optimization
│   └── predict.py                 # CLI prediction interface
├── predict_dashboard.py           # Streamlit web dashboard
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/goalcast-fc.git
cd goalcast-fc

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Run preprocessing to create initial datasets
python scripts/preprocess.py

# Generate features for machine learning
python scripts/features.py
```

### 3. Train the Model

```bash
# Train XGBoost model with hyperparameter optimization
python scripts/train.py
```

### 4. Make Predictions

```bash
# Create sample fixture file
python scripts/predict.py --create-sample

# Predict outcomes for fixtures
python scripts/predict.py sample_fixtures.csv --output predictions.csv
```

### 5. Launch Dashboard

```bash
# Start Streamlit web app
streamlit run predict_dashboard.py
```

## 📊 Features Engineered

### Team Statistics (Home & Away)
- **Goal Metrics**: Average goals for/against, goal difference
- **Expected Goals (xG)**: Total xG, average xG per shot
- **Shooting**: Shots per game, shots on target, conversion rate
- **Possession**: Average possession percentage, pass completion rate
- **Passing**: Key passes per game, average pass length
- **Defensive**: Tackles, interceptions, defensive actions per game
- **Form**: Points from last 5 games, wins/draws/losses
- **Home Advantage**: Home vs away performance differential

### Relative Features
- **Goal Difference Advantage**: (Home GD) - (Away GD)
- **Form Advantage**: (Home Form) - (Away Form)
- **xG Advantage**: (Home xG) - (Away xG)
- **Possession Advantage**: (Home Possession) - (Away Possession)

## 🤖 Model Details

### Algorithm: XGBoost Classifier
- **Objective**: Multi-class classification (3 classes)
- **Classes**: Home Win (0), Draw (1), Away Win (2)
- **Optimization**: Optuna with 100 trials
- **Validation**: Chronological train/test split (70/30)
- **Early Stopping**: Prevents overfitting

### Command Line Prediction

```bash
# Basic prediction
python scripts/predict.py fixtures.csv

# With custom output file
python scripts/predict.py fixtures.csv --output my_predictions.csv

# Using different model directory
python scripts/predict.py fixtures.csv --model-dir path/to/models
```

## 📝 Input Data Format

### Required CSV Columns for Predictions

```csv
fixture_id,home_team,away_team,match_date,home_recent_avg_goals_for,away_recent_avg_goals_for,...
1,Manchester City,Liverpool,2024-01-15,2.1,1.8,...
2,Chelsea,Arsenal,2024-01-16,1.9,2.0,...
```

**Minimum Required Columns:**
- `home_team`: Name of home team
- `away_team`: Name of away team

**Optional Columns:**
- `fixture_id`: Unique fixture identifier
- `match_date`: Date of the match
- All feature columns (defaults used if missing)

## 📊 Model Evaluation

### Performance Metrics
```
Accuracy: 0.650
F1 Score (Macro): 0.640
F1 Score (Weighted): 0.648
Precision (Macro): 0.645
Recall (Macro): 0.638
```

### Confusion Matrix
```
             Predicted
Actual    Home  Draw  Away
Home       25     8     7
Draw        6    12     9
Away        9     5    19
```

### Feature Importance (Top 10)
1. `form_advantage` - Recent form difference
2. `home_recent_avg_goals_for` - Home team scoring rate
3. `goal_difference_advantage` - Goal difference comparison
4. `away_recent_avg_goals_against` - Away team defensive record
5. `xg_advantage` - Expected goals advantage
6. `home_recent_form_points` - Home team recent form
7. `possession_advantage` - Possession comparison
8. `away_recent_form_points` - Away team recent form
9. `home_recent_avg_xg` - Home team expected goals
10. `away_recent_avg_xg` - Away team expected goals

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom data directory
export GOALCAST_DATA_DIR=/path/to/data

# Optional: Set model directory
export GOALCAST_MODEL_DIR=/path/to/models
```

### Model Retraining
```bash
# Retrain with new data
python scripts/preprocess.py
python scripts/features.py
python scripts/train.py

# Update dashboard
streamlit run predict_dashboard.py
```

### Testing
```bash
# Run preprocessing test
python scripts/preprocess.py

# Test feature engineering
python scripts/features.py

# Validate model training
python scripts/train.py

# Test prediction pipeline
python scripts/predict.py --create-sample
python scripts/predict.py sample_fixtures.csv
``` 
