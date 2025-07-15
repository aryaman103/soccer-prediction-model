"""
GoalCast FC - Prediction Dashboard

Streamlit web application for interactive match outcome predictions.
Users can upload fixture CSV files and get predictions with visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="GoalCast FC - Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
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
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitPredictor:
    """Streamlit interface for match outcome predictions."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the predictor."""
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_metadata = None
        
    def load_model_artifacts(self):
        """Load trained model and associated artifacts."""
        try:
            # Load model
            model_file = self.model_dir / "baseline_xgb.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)
                st.success("✅ Model loaded successfully!")
            else:
                st.error(f"❌ Model file not found: {model_file}")
                return False
            
            # Load scaler
            scaler_file = self.model_dir / "scaler.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
            
            # Load feature columns
            features_file = self.model_dir / "feature_columns.json"
            if features_file.exists():
                with open(features_file, 'r') as f:
                    self.feature_columns = json.load(f)
            
            # Load metadata
            metadata_file = self.model_dir / "model_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def create_sample_data(self):
        """Create sample fixture data for demonstration."""
        teams = [
            "Manchester City", "Liverpool", "Chelsea", "Arsenal", "Tottenham",
            "Manchester United", "West Ham", "Leicester City", "Brighton", "Crystal Palace"
        ]
        
        np.random.seed(42)
        fixtures = []
        
        for i in range(3):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            fixture = {
                'fixture_id': i + 1,
                'home_team': home_team,
                'away_team': away_team,
                'match_date': f"2024-01-{i+15:02d}"
            }
            
            # Add sample features if feature columns are loaded
            if self.feature_columns:
                for col in self.feature_columns:
                    if col.startswith('home_'):
                        if 'goals' in col:
                            fixture[col] = np.random.uniform(1.0, 2.5)
                        elif 'xg' in col:
                            fixture[col] = np.random.uniform(0.8, 2.0)
                        elif 'shots' in col:
                            fixture[col] = np.random.uniform(10, 18)
                        elif 'possession' in col:
                            fixture[col] = np.random.uniform(40, 65)
                        elif 'form' in col:
                            fixture[col] = np.random.uniform(0.5, 2.8)
                        else:
                            fixture[col] = np.random.uniform(0, 1)
                    elif col.startswith('away_'):
                        if 'goals' in col:
                            fixture[col] = np.random.uniform(1.0, 2.5)
                        elif 'xg' in col:
                            fixture[col] = np.random.uniform(0.8, 2.0)
                        elif 'shots' in col:
                            fixture[col] = np.random.uniform(10, 18)
                        elif 'possession' in col:
                            fixture[col] = np.random.uniform(40, 65)
                        elif 'form' in col:
                            fixture[col] = np.random.uniform(0.5, 2.8)
                        else:
                            fixture[col] = np.random.uniform(0, 1)
                    else:
                        fixture[col] = np.random.uniform(-1, 1)
            
            fixtures.append(fixture)
        
        return pd.DataFrame(fixtures)
    
    def predict_fixtures(self, fixtures_df: pd.DataFrame):
        """Predict outcomes for fixtures."""
        if self.model is None:
            st.error("❌ Model not loaded")
            return None
        
        predictions = []
        
        for idx, fixture in fixtures_df.iterrows():
            try:
                # Prepare feature vector
                feature_vector = []
                
                for col in self.feature_columns:
                    if col in fixtures_df.columns:
                        value = fixture[col]
                        if pd.isna(value):
                            value = 0
                        feature_vector.append(float(value))
                    else:
                        # Use default values for missing features
                        if 'goals' in col:
                            feature_vector.append(1.5)
                        elif 'xg' in col:
                            feature_vector.append(1.2)
                        elif 'shots' in col:
                            feature_vector.append(12)
                        elif 'possession' in col:
                            feature_vector.append(50)
                        elif 'form' in col:
                            feature_vector.append(1.5)
                        else:
                            feature_vector.append(0)
                
                # Scale features
                if self.scaler:
                    X = self.scaler.transform([feature_vector])
                else:
                    X = np.array([feature_vector])
                
                # Make prediction
                probabilities = self.model.predict_proba(X)[0]
                predicted_class = self.model.predict(X)[0]
                
                outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
                predicted_outcome = outcome_map[predicted_class]
                
                prediction = {
                    'fixture_id': fixture.get('fixture_id', idx + 1),
                    'home_team': fixture['home_team'],
                    'away_team': fixture['away_team'],
                    'predicted_outcome': predicted_outcome,
                    'home_win_prob': round(probabilities[0], 3),
                    'draw_prob': round(probabilities[1], 3),
                    'away_win_prob': round(probabilities[2], 3),
                    'confidence': round(max(probabilities), 3)
                }
                
                if 'match_date' in fixture:
                    prediction['match_date'] = fixture['match_date']
                
                predictions.append(prediction)
                
            except Exception as e:
                st.error(f"❌ Error predicting fixture {idx}: {e}")
                continue
        
        return pd.DataFrame(predictions)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">⚽ GoalCast FC</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">AI-Powered Football Match Outcome Predictions</h3>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = StreamlitPredictor()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model loading
    st.sidebar.subheader("🤖 Model Status")
    if st.sidebar.button("🔄 Load Model"):
        with st.spinner("Loading model..."):
            success = predictor.load_model_artifacts()
    else:
        # Auto-load on startup
        success = predictor.load_model_artifacts()
    
    if not success:
        st.error("❌ Please ensure the model is trained. Run `python scripts/train.py` first.")
        st.stop()
    
    # Model metrics display
    if predictor.model_metadata:
        st.sidebar.subheader("📊 Model Performance")
        metrics = predictor.model_metadata.get('metrics', {})
        
        if metrics:
            st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            st.sidebar.metric("F1 Score", f"{metrics.get('f1_macro', 0):.3f}")
            st.sidebar.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            st.sidebar.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Fixture Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file with fixture data",
            type=['csv'],
            help="Upload a CSV file containing fixture data with team names and features"
        )
        
        # Sample data option
        if st.button("📋 Use Sample Data"):
            sample_df = predictor.create_sample_data()
            st.session_state['fixtures_df'] = sample_df
            st.success("✅ Sample data loaded!")
        
        # Process uploaded file
        if uploaded_file is not None:
            try:
                fixtures_df = pd.read_csv(uploaded_file)
                st.session_state['fixtures_df'] = fixtures_df
                st.success(f"✅ Uploaded {len(fixtures_df)} fixtures successfully!")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        
        # Display fixture data
        if 'fixtures_df' in st.session_state:
            fixtures_df = st.session_state['fixtures_df']
            
            st.subheader("📋 Fixture Data")
            
            # Show basic info
            display_cols = ['fixture_id', 'home_team', 'away_team']
            if 'match_date' in fixtures_df.columns:
                display_cols.append('match_date')
            
            st.dataframe(fixtures_df[display_cols], use_container_width=True)
            
            # Predict button
            if st.button("🔮 Predict Outcomes", type="primary"):
                with st.spinner("Making predictions..."):
                    predictions_df = predictor.predict_fixtures(fixtures_df)
                
                if predictions_df is not None and len(predictions_df) > 0:
                    st.session_state['predictions_df'] = predictions_df
                    st.success(f"✅ Predicted {len(predictions_df)} fixtures!")
                    st.rerun()
    
    with col2:
        st.subheader("ℹ️ Instructions")
        st.markdown("""
        **How to use:**
        1. Upload a CSV file with fixture data
        2. Or click "Use Sample Data" for demo
        3. Click "Predict Outcomes" to get predictions
        4. View results and download predictions
        
        **Required CSV columns:**
        - `home_team`: Home team name
        - `away_team`: Away team name
        - Feature columns (optional - defaults used if missing)
        
        **Features include:**
        - Goals for/against averages
        - Expected goals (xG)
        - Shots and possession stats
        - Recent form points
        """)
    
    # Results section
    if 'predictions_df' in st.session_state:
        predictions_df = st.session_state['predictions_df']
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Fixtures", len(predictions_df))
        
        with col2:
            avg_confidence = predictions_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            home_wins = len(predictions_df[predictions_df['predicted_outcome'] == 'Home Win'])
            st.metric("Home Wins", home_wins)
        
        with col4:
            away_wins = len(predictions_df[predictions_df['predicted_outcome'] == 'Away Win'])
            st.metric("Away Wins", away_wins)
        
        # Predictions table
        st.subheader("📊 Detailed Predictions")
        
        # Format predictions for display
        display_predictions = predictions_df.copy()
        display_predictions['home_win_prob'] = display_predictions['home_win_prob'].apply(lambda x: f"{x:.1%}")
        display_predictions['draw_prob'] = display_predictions['draw_prob'].apply(lambda x: f"{x:.1%}")
        display_predictions['away_win_prob'] = display_predictions['away_win_prob'].apply(lambda x: f"{x:.1%}")
        display_predictions['confidence'] = display_predictions['confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_predictions, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Outcome distribution
            outcome_counts = predictions_df['predicted_outcome'].value_counts()
            fig_pie = px.pie(
                values=outcome_counts.values,
                names=outcome_counts.index,
                title="Predicted Outcomes Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig_hist = px.histogram(
                predictions_df,
                x='confidence',
                nbins=20,
                title="Prediction Confidence Distribution"
            )
            fig_hist.update_xaxes(title="Confidence Level")
            fig_hist.update_yaxes(title="Number of Predictions")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Individual match cards
        st.subheader("🎮 Match Predictions")
        
        for _, pred in predictions_df.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>{pred['home_team']} vs {pred['away_team']}</h4>
                    <p><strong>Predicted: {pred['predicted_outcome']}</strong> (Confidence: {pred['confidence']:.1%})</p>
                    <div style="display: flex; justify-content: space-between;">
                        <span>🏠 Home Win: {pred['home_win_prob']:.1%}</span>
                        <span>⚖️ Draw: {pred['draw_prob']:.1%}</span>
                        <span>✈️ Away Win: {pred['away_win_prob']:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Download button
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name=f"goalcast_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Feature importance visualization (if available)
    if predictor.model and hasattr(predictor.model, 'feature_importances_') and predictor.feature_columns:
        st.markdown("---")
        st.subheader("🎯 Feature Importance")
        
        # Get feature importances
        importances = predictor.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': predictor.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)
        
        # Plot
        fig_importance = px.bar(
            feature_importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 15 Most Important Features"
        )
        fig_importance.update_layout(height=600)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>⚽ GoalCast FC - Powered by XGBoost and StatsBomb Data</p>
        <p>Built with Streamlit 🚀</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 