"""
GoalCast FC - Prediction CLI Module

This module provides command-line interface for predicting match outcomes
using trained models on new fixture data.
"""

import pandas as pd
import numpy as np
import argparse
import json
import joblib
from pathlib import Path
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchPredictor:
    """
    Load trained models and predict outcomes for new fixtures.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the predictor with model directory.
        
        Args:
            model_dir: Directory containing trained model artifacts
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_metadata = None
        
    def load_model_artifacts(self):
        """Load trained model and associated artifacts."""
        logger.info(f"Loading model artifacts from {self.model_dir}")
        
        # Load model
        model_file = self.model_dir / "baseline_xgb.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.model = joblib.load(model_file)
        logger.info("Model loaded successfully")
        
        # Load scaler
        scaler_file = self.model_dir / "scaler.pkl"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.info("Scaler loaded successfully")
        else:
            logger.warning("Scaler file not found, using default scaling")
        
        # Load feature columns
        features_file = self.model_dir / "feature_columns.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.feature_columns = json.load(f)
            logger.info(f"Feature columns loaded: {len(self.feature_columns)} features")
        else:
            logger.warning("Feature columns file not found")
        
        # Load metadata
        metadata_file = self.model_dir / "model_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.model_metadata = json.load(f)
            logger.info("Model metadata loaded")
    
    def create_sample_fixture_csv(self, output_file: str = "sample_fixtures.csv"):
        """
        Create a sample fixture CSV file for demonstration.
        
        Args:
            output_file: Output file path for sample fixtures
        """
        if not self.feature_columns:
            logger.error("Feature columns not loaded. Load model first.")
            return
        
        # Sample teams
        teams = [
            "Manchester City", "Liverpool", "Chelsea", "Arsenal", "Tottenham",
            "Manchester United", "West Ham", "Leicester City", "Brighton", "Crystal Palace"
        ]
        
        # Create sample fixtures
        np.random.seed(42)
        fixtures = []
        
        for i in range(5):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            fixture = {
                'fixture_id': i + 1,
                'home_team': home_team,
                'away_team': away_team,
                'match_date': f"2024-01-{i+15:02d}"
            }
            
            # Add sample features for home team
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
                    elif 'pass' in col:
                        fixture[col] = np.random.uniform(75, 90)
                    elif 'defensive' in col:
                        fixture[col] = np.random.uniform(12, 20)
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
                    elif 'pass' in col:
                        fixture[col] = np.random.uniform(75, 90)
                    elif 'defensive' in col:
                        fixture[col] = np.random.uniform(12, 20)
                    else:
                        fixture[col] = np.random.uniform(0, 1)
                
                else:
                    # Relative features
                    fixture[col] = np.random.uniform(-1, 1)
            
            fixtures.append(fixture)
        
        # Save to CSV
        fixtures_df = pd.DataFrame(fixtures)
        fixtures_df.to_csv(output_file, index=False)
        
        logger.info(f"Sample fixtures saved to {output_file}")
        print(f"\nSample fixture file created: {output_file}")
        print(f"Contains {len(fixtures)} sample fixtures with required features")
        print("\nSample fixtures preview:")
        print(fixtures_df[['fixture_id', 'home_team', 'away_team', 'match_date']].to_string(index=False))
    
    def validate_fixture_data(self, fixtures_df: pd.DataFrame) -> bool:
        """
        Validate fixture data contains required columns.
        
        Args:
            fixtures_df: DataFrame with fixture data
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = ['home_team', 'away_team']
        missing_cols = [col for col in required_cols if col not in fixtures_df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        if not self.feature_columns:
            logger.error("Model feature columns not loaded")
            return False
        
        # Check for feature columns
        missing_features = [col for col in self.feature_columns if col not in fixtures_df.columns]
        if missing_features:
            logger.warning(f"Missing feature columns: {len(missing_features)} out of {len(self.feature_columns)}")
            logger.warning("Will use default values for missing features")
        
        return True
    
    def predict_fixtures(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict outcomes for fixtures.
        
        Args:
            fixtures_df: DataFrame with fixture data
            
        Returns:
            DataFrame with predictions added
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_artifacts first.")
        
        if not self.validate_fixture_data(fixtures_df):
            raise ValueError("Invalid fixture data")
        
        logger.info(f"Predicting outcomes for {len(fixtures_df)} fixtures...")
        
        predictions = []
        
        for idx, fixture in fixtures_df.iterrows():
            try:
                # Prepare feature vector
                feature_vector = []
                
                for col in self.feature_columns:
                    if col in fixtures_df.columns:
                        value = fixture[col]
                        # Handle NaN values
                        if pd.isna(value):
                            if 'goals' in col:
                                value = 1.5
                            elif 'xg' in col:
                                value = 1.2
                            elif 'shots' in col:
                                value = 12
                            elif 'possession' in col:
                                value = 50
                            elif 'form' in col:
                                value = 1.5
                            else:
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
                
                # Map prediction
                outcome_map = {0: 'home_win', 1: 'draw', 2: 'away_win'}
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
                
                # Add match date if available
                if 'match_date' in fixture:
                    prediction['match_date'] = fixture['match_date']
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting fixture {idx}: {e}")
                continue
        
        predictions_df = pd.DataFrame(predictions)
        logger.info(f"Successfully predicted {len(predictions_df)} fixtures")
        
        return predictions_df
    
    def save_predictions(self, predictions_df: pd.DataFrame, output_file: str):
        """
        Save predictions to CSV file.
        
        Args:
            predictions_df: DataFrame with predictions
            output_file: Output file path
        """
        predictions_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
    
    def print_predictions(self, predictions_df: pd.DataFrame):
        """Print predictions in a formatted way."""
        print("\n" + "="*80)
        print("GOALCAST FC - MATCH OUTCOME PREDICTIONS")
        print("="*80)
        
        if self.model_metadata:
            print(f"Model Accuracy: {self.model_metadata.get('metrics', {}).get('accuracy', 'N/A'):.3f}")
            print(f"Model F1 Score: {self.model_metadata.get('metrics', {}).get('f1_macro', 'N/A'):.3f}")
            print("-"*80)
        
        for _, pred in predictions_df.iterrows():
            print(f"\nFixture ID: {pred['fixture_id']}")
            if 'match_date' in pred:
                print(f"Date: {pred['match_date']}")
            print(f"Match: {pred['home_team']} vs {pred['away_team']}")
            print(f"Predicted Outcome: {pred['predicted_outcome'].upper().replace('_', ' ')}")
            print(f"Confidence: {pred['confidence']:.1%}")
            print(f"Probabilities:")
            print(f"  Home Win: {pred['home_win_prob']:.1%}")
            print(f"  Draw:     {pred['draw_prob']:.1%}")
            print(f"  Away Win: {pred['away_win_prob']:.1%}")
            print("-"*50)


def main():
    """Main CLI execution function."""
    parser = argparse.ArgumentParser(
        description="GoalCast FC - Predict match outcomes using trained models"
    )
    
    parser.add_argument(
        'fixtures_file',
        nargs='?',
        help='CSV file containing fixture data'
    )
    
    parser.add_argument(
        '--model-dir',
        default='models',
        help='Directory containing trained model artifacts (default: models)'
    )
    
    parser.add_argument(
        '--output',
        help='Output file for predictions (optional)'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a sample fixtures file for demonstration'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MatchPredictor(args.model_dir)
    
    try:
        # Load model artifacts
        predictor.load_model_artifacts()
        
        # Create sample if requested
        if args.create_sample:
            predictor.create_sample_fixture_csv("sample_fixtures.csv")
            return
        
        # Check if fixtures file provided
        if not args.fixtures_file:
            print("Error: Please provide a fixtures file or use --create-sample to generate one.")
            print("\nUsage:")
            print("  python predict.py fixtures.csv --output predictions.csv")
            print("  python predict.py --create-sample")
            return
        
        # Load fixtures
        fixtures_file = Path(args.fixtures_file)
        if not fixtures_file.exists():
            print(f"Error: Fixtures file not found: {fixtures_file}")
            return
        
        fixtures_df = pd.read_csv(fixtures_file)
        logger.info(f"Loaded {len(fixtures_df)} fixtures from {fixtures_file}")
        
        # Make predictions
        predictions_df = predictor.predict_fixtures(fixtures_df)
        
        # Print predictions
        predictor.print_predictions(predictions_df)
        
        # Save predictions if output file specified
        if args.output:
            predictor.save_predictions(predictions_df, args.output)
            print(f"\nPredictions saved to: {args.output}")
        else:
            # Save to default file
            output_file = f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictor.save_predictions(predictions_df, output_file)
            print(f"\nPredictions saved to: {output_file}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 