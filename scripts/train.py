"""
GoalCast FC - Model Training Module

This module trains XGBoost models for match outcome prediction with 
hyperparameter optimization and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score
)
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchOutcomePredictor:
    """
    XGBoost-based match outcome prediction model with hyperparameter optimization.
    """
    
    def __init__(self, data_file: str = "data/processed/match_features.csv"):
        """
        Initialize the predictor.
        
        Args:
            data_file: Path to the match features CSV file
        """
        self.data_file = Path(data_file)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.best_params = None
        self.model_metrics = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare the features data.
        
        Returns:
            DataFrame with match features
        """
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Features file not found at {self.data_file}. "
                "Please run features.py first."
            )
        
        df = pd.read_csv(self.data_file)
        df['match_date'] = pd.to_datetime(df['match_date'])
        
        logger.info(f"Loaded {len(df)} matches with {len(df.columns)} features")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for training.
        
        Args:
            df: Features DataFrame
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Define feature columns (exclude metadata and target)
        exclude_cols = [
            'match_id', 'match_date', 'home_team', 'away_team', 
            'home_score', 'away_score', 'outcome', 'outcome_numeric'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        X = df[feature_cols].fillna(0).values
        y = df['outcome_numeric'].values
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} matches")
        return X, y, feature_cols
    
    def split_data_chronologically(self, df: pd.DataFrame, 
                                 train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically to avoid data leakage.
        
        Args:
            df: Features DataFrame
            train_ratio: Ratio of data for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Sort by date
        df_sorted = df.sort_values('match_date').copy()
        
        # Split chronologically
        split_idx = int(len(df_sorted) * train_ratio)
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        logger.info(f"Chronological split: {len(train_df)} train, {len(test_df)} test")
        logger.info(f"Train period: {train_df['match_date'].min()} to {train_df['match_date'].max()}")
        logger.info(f"Test period: {test_df['match_date'].min()} to {test_df['match_date'].max()}")
        
        return train_df, test_df
    
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, 
                 y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Validation F1 score (macro)
        """
        # Suggest hyperparameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'verbosity': 0,
            
            # Hyperparameters to optimize
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)],
                 verbose=False)
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        return f1
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                n_trials: int = 100) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train, y_train: Training data
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters dictionary
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        # Split training data for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_tr, y_tr, X_val, y_val),
            n_trials=n_trials,
            timeout=600  # 10 minutes timeout
        )
        
        self.best_params = study.best_params
        logger.info(f"Best F1 score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray = None, y_test: np.ndarray = None,
                   optimize_params: bool = True) -> xgb.XGBClassifier:
        """
        Train the XGBoost model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data for evaluation
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")
        
        # Optimize hyperparameters if requested
        if optimize_params and self.best_params is None:
            self.optimize_hyperparameters(X_train, y_train)
        
        # Set parameters
        if self.best_params:
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                **self.best_params
            }
        else:
            # Default parameters
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        
        if X_test is not None and y_test is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: List[str] = None) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            X_test, y_test: Test data
            feature_names: List of feature names
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        # Store metrics
        self.model_metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Print results
        print("\n=== MODEL EVALUATION RESULTS ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"Precision (Macro): {precision:.4f}")
        print(f"Recall (Macro): {recall:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Home Win', 'Draw', 'Away Win']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        if feature_names and hasattr(self.model, 'feature_importances_'):
            self.plot_feature_importance(feature_names)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return self.model_metrics
    
    def plot_feature_importance(self, feature_names: List[str], top_n: int = 20):
        """Plot feature importance chart."""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            logger.warning("No feature importances available")
            return
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(top_n)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Feature importance plot saved to models/feature_importance.png")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Home Win', 'Draw', 'Away Win'],
                   yticklabels=['Home Win', 'Draw', 'Away Win'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Confusion matrix plot saved to models/confusion_matrix.png")
    
    def save_model(self, output_dir: str = "models"):
        """
        Save the trained model and associated artifacts.
        
        Args:
            output_dir: Directory to save model artifacts
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = output_path / "baseline_xgb.pkl"
        joblib.dump(self.model, model_file)
        
        # Save scaler
        scaler_file = output_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_file)
        
        # Save feature columns
        features_file = output_path / "feature_columns.json"
        with open(features_file, 'w') as f:
            json.dump(self.feature_columns, f)
        
        # Save model metadata
        metadata = {
            'model_type': 'XGBoost',
            'best_params': self.best_params,
            'metrics': self.model_metrics,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = output_path / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Scaler saved to {scaler_file}")
        logger.info(f"Metadata saved to {metadata_file}")
    
    def predict_match_outcome(self, home_features: Dict, away_features: Dict) -> Dict:
        """
        Predict outcome for a single match.
        
        Args:
            home_features: Dictionary of home team features
            away_features: Dictionary of away team features
            
        Returns:
            Dictionary with prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Create feature vector
        feature_vector = []
        for col in self.feature_columns:
            if col.startswith('home_'):
                feature_name = col[5:]  # Remove 'home_' prefix
                feature_vector.append(home_features.get(feature_name, 0))
            elif col.startswith('away_'):
                feature_name = col[5:]  # Remove 'away_' prefix
                feature_vector.append(away_features.get(feature_name, 0))
            else:
                # Handle relative features
                feature_vector.append(0)  # Default value
        
        # Scale features
        X = self.scaler.transform([feature_vector])
        
        # Predict
        probabilities = self.model.predict_proba(X)[0]
        
        return {
            'home_win_prob': probabilities[0],
            'draw_prob': probabilities[1],
            'away_win_prob': probabilities[2],
            'predicted_outcome': ['home_win', 'draw', 'away_win'][np.argmax(probabilities)]
        }


def main():
    """Main execution function."""
    # Initialize predictor
    predictor = MatchOutcomePredictor()
    
    # Load data
    df = predictor.load_data()
    
    # Split data chronologically
    train_df, test_df = predictor.split_data_chronologically(df, train_ratio=0.7)
    
    # Prepare features
    X_train, y_train, feature_names = predictor.prepare_features(train_df)
    predictor.feature_columns = feature_names
    
    X_test, y_test, _ = predictor.prepare_features(test_df)
    
    # Train model with hyperparameter optimization
    predictor.train_model(X_train, y_train, X_test, y_test, optimize_params=True)
    
    # Evaluate model
    predictor.evaluate_model(X_test, y_test, feature_names)
    
    # Save model
    predictor.save_model()
    
    print("\nModel training completed successfully!")
    print(f"Model saved to models/baseline_xgb.pkl")


if __name__ == "__main__":
    main() 