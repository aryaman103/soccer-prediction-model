"""
GoalCast FC - League Table Simulator

This module simulates league tables by predicting outcomes for a complete fixture list
and calculating points, goal differences, and final standings with confidence intervals.
"""

import pandas as pd
import numpy as np
import argparse
import json
import joblib
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeagueSimulator:
    """
    Simulate league tables using match predictions.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the league simulator.
        
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
        
        # Load feature columns
        features_file = self.model_dir / "feature_columns.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.feature_columns = json.load(f)
            logger.info(f"Feature columns loaded: {len(self.feature_columns)} features")
    
    def create_sample_league_fixtures(self, output_file: str = "sample_league_fixtures.csv"):
        """
        Create a complete sample fixture list for demonstration.
        
        Args:
            output_file: Output file path for sample fixtures
        """
        # Premier League teams
        teams = [
            "Manchester City", "Arsenal", "Liverpool", "Newcastle United",
            "Manchester United", "Brighton", "Aston Villa", "Tottenham",
            "West Ham United", "Chelsea", "Crystal Palace", "Fulham",
            "Wolverhampton", "Leeds United", "Everton", "Brentford",
            "Nottingham Forest", "Leicester City", "Bournemouth", "Southampton"
        ]
        
        fixtures = []
        fixture_id = 1
        
        # Generate round-robin fixtures (each team plays each other twice)
        for round_num in range(2):  # Home and away
            for i, home_team in enumerate(teams):
                for j, away_team in enumerate(teams):
                    if i != j:  # Can't play against themselves
                        # Skip reverse fixture in first round
                        if round_num == 0 and j < i:
                            continue
                        
                        fixture = {
                            'fixture_id': fixture_id,
                            'home_team': home_team,
                            'away_team': away_team,
                            'match_date': f"2024-{(fixture_id % 12) + 1:02d}-{((fixture_id // 12) % 28) + 1:02d}",
                            'gameweek': ((fixture_id - 1) // 10) + 1
                        }
                        
                        # Add realistic team features based on team strength
                        home_strength = self._get_team_strength(home_team)
                        away_strength = self._get_team_strength(away_team)
                        
                        if self.feature_columns:
                            self._add_team_features(fixture, home_strength, away_strength)
                        
                        fixtures.append(fixture)
                        fixture_id += 1
        
        # Create DataFrame and save
        fixtures_df = pd.DataFrame(fixtures)
        fixtures_df.to_csv(output_file, index=False)
        
        logger.info(f"Sample league fixtures saved to {output_file}")
        print(f"\nSample league fixture file created: {output_file}")
        print(f"Contains {len(fixtures)} fixtures for {len(teams)} teams")
        print(f"Total gameweeks: {fixtures_df['gameweek'].max()}")
        
        return fixtures_df
    
    def _get_team_strength(self, team: str) -> float:
        """Get relative team strength for realistic features."""
        # Rough strength ratings for sample teams
        strength_map = {
            "Manchester City": 0.9, "Arsenal": 0.85, "Liverpool": 0.85,
            "Manchester United": 0.75, "Newcastle United": 0.7, "Tottenham": 0.7,
            "Chelsea": 0.65, "Brighton": 0.6, "Aston Villa": 0.6,
            "West Ham United": 0.55, "Crystal Palace": 0.5, "Fulham": 0.5,
            "Brentford": 0.45, "Wolverhampton": 0.45, "Leeds United": 0.4,
            "Everton": 0.4, "Nottingham Forest": 0.35, "Leicester City": 0.35,
            "Bournemouth": 0.3, "Southampton": 0.25
        }
        return strength_map.get(team, 0.5)
    
    def _add_team_features(self, fixture: dict, home_strength: float, away_strength: float):
        """Add realistic team features based on strength."""
        # Base stats adjusted by team strength
        for col in self.feature_columns:
            if col.startswith('home_'):
                if 'goals' in col:
                    fixture[col] = 1.0 + home_strength * 1.5 + np.random.normal(0, 0.2)
                elif 'xg' in col:
                    fixture[col] = 0.8 + home_strength * 1.2 + np.random.normal(0, 0.15)
                elif 'shots' in col:
                    fixture[col] = 8 + home_strength * 10 + np.random.normal(0, 2)
                elif 'possession' in col:
                    fixture[col] = 45 + home_strength * 15 + np.random.normal(0, 3)
                elif 'form' in col:
                    fixture[col] = 0.5 + home_strength * 2.3 + np.random.normal(0, 0.3)
                else:
                    fixture[col] = home_strength + np.random.normal(0, 0.1)
            
            elif col.startswith('away_'):
                if 'goals' in col:
                    fixture[col] = 1.0 + away_strength * 1.5 + np.random.normal(0, 0.2)
                elif 'xg' in col:
                    fixture[col] = 0.8 + away_strength * 1.2 + np.random.normal(0, 0.15)
                elif 'shots' in col:
                    fixture[col] = 8 + away_strength * 10 + np.random.normal(0, 2)
                elif 'possession' in col:
                    fixture[col] = 45 + away_strength * 15 + np.random.normal(0, 3)
                elif 'form' in col:
                    fixture[col] = 0.5 + away_strength * 2.3 + np.random.normal(0, 0.3)
                else:
                    fixture[col] = away_strength + np.random.normal(0, 0.1)
            
            else:
                # Relative features
                fixture[col] = (home_strength - away_strength) + np.random.normal(0, 0.2)
    
    def predict_fixtures(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Predict outcomes for all fixtures."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_artifacts first.")
        
        logger.info(f"Predicting outcomes for {len(fixtures_df)} fixtures...")
        
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
                        # Default values for missing features
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
                    'gameweek': fixture.get('gameweek', 1),
                    'home_team': fixture['home_team'],
                    'away_team': fixture['away_team'],
                    'predicted_outcome': predicted_outcome,
                    'home_win_prob': probabilities[0],
                    'draw_prob': probabilities[1],
                    'away_win_prob': probabilities[2],
                    'confidence': max(probabilities)
                }
                
                if 'match_date' in fixture:
                    prediction['match_date'] = fixture['match_date']
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting fixture {idx}: {e}")
                continue
        
        return pd.DataFrame(predictions)
    
    def simulate_league_table(self, predictions_df: pd.DataFrame, 
                            num_simulations: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate final league table using Monte Carlo simulation.
        
        Args:
            predictions_df: DataFrame with match predictions
            num_simulations: Number of simulations to run
            
        Returns:
            Tuple of (final_table, simulation_stats)
        """
        logger.info(f"Running {num_simulations} league simulations...")
        
        # Get all teams
        teams = set(predictions_df['home_team'].tolist() + predictions_df['away_team'].tolist())
        teams = sorted(list(teams))
        
        # Store simulation results
        all_simulations = []
        
        for sim in range(num_simulations):
            # Initialize team stats
            team_stats = {team: {
                'points': 0, 'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
                'goals_for': 0, 'goals_against': 0, 'goal_difference': 0
            } for team in teams}
            
            # Simulate each match
            for _, match in predictions_df.iterrows():
                home_team = match['home_team']
                away_team = match['away_team']
                
                # Normalize probabilities to ensure they sum to 1
                probs = np.array([match['home_win_prob'], match['draw_prob'], match['away_win_prob']])
                probs = probs / probs.sum()
                
                # Sample outcome based on probabilities
                outcome = np.random.choice(
                    ['home_win', 'draw', 'away_win'],
                    p=probs
                )
                
                # Generate realistic scores based on outcome
                if outcome == 'home_win':
                    home_goals = np.random.poisson(2) + 1  # At least 1 goal to win
                    away_goals = np.random.poisson(1)
                    if home_goals <= away_goals:
                        home_goals = away_goals + 1
                elif outcome == 'away_win':
                    away_goals = np.random.poisson(2) + 1
                    home_goals = np.random.poisson(1)
                    if away_goals <= home_goals:
                        away_goals = home_goals + 1
                else:  # draw
                    goals = np.random.poisson(1.5)
                    home_goals = away_goals = goals
                
                # Update team stats
                team_stats[home_team]['played'] += 1
                team_stats[away_team]['played'] += 1
                team_stats[home_team]['goals_for'] += home_goals
                team_stats[home_team]['goals_against'] += away_goals
                team_stats[away_team]['goals_for'] += away_goals
                team_stats[away_team]['goals_against'] += home_goals
                
                if outcome == 'home_win':
                    team_stats[home_team]['points'] += 3
                    team_stats[home_team]['won'] += 1
                    team_stats[away_team]['lost'] += 1
                elif outcome == 'away_win':
                    team_stats[away_team]['points'] += 3
                    team_stats[away_team]['won'] += 1
                    team_stats[home_team]['lost'] += 1
                else:  # draw
                    team_stats[home_team]['points'] += 1
                    team_stats[away_team]['points'] += 1
                    team_stats[home_team]['drawn'] += 1
                    team_stats[away_team]['drawn'] += 1
            
            # Calculate goal differences and create table
            simulation_table = []
            for team in teams:
                stats = team_stats[team]
                stats['goal_difference'] = stats['goals_for'] - stats['goals_against']
                stats['team'] = team
                simulation_table.append(stats)
            
            # Sort by points, then goal difference, then goals for
            simulation_table.sort(
                key=lambda x: (x['points'], x['goal_difference'], x['goals_for']),
                reverse=True
            )
            
            # Add positions
            for i, team_data in enumerate(simulation_table):
                team_data['position'] = i + 1
            
            all_simulations.append(simulation_table)
        
        # Calculate final average table
        final_stats = {team: defaultdict(list) for team in teams}
        
        for simulation in all_simulations:
            for team_data in simulation:
                team = team_data['team']
                for stat in ['points', 'goal_difference', 'goals_for', 'goals_against', 'position']:
                    final_stats[team][stat].append(team_data[stat])
        
        # Create final table with averages and confidence intervals
        final_table = []
        for team in teams:
            team_final = {
                'team': team,
                'avg_points': np.mean(final_stats[team]['points']),
                'points_std': np.std(final_stats[team]['points']),
                'avg_position': np.mean(final_stats[team]['position']),
                'position_std': np.std(final_stats[team]['position']),
                'avg_goal_diff': np.mean(final_stats[team]['goal_difference']),
                'avg_goals_for': np.mean(final_stats[team]['goals_for']),
                'avg_goals_against': np.mean(final_stats[team]['goals_against']),
                'title_prob': sum(1 for pos in final_stats[team]['position'] if pos == 1) / num_simulations,
                'top4_prob': sum(1 for pos in final_stats[team]['position'] if pos <= 4) / num_simulations,
                'relegation_prob': sum(1 for pos in final_stats[team]['position'] if pos >= len(teams) - 2) / num_simulations
            }
            final_table.append(team_final)
        
        # Sort by average points
        final_table.sort(key=lambda x: x['avg_points'], reverse=True)
        
        # Add final positions
        for i, team_data in enumerate(final_table):
            team_data['final_position'] = i + 1
        
        final_table_df = pd.DataFrame(final_table)
        
        # Create simulation statistics
        simulation_stats = []
        for team in teams:
            stats = {
                'team': team,
                'min_points': min(final_stats[team]['points']),
                'max_points': max(final_stats[team]['points']),
                'min_position': min(final_stats[team]['position']),
                'max_position': max(final_stats[team]['position']),
                'points_95th': np.percentile(final_stats[team]['points'], 95),
                'points_5th': np.percentile(final_stats[team]['points'], 5),
                'position_95th': np.percentile(final_stats[team]['position'], 95),
                'position_5th': np.percentile(final_stats[team]['position'], 5)
            }
            simulation_stats.append(stats)
        
        simulation_stats_df = pd.DataFrame(simulation_stats)
        
        logger.info("League simulation completed successfully")
        return final_table_df, simulation_stats_df
    
    def save_results(self, final_table_df: pd.DataFrame, simulation_stats_df: pd.DataFrame,
                    predictions_df: pd.DataFrame, output_prefix: str = "league_simulation"):
        """Save simulation results to files."""
        
        # Save final table
        table_file = f"{output_prefix}_table.csv"
        final_table_df.to_csv(table_file, index=False)
        
        # Save simulation stats
        stats_file = f"{output_prefix}_stats.csv"
        simulation_stats_df.to_csv(stats_file, index=False)
        
        # Save predictions
        predictions_file = f"{output_prefix}_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        
        logger.info(f"Results saved to {table_file}, {stats_file}, and {predictions_file}")
        
        return table_file, stats_file, predictions_file
    
    def print_league_table(self, final_table_df: pd.DataFrame, simulation_stats_df: pd.DataFrame):
        """Print formatted league table with simulation results."""
        
        print("\n" + "="*120)
        print("GOALCAST FC - SIMULATED LEAGUE TABLE")
        print("="*120)
        
        if self.model_metadata:
            print(f"Model Accuracy: {self.model_metadata.get('metrics', {}).get('accuracy', 'N/A'):.3f}")
            print(f"Simulations Run: 1000")
            print("-"*120)
        
        # Header
        print(f"{'Pos':<3} {'Team':<20} {'Pts':<6} {'±':<4} {'GD':<6} {'GF':<4} {'GA':<4} "
              f"{'Title%':<7} {'Top4%':<6} {'Rel%':<6} {'Pos Range':<12}")
        print("-"*120)
        
        # Merge stats
        merged_df = final_table_df.merge(simulation_stats_df, on='team')
        
        for _, row in merged_df.iterrows():
            pos_range = f"{int(row['min_position'])}-{int(row['max_position'])}"
            print(f"{row['final_position']:<3} {row['team']:<20} "
                  f"{row['avg_points']:<6.1f} {row['points_std']:<4.1f} "
                  f"{row['avg_goal_diff']:<6.1f} {row['avg_goals_for']:<4.1f} {row['avg_goals_against']:<4.1f} "
                  f"{row['title_prob']:<7.1%} {row['top4_prob']:<6.1%} {row['relegation_prob']:<6.1%} "
                  f"{pos_range:<12}")
        
        print("-"*120)
        print("Legend: Pts=Average Points, ±=Standard Deviation, GD=Goal Difference")
        print("        Title%=Championship Probability, Top4%=Top 4 Probability, Rel%=Relegation Probability")


def main():
    """Main CLI execution function."""
    parser = argparse.ArgumentParser(
        description="GoalCast FC - Simulate league tables from fixture predictions"
    )
    
    parser.add_argument(
        'fixtures_file',
        nargs='?',
        help='CSV file containing fixture data for full season'
    )
    
    parser.add_argument(
        '--model-dir',
        default='models',
        help='Directory containing trained model artifacts (default: models)'
    )
    
    parser.add_argument(
        '--simulations',
        type=int,
        default=1000,
        help='Number of simulations to run (default: 1000)'
    )
    
    parser.add_argument(
        '--output-prefix',
        default='league_simulation',
        help='Prefix for output files (default: league_simulation)'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a sample league fixture file for demonstration'
    )
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = LeagueSimulator(args.model_dir)
    
    try:
        # Load model artifacts
        simulator.load_model_artifacts()
        
        # Create sample if requested
        if args.create_sample:
            simulator.create_sample_league_fixtures("sample_league_fixtures.csv")
            return
        
        # Check if fixtures file provided
        if not args.fixtures_file:
            print("Error: Please provide a fixtures file or use --create-sample to generate one.")
            print("\nUsage:")
            print("  python league_simulator.py fixtures.csv --simulations 1000")
            print("  python league_simulator.py --create-sample")
            return
        
        # Load fixtures
        fixtures_file = Path(args.fixtures_file)
        if not fixtures_file.exists():
            print(f"Error: Fixtures file not found: {fixtures_file}")
            return
        
        fixtures_df = pd.read_csv(fixtures_file)
        logger.info(f"Loaded {len(fixtures_df)} fixtures from {fixtures_file}")
        
        # Predict all fixtures
        predictions_df = simulator.predict_fixtures(fixtures_df)
        
        # Run league simulation
        final_table_df, simulation_stats_df = simulator.simulate_league_table(
            predictions_df, args.simulations
        )
        
        # Print results
        simulator.print_league_table(final_table_df, simulation_stats_df)
        
        # Save results
        files = simulator.save_results(
            final_table_df, simulation_stats_df, predictions_df, args.output_prefix
        )
        
        print(f"\nSimulation results saved to:")
        for file in files:
            print(f"  - {file}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 