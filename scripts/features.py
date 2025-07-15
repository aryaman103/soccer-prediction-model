"""
GoalCast FC - Feature Engineering Module

This module computes team-level statistics and creates features suitable for 
machine learning match outcome prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Computes team statistics and engineered features from match and event data.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the feature engineer.
        
        Args:
            data_dir: Path to processed data directory
        """
        self.data_dir = Path(data_dir)
        
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed matches and events data.
        
        Returns:
            Tuple of (matches_df, events_df)
        """
        matches_file = self.data_dir / "matches.csv"
        events_file = self.data_dir / "events.csv"
        
        if not matches_file.exists() or not events_file.exists():
            raise FileNotFoundError(
                f"Processed data files not found in {self.data_dir}. "
                "Please run preprocess.py first."
            )
        
        matches_df = pd.read_csv(matches_file)
        events_df = pd.read_csv(events_file)
        
        # Convert date column
        matches_df['match_date'] = pd.to_datetime(matches_df['match_date'])
        
        logger.info(f"Loaded {len(matches_df)} matches and {len(events_df)} events")
        return matches_df, events_df
    
    def compute_team_stats_from_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute team-level statistics from events data.
        
        Args:
            events_df: Events DataFrame
            
        Returns:
            DataFrame with team statistics per match
        """
        logger.info("Computing team statistics from events...")
        
        team_stats = []
        
        for match_id in events_df['match_id'].unique():
            match_events = events_df[events_df['match_id'] == match_id]
            
            for team in match_events['team'].unique():
                if pd.isna(team) or team == "Unknown_team":
                    continue
                    
                team_events = match_events[match_events['team'] == team]
                
                # Basic event counts
                stats = {
                    'match_id': match_id,
                    'team': team,
                    'total_events': len(team_events),
                    'passes': len(team_events[team_events['event_type'] == 'Pass']),
                    'shots': len(team_events[team_events['event_type'] == 'Shot']),
                    'tackles': len(team_events[team_events['event_type'] == 'Tackle']),
                    'interceptions': len(team_events[team_events['event_type'] == 'Interception']),
                    'fouls': len(team_events[team_events['event_type'] == 'Foul']),
                }
                
                # Shot-related statistics
                shot_events = team_events[team_events['event_type'] == 'Shot']
                if len(shot_events) > 0:
                    stats['shots_on_target'] = len(shot_events) // 2  # Approximation
                    stats['avg_shot_xg'] = shot_events['shot_xg'].mean() if 'shot_xg' in shot_events.columns else 0.1
                    stats['total_xg'] = shot_events['shot_xg'].sum() if 'shot_xg' in shot_events.columns else len(shot_events) * 0.1
                else:
                    stats['shots_on_target'] = 0
                    stats['avg_shot_xg'] = 0
                    stats['total_xg'] = 0
                
                # Pass-related statistics
                pass_events = team_events[team_events['event_type'] == 'Pass']
                if len(pass_events) > 0:
                    stats['pass_completion_rate'] = 0.85  # Approximation
                    stats['avg_pass_length'] = pass_events['pass_length'].mean() if 'pass_length' in pass_events.columns else 20
                    stats['key_passes'] = len(pass_events) // 10  # Approximation
                else:
                    stats['pass_completion_rate'] = 0
                    stats['avg_pass_length'] = 0
                    stats['key_passes'] = 0
                
                # Defensive actions
                stats['defensive_actions'] = stats['tackles'] + stats['interceptions']
                
                # Possession approximation (based on pass percentage)
                total_match_passes = len(match_events[match_events['event_type'] == 'Pass'])
                stats['possession_pct'] = (stats['passes'] / total_match_passes * 100) if total_match_passes > 0 else 50
                
                team_stats.append(stats)
        
        return pd.DataFrame(team_stats)
    
    def compute_historical_features(self, matches_df: pd.DataFrame, 
                                  team_stats_df: pd.DataFrame, 
                                  window_size: int = 5) -> pd.DataFrame:
        """
        Compute historical/rolling features for each team.
        
        Args:
            matches_df: Matches DataFrame
            team_stats_df: Team statistics DataFrame
            window_size: Number of recent matches to consider
            
        Returns:
            DataFrame with historical features
        """
        logger.info(f"Computing historical features with window size {window_size}...")
        
        # Sort matches by date
        matches_sorted = matches_df.sort_values('match_date').copy()
        
        # Merge team stats with match outcomes
        home_stats = team_stats_df.merge(
            matches_sorted[['match_id', 'home_team', 'away_team', 'home_score', 'away_score', 'match_date']],
            on='match_id'
        )
        home_stats = home_stats[home_stats['team'] == home_stats['home_team']].copy()
        home_stats['is_home'] = True
        home_stats['goals_for'] = home_stats['home_score']
        home_stats['goals_against'] = home_stats['away_score']
        
        away_stats = team_stats_df.merge(
            matches_sorted[['match_id', 'home_team', 'away_team', 'home_score', 'away_score', 'match_date']],
            on='match_id'
        )
        away_stats = away_stats[away_stats['team'] == away_stats['away_team']].copy()
        away_stats['is_home'] = False
        away_stats['goals_for'] = away_stats['away_score']
        away_stats['goals_against'] = away_stats['home_score']
        
        # Combine home and away stats
        all_team_matches = pd.concat([home_stats, away_stats], ignore_index=True)
        all_team_matches = all_team_matches.sort_values(['team', 'match_date'])
        
        # Compute rolling features
        historical_features = []
        
        for team in all_team_matches['team'].unique():
            if pd.isna(team):
                continue
                
            team_matches = all_team_matches[all_team_matches['team'] == team].copy()
            team_matches = team_matches.sort_values('match_date')
            
            for idx, match in team_matches.iterrows():
                # Get recent matches (excluding current match)
                recent_matches = team_matches[team_matches['match_date'] < match['match_date']].tail(window_size)
                
                if len(recent_matches) == 0:
                    # No historical data - use defaults
                    features = self._get_default_features(team, match['match_id'])
                else:
                    features = self._compute_team_historical_features(recent_matches, team, match['match_id'])
                
                historical_features.append(features)
        
        return pd.DataFrame(historical_features)
    
    def _get_default_features(self, team: str, match_id: int) -> Dict:
        """Get default features for teams with no historical data."""
        return {
            'match_id': match_id,
            'team': team,
            'recent_avg_goals_for': 1.5,
            'recent_avg_goals_against': 1.5,
            'recent_avg_xg': 1.2,
            'recent_avg_shots': 12,
            'recent_avg_possession': 50,
            'recent_form_points': 1.0,  # Average points per game
            'recent_home_advantage': 0.1,
            'recent_avg_pass_completion': 80,
            'recent_avg_defensive_actions': 15,
            'games_played': 0
        }
    
    def _compute_team_historical_features(self, recent_matches: pd.DataFrame, 
                                        team: str, match_id: int) -> Dict:
        """Compute historical features for a specific team."""
        features = {
            'match_id': match_id,
            'team': team,
            'games_played': len(recent_matches)
        }
        
        # Goal statistics
        features['recent_avg_goals_for'] = recent_matches['goals_for'].mean()
        features['recent_avg_goals_against'] = recent_matches['goals_against'].mean()
        features['recent_goal_difference'] = features['recent_avg_goals_for'] - features['recent_avg_goals_against']
        
        # xG and shooting statistics  
        features['recent_avg_xg'] = recent_matches['total_xg'].mean()
        features['recent_avg_shots'] = recent_matches['shots'].mean()
        features['recent_shot_conversion'] = (
            recent_matches['goals_for'].sum() / recent_matches['shots'].sum() 
            if recent_matches['shots'].sum() > 0 else 0.1
        )
        
        # Possession and passing
        features['recent_avg_possession'] = recent_matches['possession_pct'].mean()
        features['recent_avg_pass_completion'] = recent_matches['pass_completion_rate'].mean()
        features['recent_avg_key_passes'] = recent_matches['key_passes'].mean()
        
        # Defensive statistics
        features['recent_avg_defensive_actions'] = recent_matches['defensive_actions'].mean()
        
        # Form calculation (3 points for win, 1 for draw, 0 for loss)
        form_points = []
        for _, match in recent_matches.iterrows():
            if match['goals_for'] > match['goals_against']:
                form_points.append(3)  # Win
            elif match['goals_for'] == match['goals_against']:
                form_points.append(1)  # Draw
            else:
                form_points.append(0)  # Loss
        
        features['recent_form_points'] = np.mean(form_points) if form_points else 1.0
        features['recent_wins'] = sum(1 for p in form_points if p == 3)
        features['recent_draws'] = sum(1 for p in form_points if p == 1)
        features['recent_losses'] = sum(1 for p in form_points if p == 0)
        
        # Home advantage
        home_matches = recent_matches[recent_matches['is_home'] == True]
        if len(home_matches) > 0:
            home_points = []
            for _, match in home_matches.iterrows():
                if match['goals_for'] > match['goals_against']:
                    home_points.append(3)
                elif match['goals_for'] == match['goals_against']:
                    home_points.append(1)
                else:
                    home_points.append(0)
            features['recent_home_advantage'] = np.mean(home_points) - features['recent_form_points']
        else:
            features['recent_home_advantage'] = 0.1
        
        return features
    
    def create_match_features(self, matches_df: pd.DataFrame, 
                             historical_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create final match-level features with home and away team statistics.
        
        Args:
            matches_df: Matches DataFrame
            historical_features_df: Historical features DataFrame
            
        Returns:
            DataFrame with match features ready for ML
        """
        logger.info("Creating match-level features...")
        
        match_features = []
        
        for _, match in matches_df.iterrows():
            # Get home team features
            home_features = historical_features_df[
                (historical_features_df['match_id'] == match['match_id']) & 
                (historical_features_df['team'] == match['home_team'])
            ]
            
            # Get away team features
            away_features = historical_features_df[
                (historical_features_df['match_id'] == match['match_id']) & 
                (historical_features_df['team'] == match['away_team'])
            ]
            
            if len(home_features) == 0 or len(away_features) == 0:
                # Use default features if team not found
                home_features = pd.DataFrame([self._get_default_features(match['home_team'], match['match_id'])])
                away_features = pd.DataFrame([self._get_default_features(match['away_team'], match['match_id'])])
            
            home_features = home_features.iloc[0]
            away_features = away_features.iloc[0]
            
            # Create match features
            match_feature = {
                'match_id': match['match_id'],
                'match_date': match['match_date'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'home_score': match['home_score'],
                'away_score': match['away_score']
            }
            
            # Add home team features with prefix
            for col in home_features.index:
                if col not in ['match_id', 'team']:
                    match_feature[f'home_{col}'] = home_features[col]
            
            # Add away team features with prefix
            for col in away_features.index:
                if col not in ['match_id', 'team']:
                    match_feature[f'away_{col}'] = away_features[col]
            
            # Add relative features (home vs away)
            match_feature['goal_difference_advantage'] = (
                home_features['recent_avg_goals_for'] - home_features['recent_avg_goals_against']
            ) - (away_features['recent_avg_goals_for'] - away_features['recent_avg_goals_against'])
            
            match_feature['form_advantage'] = home_features['recent_form_points'] - away_features['recent_form_points']
            match_feature['xg_advantage'] = home_features['recent_avg_xg'] - away_features['recent_avg_xg']
            match_feature['possession_advantage'] = home_features['recent_avg_possession'] - away_features['recent_avg_possession']
            
            # Create target variable
            if match['home_score'] > match['away_score']:
                match_feature['outcome'] = 'home_win'
                match_feature['outcome_numeric'] = 0
            elif match['home_score'] < match['away_score']:
                match_feature['outcome'] = 'away_win'
                match_feature['outcome_numeric'] = 2
            else:
                match_feature['outcome'] = 'draw'
                match_feature['outcome_numeric'] = 1
            
            match_features.append(match_feature)
        
        match_features_df = pd.DataFrame(match_features)
        logger.info(f"Created features for {len(match_features_df)} matches")
        
        return match_features_df
    
    def save_features(self, features_df: pd.DataFrame, 
                     output_file: str = "data/processed/match_features.csv") -> None:
        """
        Save engineered features to CSV file.
        
        Args:
            features_df: Features DataFrame
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(features_df)} match features to {output_path}")
        
        # Save feature summary
        summary = {
            'total_matches': len(features_df),
            'feature_columns': [col for col in features_df.columns 
                              if col not in ['match_id', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'outcome']],
            'outcomes_distribution': features_df['outcome'].value_counts().to_dict(),
            'date_range': {
                'start': str(features_df['match_date'].min()),
                'end': str(features_df['match_date'].max())
            }
        }
        
        summary_file = output_path.parent / "features_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved feature summary to {summary_file}")


def main():
    """Main execution function."""
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load processed data
    matches_df, events_df = engineer.load_processed_data()
    
    # Compute team statistics from events
    team_stats_df = engineer.compute_team_stats_from_events(events_df)
    
    # Compute historical features
    historical_features_df = engineer.compute_historical_features(matches_df, team_stats_df)
    
    # Create final match features
    match_features_df = engineer.create_match_features(matches_df, historical_features_df)
    
    # Display summary
    print("\n=== FEATURE ENGINEERING SUMMARY ===")
    print(f"Total matches: {len(match_features_df)}")
    print(f"Feature columns: {len([col for col in match_features_df.columns if 'home_' in col or 'away_' in col])}")
    print(f"Outcome distribution:")
    for outcome, count in match_features_df['outcome'].value_counts().items():
        print(f"  {outcome}: {count} ({count/len(match_features_df)*100:.1f}%)")
    
    # Save features
    engineer.save_features(match_features_df)
    
    print("\nFeature engineering completed successfully!")


if __name__ == "__main__":
    main() 