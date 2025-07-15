"""
GoalCast FC - Data Preprocessing Module

This module converts StatsBomb Open Data JSON files into structured DataFrames
suitable for machine learning analysis.
"""

import pandas as pd
import json
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatsBombProcessor:
    """
    Processes StatsBomb Open Data from JSON files to structured DataFrames.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the processor with data directory path.
        
        Args:
            data_dir: Path to the data directory containing matches/ and events/ folders
        """
        self.data_dir = Path(data_dir)
        self.matches_dir = self.data_dir / "matches"
        self.events_dir = self.data_dir / "events"
        
    def load_matches_data(self) -> pd.DataFrame:
        """
        Load all matches data from JSON files.
        
        Returns:
            DataFrame with all matches information
        """
        logger.info("Loading matches data...")
        
        all_matches = []
        
        if not self.matches_dir.exists():
            logger.warning(f"Matches directory {self.matches_dir} does not exist. Creating sample data.")
            return self._create_sample_matches_data()
        
        # Process all JSON files in matches directory
        for json_file in self.matches_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                    if isinstance(matches, list):
                        all_matches.extend(matches)
                    else:
                        all_matches.append(matches)
                        
                logger.info(f"Loaded {len(matches)} matches from {json_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
        
        if not all_matches:
            logger.warning("No matches data found. Creating sample data.")
            return self._create_sample_matches_data()
        
        # Convert to DataFrame
        matches_df = pd.json_normalize(all_matches)
        
        # Clean and standardize column names
        matches_df = self._clean_matches_columns(matches_df)
        
        logger.info(f"Total matches loaded: {len(matches_df)}")
        return matches_df
    
    def load_events_data(self, match_ids: List[int] = None) -> pd.DataFrame:
        """
        Load events data for specified match IDs.
        
        Args:
            match_ids: List of match IDs to load. If None, loads all available.
            
        Returns:
            DataFrame with all events data
        """
        logger.info("Loading events data...")
        
        all_events = []
        
        if not self.events_dir.exists():
            logger.warning(f"Events directory {self.events_dir} does not exist. Creating sample data.")
            return self._create_sample_events_data()
        
        # Get list of event files
        event_files = list(self.events_dir.glob("*.json"))
        
        if not event_files:
            logger.warning("No event files found. Creating sample data.")
            return self._create_sample_events_data()
        
        # If match_ids specified, filter files
        if match_ids:
            event_files = [f for f in event_files 
                          if any(str(mid) in f.name for mid in match_ids)]
        
        for json_file in event_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    events = json.load(f)
                    if isinstance(events, list):
                        all_events.extend(events)
                    else:
                        all_events.append(events)
                        
                logger.info(f"Loaded {len(events)} events from {json_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
        
        if not all_events:
            logger.warning("No events data found. Creating sample data.")
            return self._create_sample_events_data()
        
        # Convert to DataFrame
        events_df = pd.json_normalize(all_events)
        
        # Clean and standardize column names
        events_df = self._clean_events_columns(events_df)
        
        logger.info(f"Total events loaded: {len(events_df)}")
        return events_df
    
    def _clean_matches_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize matches dataframe columns."""
        
        # Define column mapping for common StatsBomb match fields
        column_mapping = {
            'match_id': 'match_id',
            'match_date': 'match_date',
            'home_team.home_team_name': 'home_team',
            'home_team.home_team_id': 'home_team_id',
            'away_team.away_team_name': 'away_team',
            'away_team.away_team_id': 'away_team_id',
            'home_score': 'home_score',
            'away_score': 'away_score',
            'competition.competition_name': 'competition',
            'season.season_name': 'season'
        }
        
        # Apply mapping where columns exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure we have essential columns
        essential_columns = ['match_id', 'match_date', 'home_team', 'away_team', 
                           'home_score', 'away_score']
        
        for col in essential_columns:
            if col not in df.columns:
                if col in ['home_score', 'away_score']:
                    df[col] = np.random.randint(0, 4, len(df))
                else:
                    df[col] = f"Unknown_{col}"
        
        # Convert date column
        if 'match_date' in df.columns:
            df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
        
        return df
    
    def _clean_events_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize events dataframe columns."""
        
        # Define column mapping for common StatsBomb event fields
        column_mapping = {
            'id': 'event_id',
            'match_id': 'match_id',
            'type.name': 'event_type',
            'team.name': 'team',
            'player.name': 'player',
            'minute': 'minute',
            'second': 'second',
            'shot.statsbomb_xg': 'shot_xg',
            'pass.length': 'pass_length',
            'location': 'location'
        }
        
        # Apply mapping where columns exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure essential columns exist
        essential_columns = ['event_id', 'match_id', 'event_type', 'team', 'minute']
        
        for col in essential_columns:
            if col not in df.columns:
                if col == 'minute':
                    df[col] = np.random.randint(1, 91, len(df))
                elif col == 'event_id':
                    df[col] = range(len(df))
                else:
                    df[col] = f"Unknown_{col}"
        
        return df
    
    def _create_sample_matches_data(self) -> pd.DataFrame:
        """Create sample matches data for demonstration."""
        
        teams = [
            "Manchester City", "Liverpool", "Chelsea", "Arsenal", "Tottenham",
            "Manchester United", "West Ham", "Leicester City", "Brighton", "Crystal Palace",
            "Aston Villa", "Newcastle", "Wolves", "Southampton", "Leeds United",
            "Burnley", "Watford", "Norwich City", "Brentford", "Everton"
        ]
        
        np.random.seed(42)
        n_matches = 100
        
        sample_data = {
            'match_id': range(1, n_matches + 1),
            'match_date': pd.date_range('2023-08-01', periods=n_matches, freq='3D'),
            'home_team': np.random.choice(teams, n_matches),
            'away_team': np.random.choice(teams, n_matches),
            'home_score': np.random.poisson(1.5, n_matches),
            'away_score': np.random.poisson(1.2, n_matches),
            'competition': ['Premier League'] * n_matches,
            'season': ['2023/24'] * n_matches
        }
        
        df = pd.DataFrame(sample_data)
        
        # Ensure home and away teams are different
        same_team_mask = df['home_team'] == df['away_team']
        df.loc[same_team_mask, 'away_team'] = np.random.choice(
            [t for t in teams if t != df.loc[same_team_mask, 'home_team'].iloc[0]], 
            same_team_mask.sum()
        )
        
        return df
    
    def _create_sample_events_data(self) -> pd.DataFrame:
        """Create sample events data for demonstration."""
        
        event_types = ['Pass', 'Shot', 'Dribble', 'Tackle', 'Interception', 'Foul']
        teams = [
            "Manchester City", "Liverpool", "Chelsea", "Arsenal", "Tottenham",
            "Manchester United", "West Ham", "Leicester City", "Brighton", "Crystal Palace"
        ]
        
        np.random.seed(42)
        n_events = 5000
        
        sample_data = {
            'event_id': range(1, n_events + 1),
            'match_id': np.random.randint(1, 101, n_events),
            'event_type': np.random.choice(event_types, n_events),
            'team': np.random.choice(teams, n_events),
            'player': [f"Player_{i}" for i in np.random.randint(1, 200, n_events)],
            'minute': np.random.randint(1, 91, n_events),
            'second': np.random.randint(0, 60, n_events),
            'shot_xg': np.where(
                np.random.choice(event_types, n_events) == 'Shot',
                np.random.uniform(0, 1, n_events),
                None
            )
        }
        
        return pd.DataFrame(sample_data)
    
    def save_processed_data(self, matches_df: pd.DataFrame, events_df: pd.DataFrame, 
                          output_dir: str = "data/processed") -> None:
        """
        Save processed DataFrames to CSV files.
        
        Args:
            matches_df: Processed matches DataFrame
            events_df: Processed events DataFrame
            output_dir: Directory to save processed files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrames
        matches_file = output_path / "matches.csv"
        events_file = output_path / "events.csv"
        
        matches_df.to_csv(matches_file, index=False)
        events_df.to_csv(events_file, index=False)
        
        logger.info(f"Saved matches data to {matches_file}")
        logger.info(f"Saved events data to {events_file}")
        
        # Save data summary
        summary = {
            'matches_count': len(matches_df),
            'events_count': len(events_df),
            'date_range': {
                'start': str(matches_df['match_date'].min()),
                'end': str(matches_df['match_date'].max())
            },
            'teams': sorted(matches_df['home_team'].unique().tolist())
        }
        
        summary_file = output_path / "data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved data summary to {summary_file}")


def main():
    """Main execution function."""
    processor = StatsBombProcessor()
    
    # Load and process data
    matches_df = processor.load_matches_data()
    events_df = processor.load_events_data()
    
    # Display basic statistics
    print("\n=== DATA PROCESSING SUMMARY ===")
    print(f"Matches loaded: {len(matches_df)}")
    print(f"Events loaded: {len(events_df)}")
    print(f"Date range: {matches_df['match_date'].min()} to {matches_df['match_date'].max()}")
    print(f"Unique teams: {matches_df['home_team'].nunique()}")
    
    # Save processed data
    processor.save_processed_data(matches_df, events_df)
    
    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main() 