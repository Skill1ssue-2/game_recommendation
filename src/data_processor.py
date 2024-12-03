import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, csv_path):
        self.games_df = None
        self.load_and_process_data(csv_path)
        
    def load_and_process_data(self, csv_path):
        # Load the dataset
        self.games_df = pd.read_csv(csv_path)
        
        # Remove unnecessary columns and duplicates
        if 'Unnamed: 0' in self.games_df.columns:
            self.games_df.drop(columns=['Unnamed: 0'], inplace=True)
        self.games_df = self.games_df.drop_duplicates(subset=['Title'], keep='first')
        self.games_df = self.games_df.reset_index(drop=True)
        
        # Convert numerical columns to float
        numerical_columns = ['Times Listed', 'Number of Reviews', 'Plays', 
                           'Playing', 'Backlogs', 'Wishlist', 'Rating']
        for col in numerical_columns:
            self.games_df[col] = pd.to_numeric(self.games_df[col], errors='coerce')
            # Fill NaN values with 0
            self.games_df[col] = self.games_df[col].fillna(0)
        
        # Convert list-like columns from strings to actual lists
        self.games_df['Team'] = self.games_df['Team'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        self.games_df['Genres'] = self.games_df['Genres'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        
        # Create binary features for each genre
        genres = set([genre for sublist in self.games_df['Genres'] for genre in sublist])
        for genre in genres:
            self.games_df[f'Genre_{genre}'] = self.games_df['Genres'].apply(lambda x: 1 if genre in x else 0)
        
        # Create and normalize engagement metrics
        self._create_engagement_metrics()
        
    def _create_engagement_metrics(self):
        # Create aggregate engagement score
        self.games_df['Engagement_Score'] = (
            self.games_df['Times Listed'].astype(float) + 
            self.games_df['Number of Reviews'].astype(float) + 
            self.games_df['Plays'].astype(float) + 
            self.games_df['Playing'].astype(float) + 
            self.games_df['Backlogs'].astype(float) + 
            self.games_df['Wishlist'].astype(float)
        ) / 6
        
        # Normalize numerical features
        numerical_features = ['Rating', 'Times Listed', 'Number of Reviews', 'Plays', 
                            'Playing', 'Backlogs', 'Wishlist', 'Engagement_Score']
        
        # Ensure all features are numeric
        for feature in numerical_features:
            self.games_df[feature] = pd.to_numeric(self.games_df[feature], errors='coerce')
            self.games_df[feature] = self.games_df[feature].fillna(0)
        
        scaler = MinMaxScaler()
        self.games_df[numerical_features] = scaler.fit_transform(self.games_df[numerical_features])
