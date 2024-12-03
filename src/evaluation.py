import pandas as pd
import numpy as np

class Evaluator:
    def __init__(self, recommender):
        self.recommender = recommender
        
    def calculate_relevance_metrics(self, original_game, recommended_games, k=5):
        """Calculate precision and recall based on genre and rating similarity"""
        # Get original game data
        orig_data = self.recommender.data_processor.games_df[
            self.recommender.data_processor.games_df['Title'] == original_game
        ].iloc[0]
        
        orig_genres = set(orig_data['Genres'])
        orig_rating = orig_data['Rating']
        
        def is_relevant(game_genres, game_rating):
            """Helper function to determine relevance consistently"""
            genre_overlap = len(set(game_genres) & orig_genres) / max(len(orig_genres), 1)
            rating_diff = abs(game_rating - orig_rating)
            
            # Game is relevant if it has decent genre overlap (>25%) AND similar rating (within 0.3)
            return (genre_overlap >= 0.25) and (rating_diff <= 0.3)
        
        # Count relevant recommendations
        relevant_count = sum(1 for _, game in recommended_games.iterrows() 
                            if is_relevant(game['Genres'], game['Rating']))
        
        precision = relevant_count / k if k > 0 else 0
        
        # Calculate total relevant items in catalog using same criteria
        total_relevant = sum(1 for _, game in self.recommender.data_processor.games_df.iterrows()
                            if is_relevant(game['Genres'], game['Rating']))
        
        recall = relevant_count / total_relevant if total_relevant > 0 else 0
        
        return precision, recall

    def evaluate_recommendations(self, test_games, weights=[0.7, 0.5, 0.3], k=5):
        results = []
        
        for game in test_games:
            for weight in weights:
                # Get different types of recommendations
                standard_recs = self.recommender.get_recommendations(game, n_recommendations=k, content_weight=weight)
                diverse_recs = self.recommender.get_diverse_recommendations(game, n_recommendations=k, content_weight=weight)
                weighted_recs = self.recommender.get_weighted_recommendations(game, n_recommendations=k, content_weight=weight)
                
                if isinstance(standard_recs, str):
                    continue
                
                # Calculate precision and recall for each method
                standard_precision, standard_recall = self.calculate_relevance_metrics(game, standard_recs, k)
                diverse_precision, diverse_recall = self.calculate_relevance_metrics(game, diverse_recs, k)
                weighted_precision, weighted_recall = self.calculate_relevance_metrics(game, weighted_recs, k)
                
                result = {
                    'Game': game,
                    'Content_Weight': weight,
                    'Standard_Avg_Rating': standard_recs['Rating'].mean(),
                    'Diverse_Avg_Rating': diverse_recs['Rating'].mean(),
                    'Weighted_Avg_Rating': weighted_recs['Rating'].mean(),
                    'Standard_Genre_Count': len(set([g for genres in standard_recs['Genres'] for g in genres])),
                    'Diverse_Genre_Count': len(set([g for genres in diverse_recs['Genres'] for g in genres])),
                    'Weighted_Genre_Count': len(set([g for genres in weighted_recs['Genres'] for g in genres])),
                    # Add precision and recall metrics
                    'Standard_Precision': standard_precision,
                    'Standard_Recall': standard_recall,
                    'Diverse_Precision': diverse_precision,
                    'Diverse_Recall': diverse_recall,
                    'Weighted_Precision': weighted_precision,
                    'Weighted_Recall': weighted_recall
                }
                results.append(result)
        
        return pd.DataFrame(results)
