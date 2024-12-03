import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class GameRecommender:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.games_df = data_processor.games_df
        
    def get_content_similarity(self):
        feature_cols = [col for col in self.games_df.columns if col.startswith('Genre_')] + \
                      ['Rating', 'Engagement_Score']
        feature_matrix = self.games_df[feature_cols]
        return cosine_similarity(feature_matrix)
        
    def get_collaborative_similarity(self):
        engagement_features = ['Playing', 'Backlogs', 'Wishlist']
        engagement_matrix = self.games_df[engagement_features].values
        return cosine_similarity(engagement_matrix)
    
    def _get_hybrid_scores(self, idx, content_weight=0.7):
        """Calculate hybrid similarity scores for a given game index"""
        content_similarity = self.get_content_similarity()
        content_scores = content_similarity[idx]
        
        collab_similarity = self.get_collaborative_similarity()
        collab_scores = collab_similarity[idx]
        
        return (content_weight * content_scores) + ((1 - content_weight) * collab_scores)
        
    def get_recommendations(self, game_title, n_recommendations=5, content_weight=0.7):
        try:
            idx = self.games_df[self.games_df['Title'] == game_title].index[0]
            hybrid_scores = self._get_hybrid_scores(idx, content_weight)
            
            # Get recommendations
            sim_scores = list(enumerate(hybrid_scores))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = [s for s in sim_scores if s[0] != idx][:n_recommendations]
            
            game_indices = [i[0] for i in sim_scores]
            similarity_scores = [i[1] for i in sim_scores]
            
            # Create recommendations dataframe
            recommendations = self.games_df.iloc[game_indices][['Title', 'Genres', 'Rating']]
            recommendations['Similarity_Score'] = similarity_scores
            
            return recommendations
            
        except IndexError:
            return f"Game '{game_title}' not found! Please check the spelling."
            
    def get_diverse_recommendations(self, game_title, n_recommendations=5, content_weight=0.7):
        try:
            idx = self.games_df[self.games_df['Title'] == game_title].index[0]
            base_game_genres = set(self.games_df.loc[idx, 'Genres'])
            
            # Get similarity scores
            hybrid_scores = self._get_hybrid_scores(idx, content_weight)
            
            # Track seen titles to avoid duplicates
            seen_titles = {game_title}
            recommendations = []
            recommended_genres = set()
            
            for i in np.argsort(hybrid_scores)[::-1]:
                current_title = self.games_df.loc[i, 'Title']
                if current_title in seen_titles:
                    continue
                    
                current_genres = set(self.games_df.loc[i, 'Genres'])
                
                # Add game if it brings new genres
                if len(current_genres - recommended_genres) > 0:
                    recommendations.append(i)
                    recommended_genres.update(current_genres)
                    seen_titles.add(current_title)
                    
                    if len(recommendations) >= n_recommendations:
                        break
            
            # Create recommendations dataframe
            result = self.games_df.iloc[recommendations][['Title', 'Genres', 'Rating']]
            result['Similarity_Score'] = hybrid_scores[recommendations]
            result['New_Genres'] = result['Genres'].apply(
                lambda x: len(set(x) - base_game_genres)
            )
            return result
            
        except IndexError:
            return f"Game '{game_title}' not found! Please check the spelling."
            
    def get_weighted_recommendations(self, game_title, n_recommendations=5, 
                                   content_weight=0.7, rating_weight=0.3, 
                                   popularity_weight=0.2):
        try:
            idx = self.games_df[self.games_df['Title'] == game_title].index[0]
            
            # Get similarity scores
            hybrid_scores = self._get_hybrid_scores(idx, content_weight)
            
            # Incorporate rating and popularity
            normalized_scores = (
                (1 - rating_weight - popularity_weight) * hybrid_scores +
                rating_weight * self.games_df['Rating'].values +
                popularity_weight * self.games_df['Engagement_Score'].values
            )
            
            # Get recommendations avoiding duplicates
            seen_titles = {game_title}
            recommendations = []
            
            for i in np.argsort(normalized_scores)[::-1]:
                current_title = self.games_df.loc[i, 'Title']
                if current_title in seen_titles:
                    continue
                    
                recommendations.append(i)
                seen_titles.add(current_title)
                
                if len(recommendations) >= n_recommendations:
                    break
            
            # Create recommendations dataframe
            result = self.games_df.iloc[recommendations][['Title', 'Genres', 'Rating']]
            result['Similarity_Score'] = normalized_scores[recommendations]
            result['Original_Score'] = hybrid_scores[recommendations]
            return result
            
        except IndexError:
            return f"Game '{game_title}' not found! Please check the spelling."
