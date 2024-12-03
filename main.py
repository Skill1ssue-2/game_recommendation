from src.data_processor import DataProcessor
from src.recommender import GameRecommender
from src.evaluation import Evaluator
import pandas as pd

def format_recommendations(recommendations, title):
    print(f"\nRecommendations for {title}:")
    print("-" * 70)
    if isinstance(recommendations, str):
        print(recommendations)
        return
        
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(recommendations.to_string(index=False))
    print("\n")

def main():
    # Initialize the system
    data_processor = DataProcessor('data/games.csv')
    recommender = GameRecommender(data_processor)
    evaluator = Evaluator(recommender)
    
    test_games = ['Elden Ring', 'Minecraft', 'Hollow Knight']
    
    for game in test_games:
        print(f"\n{'='*30} {game} {'='*30}")
        format_recommendations(recommender.get_recommendations(game), "Standard")
        format_recommendations(recommender.get_diverse_recommendations(game), "Diverse")
        format_recommendations(recommender.get_weighted_recommendations(game), "Weighted")
    
    print("\nEvaluation Results:")
    print("-" * 70)
    evaluation_results = evaluator.evaluate_recommendations(test_games)
    print(evaluation_results.to_string())

if __name__ == "__main__":
    main()
