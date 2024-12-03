import streamlit as st
import pandas as pd
from src.data_processor import DataProcessor
from src.recommender import GameRecommender
from src.evaluation import Evaluator

def load_recommender():
    data_processor = DataProcessor('data/games.csv')
    recommender = GameRecommender(data_processor)
    return recommender, Evaluator(recommender)

def format_genres(genres_list):
    return ', '.join(genres_list)

def main():
    st.set_page_config(
        page_title="Game Recommender",
        page_icon="🎮",
        layout="wide"
    )
    
    st.title('🎮 Game Recommendation System')
    
    try:
        # Initialize recommender
        recommender, evaluator = load_recommender()
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 3])
        
        # Sidebar controls
        with st.sidebar:
            st.header('Settings')
            rec_type = st.selectbox(
                'Recommendation Type',
                ['Standard', 'Diverse', 'Weighted'],
                help="Standard: Basic similarity\nDiverse: Different genres\nWeighted: Balanced with ratings"
            )
            
            content_weight = st.slider(
                'Content Weight',
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values focus more on game content, lower values on user engagement"
            )
            
            n_recommendations = st.slider(
                'Number of Recommendations',
                min_value=1,
                max_value=10,
                value=5
            )
        
        # Main area
        with col1:
            st.subheader('Find Similar Games')
            game_title = st.text_input('Enter a game title:', 'Elden Ring')
            search_button = st.button('Get Recommendations', type='primary')
            
            # About section
            with st.expander("About the Recommendation Types"):
                st.write("""
                ### How it works
                
                **Standard Recommendations**
                - Based on game similarity
                - Considers genres and ratings
                - Best for finding very similar games
                
                **Diverse Recommendations**
                - Includes different genres
                - More variety in suggestions
                - Good for exploring new genres
                
                **Weighted Recommendations**
                - Balances similarity with ratings
                - Considers game popularity
                - Best for high-quality recommendations
                """)
        
        # Results area
        with col2:
            if search_button:
                st.subheader('Recommended Games')
                
                # Get recommendations based on selected type
                if rec_type == 'Standard':
                    recommendations = recommender.get_recommendations(
                        game_title, 
                        n_recommendations=n_recommendations,
                        content_weight=content_weight
                    )
                elif rec_type == 'Diverse':
                    recommendations = recommender.get_diverse_recommendations(
                        game_title,
                        n_recommendations=n_recommendations,
                        content_weight=content_weight
                    )
                else:  # Weighted
                    recommendations = recommender.get_weighted_recommendations(
                        game_title,
                        n_recommendations=n_recommendations,
                        content_weight=content_weight
                    )
                
                if isinstance(recommendations, str):
                    st.error(recommendations)
                else:
                    # Format the recommendations
                    display_df = recommendations.copy()
                    display_df['Genres'] = display_df['Genres'].apply(format_genres)
                    
                    # Display as a nice table
                    st.dataframe(
                        display_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Title": st.column_config.TextColumn("Title", width="medium"),
                            "Genres": st.column_config.TextColumn("Genres", width="large"),
                            "Rating": st.column_config.NumberColumn(
                                "Rating",
                                format="%.2f ⭐",
                                help="Game rating out of 1.0"
                            ),
                            "Similarity_Score": st.column_config.NumberColumn(
                                "Similarity",
                                format="%.3f 🎯",
                                help="How similar this game is to your selection"
                            ),
                        }
                    )
                    
                    # Create two columns for charts
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        st.subheader('Rating Distribution')
                        ratings_chart = pd.to_numeric(display_df['Rating'])
                        st.bar_chart(ratings_chart)
                    
                    with chart_col2:
                        st.subheader('Genre Distribution')
                        all_genres = []
                        for genres in display_df['Genres'].str.split(', '):
                            all_genres.extend(genres)
                        genre_counts = pd.Series(all_genres).value_counts()
                        st.bar_chart(genre_counts)
                    
                    # Additional metrics
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Average Rating", f"{display_df['Rating'].mean():.2f} ⭐")
                    with metrics_col2:
                        st.metric("Unique Genres", f"{len(genre_counts)} 🎯")
                    with metrics_col3:
                        st.metric("Similarity Score", f"{display_df['Similarity_Score'].mean():.3f} 🎮")
                    
                    # After the existing metrics section:
                    with st.expander("Recommendation Metrics"):
                        # Calculate precision and recall
                        precision, recall = evaluator.calculate_relevance_metrics(
                            game_title, 
                            recommendations,
                            k=n_recommendations
                        )
                        
                        # Display metrics in columns
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Precision@K", f"{precision:.3f} 📊")
                        with metric_cols[1]:
                            st.metric("Recall@K", f"{recall:.3f} 📈")
                        with metric_cols[2]:
                            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            st.metric("F1 Score", f"{f1_score:.3f} 🎯")
                        with metric_cols[3]:
                            st.metric("Coverage", f"{len(set([g for genres in recommendations['Genres'] for g in genres]))}/{len(recommendations)} 🔍")
                        
                        # Add explanation
                        st.markdown("""
                        ### Metric Explanations:
                        - **Precision@K**: Proportion of relevant recommendations among the top K results
                        - **Recall@K**: Proportion of relevant items that were recommended
                        - **F1 Score**: Harmonic mean of precision and recall
                        - **Coverage**: Number of unique genres covered in recommendations
                        
                        *Relevance is determined by genre overlap and rating similarity with the original game*
                        """)
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Error details:", e)

if __name__ == '__main__':
    main() 