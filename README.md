# Movie-Recommendation-System
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load users and movies datasets
users_df = pd.read_csv('users.csv')  # Ensure this file is accessible
movies_df = pd.read_csv('movies.csv')  # Ensure this file is accessible

# Function for Popularity-Based Recommendations
def popularity_based_recommendations(num_recommendations=5):
    # Calculate average ratings for each movie
    average_ratings = users_df.iloc[:, 4:].mean().sort_values(ascending=False)

    # Get the top N movie indices based on average ratings
    top_movie_indices = average_ratings.index[:num_recommendations]

    # Retrieve the corresponding movie titles from movies_df
    movie_titles = []
    for idx in top_movie_indices:
        movie_id = int(idx.split('_')[1])  # Assuming MovieID corresponds to index after 'Rating_'
        title = movies_df.loc[movies_df['MovieID'] == movie_id, 'Title']
        if not title.empty:
            movie_titles.append(title.values[0])
    
    return movie_titles

# Function for Content-Based Recommendations
def content_based_recommendations(user_id, num_recommendations=5):
    # Here we can implement content-based filtering logic
    # For simplicity, let's assume it returns the first N movies
    return movies_df['Title'].head(num_recommendations).tolist()

# Function for Collaborative Filtering Recommendations
def collaborative_filtering_recommendation(user_id, num_recommendations=5):
    user_ratings = users_df.loc[users_df['UserID'] == user_id].iloc[:, 4:]
    
    # Calculate similarity
    similarity_matrix = cosine_similarity(users_df.iloc[:, 4:])
    similar_users_indices = similarity_matrix[user_id - 1].argsort()[::-1][1:num_recommendations + 1]
    
    # Get recommendations based on similar users
    recommended_movies = set()
    for similar_user_index in similar_users_indices:
        similar_user_ratings = users_df.iloc[similar_user_index, 4:]
        for idx, rating in enumerate(similar_user_ratings):
            if rating >= 3:  # Assuming we recommend movies rated 3 or above
                movie_id = users_df.columns[idx + 4].split('_')[1]  # Extract movie ID
                recommended_movies.add(int(movie_id))
                
    # Get movie titles
    movie_titles = movies_df[movies_df['MovieID'].isin(recommended_movies)]['Title'].tolist()
    
    return movie_titles[:num_recommendations]

# Function to get recommendations for all users
def get_recommendations_for_all_users(num_recommendations=5):
    recommendations = {}
    for user_id in users_df['UserID']:
        recommendations[user_id] = {
            'Popularity-Based Recommendations': popularity_based_recommendations(num_recommendations),
            'Content-Based Recommendations': content_based_recommendations(user_id, num_recommendations),
            'Collaborative Filtering Recommendations': collaborative_filtering_recommendation(user_id, num_recommendations)
        }
    return recommendations

# Example usage for all users
if __name__ == "__main__":
    num_recommendations = 5
    all_user_recommendations = get_recommendations_for_all_users(num_recommendations)
    
    for user_id, recs in all_user_recommendations.items():
        print(f"\nRecommendations for User ID {user_id}:")
        print("Popularity-Based Recommendations:", recs['Popularity-Based Recommendations'])
        print("Content-Based Recommendations:", recs['Content-Based Recommendations'])
        print("Collaborative Filtering Recommendations:", recs['Collaborative Filtering Recommendations'])
