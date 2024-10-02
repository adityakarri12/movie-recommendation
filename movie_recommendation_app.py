import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load datasets
@st.cache_data
def load_data():
    movies = pd.read_csv(r"C:\Users\ASus\movie recommendations\movies.csv")
    ratings = pd.read_csv(r"C:\Users\ASus\movie recommendations\ratings.csv")
    return movies, ratings

movies, ratings = load_data()

st.title('Movie Recommendation System')

# Display dataset preview
if st.checkbox('Show movies dataset'):
    st.write(movies.head())

if st.checkbox('Show ratings dataset'):
    st.write(ratings.head())

# Prepare the dataset
final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
final_dataset.fillna(0, inplace=True)

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# Filter movies and users based on votes
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Build the model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Movie Recommendation function
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name, case=False)]  
    if len(movie_list):        
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend+1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"

# Streamlit input and output
movie_input = st.text_input('Enter a movie name to get recommendations:', 'Iron Man')

if st.button('Get Recommendations'):
    recommendations = get_movie_recommendation(movie_input)
    st.write(recommendations)
