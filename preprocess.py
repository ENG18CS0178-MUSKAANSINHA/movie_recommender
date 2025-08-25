import pandas as pd
import numpy as np
import ast
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Download stopwords (only once)
nltk.download('stopwords')

# Load dataset
movies = pd.read_csv("movies.csv")   # use TMDB 5000 dataset subset

# Clean columns: movie_id, title, cast, crew, overview, genres, keywords
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop missing
movies.dropna(inplace=True)

# Convert stringified lists into Python objects
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3])   # top 3 actors

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['crew'] = movies['crew'].apply(fetch_director)

# Combine text features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]

# Text preprocessing
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Cosine similarity
similarity = cosine_similarity(vectors)

# Save
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ Preprocessing complete. Files saved: movies.pkl & similarity.pkl")
