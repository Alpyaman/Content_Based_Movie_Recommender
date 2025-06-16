import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import joblib  # for saving/loading similarity matrix cache

# Load movies dataset
movies_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
               'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
               'Documentary', 'Drama', 'Fantasy', 'Film-noir', 'Horror', 'Musical', 'Mystery',
               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/Users/alpya/Documents/movie-recommender/data/raw/ml-100k/u.item', 
                     sep='|', encoding='latin-1', names=movies_cols, usecols=range(24))

# Extract year from title, fill missing with 0
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)

# Create unique title to avoid duplicates
movies['title_unique'] = movies['title'] + " [" + movies['item_id'].astype(str) + ']'

# Load ratings
rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/Users/alpya/Documents/movie-recommender/data/raw/ml-100k/u.data', 
                      sep='\t', names=rating_cols)

# Compute average rating per movie and merge
avg_ratings = ratings.groupby('item_id')['rating'].mean()
movies = movies.merge(avg_ratings, left_on='item_id', right_index=True, how='left')
movies.rename(columns={'rating': 'avg_rating'}, inplace=True)
movies['avg_rating'] = movies['avg_rating'].fillna(0)

# Prepare TF-IDF on titles
movies['title_clean'] = movies['title'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['title_clean'])

# Normalize year
scaler = MinMaxScaler()
year_scaled = scaler.fit_transform(movies[['year']])

# Genre columns
genre_cols = movies_cols[5:]
genre_sparse = csr_matrix(movies[genre_cols].values)
year_sparse = csr_matrix(year_scaled)

# Combine all features
combined_features = hstack([genre_sparse, tfidf_matrix, year_sparse])

# Cache/load similarity matrix for speed
SIM_CACHE_FILE = 'combined_sim.npy'

try:
    combined_sim = joblib.load(SIM_CACHE_FILE)
    print("Loaded similarity matrix from cache.")
except FileNotFoundError:
    print("Computing similarity matrix...")
    combined_sim = cosine_similarity(combined_features)
    joblib.dump(combined_sim, SIM_CACHE_FILE)
    print("Saved similarity matrix to cache.")

# Map title_unique to index
indices = pd.Series(movies.index.values, index=movies['title_unique']).drop_duplicates()

def recommend_movies_advanced(title_unique, num_recommendations=5, genre_filter=None, year_range=None):
    if title_unique not in indices:
        print(f"[WARN] Title not found: {title_unique}")
        return []

    idx = indices[title_unique]
    if isinstance(idx, (pd.Series, np.ndarray)):
        print(f"[ERROR] Duplicate index for title {title_unique} -> idx: {idx}")
        return []

    # Apply genre filter mask
    if genre_filter is not None and genre_filter in genre_cols:
        genre_mask = movies[genre_filter] == 1
    else:
        genre_mask = pd.Series([True] * len(movies))

    # Apply year range filter mask
    if year_range is not None and len(year_range) == 2:
        year_mask = (movies['year'] >= year_range[0]) & (movies['year'] <= year_range[1])
    else:
        year_mask = pd.Series([True] * len(movies))

    combined_mask = genre_mask & year_mask

    # Filter similarity scores by mask
    sim_scores = [(i, score) for i, score in enumerate(combined_sim[idx]) if combined_mask.iloc[i]]

    # Sort by similarity score descending, exclude the movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != idx][:num_recommendations]

    movie_indices = [i[0] for i in sim_scores]
    return movies['title_unique'].iloc[movie_indices].tolist()

# Export what app.py needs
__all__ = ['movies', 'recommend_movies_advanced', 'movies_cols', 'genre_cols']
