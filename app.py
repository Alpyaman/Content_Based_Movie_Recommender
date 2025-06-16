import streamlit as st
from recommender import movies, recommend_movies_advanced, genre_cols

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Content-Based Movie Recommender")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Movie selection
selected_movie = st.sidebar.selectbox(
    "Choose a movie:",
    options=sorted(movies['title_unique'].tolist())
)

# Genre filter
genre = st.sidebar.selectbox("Filter by Genre (optional):", ["All"] + genre_cols)

# Year range filter slider
min_year = int(movies['year'].replace(0, movies['year'].max()).min())
max_year = int(movies['year'].max())
selected_year_range = st.sidebar.slider("Filter by Year Range:", min_year, max_year, (min_year, max_year))

# Recommend button
if st.sidebar.button("🎥 Recommend Similar Movies"):
    genre_filter = None if genre == "All" else genre
    year_range = selected_year_range

    recommendations = recommend_movies_advanced(
        title_unique=selected_movie,
        num_recommendations=5,
        genre_filter=genre_filter,
        year_range=year_range
    )

    st.subheader("📌 Top 5 Similar Movies")
    if recommendations:
        for i, title_unique in enumerate(recommendations, 1):
            movie_row = movies[movies['title_unique'] == title_unique]
            if not movie_row.empty:
                genres = [g for g in genre_cols if movie_row[g].values[0] == 1]
                year = movie_row['year'].values[0]
                avg_rating = movie_row['avg_rating'].values[0]

                st.markdown(f"**{i}. {title_unique}**")
                st.markdown(f"🎭 Genres: `{', '.join(genres)}`")
                st.markdown(f"📅 Year: `{year}`")
                st.markdown(f"⭐ Average Rating: `{avg_rating:.2f}`")
                st.markdown("---")
    else:
        st.warning("No recommendations found for the given filters.")
