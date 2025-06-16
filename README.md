# Content-Based Movie Recommender

A Streamlit web app that recommends movies based on content similarity, using features like genres, release year, and title text (TF-IDF).

## Features

- 🎯 Content-based movie recommendations using TF-IDF and genre/year metadata
- 🧠 Smart filtering by genre and year range
- ⭐ Shows average user ratings, genres, and other movie metadata
- 🖥️ Simple and interactive Streamlit interface

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Alpyaman/Content_Based_Movie_Recommender.git
cd Content_Based_Movie_Recommender
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate # Linux/Mac
.\venv\Scripts\activate # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download MovieLens 100k dataset and places files.

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```
Use the sidebar filters to select a movie, genre, and year range, based on your selected movies and then get recommendations!

## Project Structure
- `app.py` - Main Streamlit app
- `recommender.py` - Recommendation engine and data loading
- `data/` - MovieLens data files (not included)
- `requirements.txt` - Python dependencies

  ## Acknowledgments
  - MovieLens dataset by GroupLens Research
  - Built with Python, scikit-learn, Streamlit
