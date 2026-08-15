"""
data_simulator.py
------------------
Loads a real movie catalog (raw IMDB export: IMDB-Movie-Data.csv) and
simulates user profiles + watch histories on top of it. Real movies/
genres/directors make the recommendation engine's output far more
convincing than a hand-typed list, while the users themselves stay
synthetic since we don't have real per-user ratings to work with.
 
Expected raw CSV columns: Rank, Title, Genre, Description, Director,
Actors, Year, Runtime (Minutes), Rating, Votes, Revenue (Millions),
Metascore. Uses the full catalog (see CATALOG_SIZE).
"""
 
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
 
random.seed(42)
 
CATALOG_SIZE = 1000  # how many movies to load (1000 = the full raw dataset)
 
_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IMDB-Movie-Data.csv")
 
 
def _load_movies(path: str = _DATA_PATH, catalog_size: int = CATALOG_SIZE) -> list:
    """Load the raw IMDB CSV, trim to the most-voted `catalog_size` movies,
    and return a list of movie dicts. `genres` becomes a list;
    `primary_genre` (first listed genre) is kept as a single-label field so
    watch-history logging and the pie/line charts don't need to change."""
    df = pd.read_csv(path)
    df = df.sort_values("Votes", ascending=False).head(catalog_size)
 
    # The raw Kaggle export has real gaps: ~3% of rows are missing Metascore
    # and Revenue (some films were never scored by critics, or lack reported
    # box-office numbers). Left as NaN, these would silently corrupt every
    # normalized score downstream (NaN poisons min/max and sort order), so
    # fill them here: Metascore defaults to the dataset's median critic
    # score, Revenue to 0 (it isn't used in any ranking, just kept for
    # reference).
    df["Metascore"] = df["Metascore"].fillna(df["Metascore"].median())
    df["Revenue (Millions)"] = df["Revenue (Millions)"].fillna(0.0)
 
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle order
 
    movies = []
    for _, row in df.iterrows():
        genres = [g.strip() for g in row["Genre"].split(",")]
        movies.append({
            "title": row["Title"],
            "genres": genres,
            "primary_genre": genres[0],
            "director": row["Director"],
            "year": int(row["Year"]),
            "runtime_minutes": int(row["Runtime (Minutes)"]),
            "rating": float(row["Rating"]),                    # IMDB average rating (0-10)
            "votes": int(row["Votes"]),                         # popularity signal
            "metascore": float(row["Metascore"]),                # critic score (0-100)
            "revenue_millions": float(row["Revenue (Millions)"]),
        })
    return movies
 
 
MOVIES = _load_movies()
GENRES = sorted({g for m in MOVIES for g in m["genres"]})
 
FIRST_NAMES = ["John", "Maria", "Alex", "Priya", "Wei", "Fatima", "Lucas",
               "Sofia", "Omar", "Emma", "Noah", "Ana"]
 
 
def _random_date_within(days_back: int) -> str:
    d = datetime.now() - timedelta(days=random.randint(0, days_back))
    return d.strftime("%Y-%m-%d")
 
 
def _weighted_rating(movie_genres: list, preferences: list, base_quality: float) -> int:
    """Simulate a 1-5 user rating. Blends: (1) how much the movie's genres
    overlap the user's preferences, and (2) the movie's real-world quality
    (IMDB rating), so simulated taste isn't pure noise -- a user who likes
    thrillers rates a good thriller high and a mediocre one only okay."""
    overlap = len(set(movie_genres) & set(preferences))
    quality_component = (base_quality - 5) / 2.5   # ~ -2..+2 for a 0-10 scale, centered on 5
    preference_component = 1.5 if overlap >= 2 else (0.7 if overlap == 1 else -0.5)
    noise = random.uniform(-0.6, 0.6)
    score = 3 + quality_component * 0.5 + preference_component + noise
    return max(1, min(5, round(score)))
 
 
def _weighted_sample(movies: list, k: int) -> list:
    """Sample k movies without replacement, biased toward popular titles
    (by raw vote count -- log-scaling flattens the signal too much to be
    useful here). Mirrors real viewing behavior (people mostly watch
    what's popular) and, as a side effect, keeps meaningful overlap
    between different simulated users' watch histories even across a
    1000-movie catalog -- without this, uniform random sampling makes it
    very unlikely two users ever watched the same movie, starving the
    Pearson-correlation step of shared ratings to compare."""
    k = min(k, len(movies))
    if k == 0:
        return []
    weights = np.array([m["votes"] for m in movies], dtype=float)
    probs = weights / weights.sum()
    idx = np.random.choice(len(movies), size=k, replace=False, p=probs)
    return [movies[i] for i in idx]
 
 
def generate_users(n_users: int = 12, history_len: int = 25) -> list:
    """Create n_users synthetic user profiles with preferences and watch
    history, drawn from the real movie catalog."""
    np.random.seed(42)
    users = []
    for i in range(n_users):
        name = FIRST_NAMES[i % len(FIRST_NAMES)] + (str(i) if i >= len(FIRST_NAMES) else "")
        age = random.randint(18, 55)
        preferences = random.sample(GENRES, k=random.randint(2, 3))
 
        pool_preferred = [m for m in MOVIES if set(m["genres"]) & set(preferences)]
        pool_other = [m for m in MOVIES if not (set(m["genres"]) & set(preferences))]
        n_pref = min(len(pool_preferred), max(1, int(history_len * 0.7)))
        n_other = min(len(pool_other), history_len - n_pref)
        chosen = _weighted_sample(pool_preferred, n_pref) + _weighted_sample(pool_other, n_other)
        random.shuffle(chosen)
 
        watch_history = []
        for m in chosen:
            watch_history.append({
                "movie": m["title"],
                "genre": m["primary_genre"],
                "director": m["director"],
                "rating": _weighted_rating(m["genres"], preferences, m["rating"]),
                "date": _random_date_within(30),
            })
        watch_history.sort(key=lambda w: w["date"])
 
        users.append({
            "name": name,
            "age": age,
            "preferences": preferences,
            "watch_history": watch_history,
        })
    return users
 
 
if __name__ == "__main__":
    import json
    print(f"Loaded {len(MOVIES)} movies across {len(GENRES)} genres: {GENRES}")
    users = generate_users(3)
    print(json.dumps(users[0], indent=2))
 
 
