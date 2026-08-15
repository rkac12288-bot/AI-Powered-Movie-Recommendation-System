"""
data_simulator.py
------------------
Simulates the movie catalog, user profiles, and watch histories used by the
recommendation engine. In a real system this would be replaced by a database
or an API (e.g. TMDB), but for a hackathon demo, synthetic data lets the
whole pipeline run end-to-end with no external dependencies.
"""

import random
from datetime import datetime, timedelta

random.seed(42)

# ---------------------------------------------------------------------------
# Movie catalog
# ---------------------------------------------------------------------------
MOVIES = [
    {"title": "Inception", "genre": "sci-fi", "director": "Christopher Nolan", "avg_rating": 4.8},
    {"title": "Interstellar", "genre": "sci-fi", "director": "Christopher Nolan", "avg_rating": 4.7},
    {"title": "The Dark Knight", "genre": "action", "director": "Christopher Nolan", "avg_rating": 4.9},
    {"title": "Tenet", "genre": "sci-fi", "director": "Christopher Nolan", "avg_rating": 4.2},
    {"title": "Mad Max: Fury Road", "genre": "action", "director": "George Miller", "avg_rating": 4.6},
    {"title": "John Wick", "genre": "action", "director": "Chad Stahelski", "avg_rating": 4.4},
    {"title": "Se7en", "genre": "thriller", "director": "David Fincher", "avg_rating": 4.7},
    {"title": "Gone Girl", "genre": "thriller", "director": "David Fincher", "avg_rating": 4.5},
    {"title": "Zodiac", "genre": "thriller", "director": "David Fincher", "avg_rating": 4.3},
    {"title": "The Shawshank Redemption", "genre": "drama", "director": "Frank Darabont", "avg_rating": 4.9},
    {"title": "Forrest Gump", "genre": "drama", "director": "Robert Zemeckis", "avg_rating": 4.7},
    {"title": "The Godfather", "genre": "drama", "director": "Francis Ford Coppola", "avg_rating": 4.9},
    {"title": "Superbad", "genre": "comedy", "director": "Greg Mottola", "avg_rating": 4.1},
    {"title": "The Grand Budapest Hotel", "genre": "comedy", "director": "Wes Anderson", "avg_rating": 4.4},
    {"title": "Knives Out", "genre": "comedy", "director": "Rian Johnson", "avg_rating": 4.5},
    {"title": "The Conjuring", "genre": "horror", "director": "James Wan", "avg_rating": 4.2},
    {"title": "Hereditary", "genre": "horror", "director": "Ari Aster", "avg_rating": 4.1},
    {"title": "Get Out", "genre": "horror", "director": "Jordan Peele", "avg_rating": 4.6},
    {"title": "La La Land", "genre": "romance", "director": "Damien Chazelle", "avg_rating": 4.5},
    {"title": "Pride and Prejudice", "genre": "romance", "director": "Joe Wright", "avg_rating": 4.3},
    {"title": "Titanic", "genre": "romance", "director": "James Cameron", "avg_rating": 4.6},
    {"title": "Avengers: Endgame", "genre": "action", "director": "Anthony & Joe Russo", "avg_rating": 4.7},
    {"title": "Dune", "genre": "sci-fi", "director": "Denis Villeneuve", "avg_rating": 4.6},
    {"title": "Arrival", "genre": "sci-fi", "director": "Denis Villeneuve", "avg_rating": 4.5},
    {"title": "Prisoners", "genre": "thriller", "director": "Denis Villeneuve", "avg_rating": 4.5},
    {"title": "Whiplash", "genre": "drama", "director": "Damien Chazelle", "avg_rating": 4.8},
    {"title": "Parasite", "genre": "thriller", "director": "Bong Joon-ho", "avg_rating": 4.8},
    {"title": "The Hangover", "genre": "comedy", "director": "Todd Phillips", "avg_rating": 4.0},
    {"title": "A Quiet Place", "genre": "horror", "director": "John Krasinski", "avg_rating": 4.4},
    {"title": "Notting Hill", "genre": "romance", "director": "Roger Michell", "avg_rating": 4.1},
]

GENRES = sorted({m["genre"] for m in MOVIES})

FIRST_NAMES = ["John", "Maria", "Alex", "Priya", "Wei", "Fatima", "Lucas",
               "Sofia", "Omar", "Emma", "Noah", "Ana"]


def _random_date_within(days_back: int) -> str:
    d = datetime.now() - timedelta(days=random.randint(0, days_back))
    return d.strftime("%Y-%m-%d")


def _weighted_rating(genre: str, preferences: list) -> int:
    """Users tend to rate movies in their preferred genres higher (with noise)."""
    base = 4 if genre in preferences else 3
    noise = random.choice([-1, 0, 0, 0, 1])
    return max(1, min(5, base + noise))


def generate_users(n_users: int = 10, history_len: int = 8) -> list:
    """Create n_users synthetic user profiles with preferences and watch history."""
    users = []
    for i in range(n_users):
        name = FIRST_NAMES[i % len(FIRST_NAMES)] + (str(i) if i >= len(FIRST_NAMES) else "")
        age = random.randint(18, 55)
        preferences = random.sample(GENRES, k=random.randint(2, 3))

        # Bias the sample of watched movies toward the user's preferred genres
        pool_preferred = [m for m in MOVIES if m["genre"] in preferences]
        pool_other = [m for m in MOVIES if m["genre"] not in preferences]
        n_pref = min(len(pool_preferred), max(1, int(history_len * 0.7)))
        n_other = min(len(pool_other), history_len - n_pref)
        chosen = random.sample(pool_preferred, n_pref) + random.sample(pool_other, n_other)
        random.shuffle(chosen)

        watch_history = []
        for m in chosen:
            watch_history.append({
                "movie": m["title"],
                "genre": m["genre"],
                "director": m["director"],
                "rating": _weighted_rating(m["genre"], preferences),
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
    users = generate_users(5)
    print(json.dumps(users[0], indent=2))
