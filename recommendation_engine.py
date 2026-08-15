"""
recommendation_engine.py
-------------------------
Three complementary recommendation strategies:

1. Content-based filtering  -> matches a user's preferred genres against the
   catalog, ranked by average rating (with a "since you liked X" framing).
2. Collaborative filtering  -> builds a user-item rating matrix and uses
   scipy.stats.pearsonr to find users with similar taste, then recommends
   what they rated highly.
3. Cluster-based filtering  -> uses scipy.cluster.vq.kmeans2 to group users
   by genre-preference vectors, then surfaces each cluster's top movies.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.cluster.vq import kmeans2, whiten

from data_simulator import MOVIES, GENRES


# ---------------------------------------------------------------------------
# 1. Content-based filtering
# ---------------------------------------------------------------------------
def content_based_recommend(user: dict, movies: list = MOVIES, n: int = 5) -> list:
    """Recommend unwatched movies that match the user's preferred genres,
    ranked by average rating."""
    watched_titles = {w["movie"] for w in user["watch_history"]}
    candidates = [
        m for m in movies
        if m["genre"] in user["preferences"] and m["title"] not in watched_titles
    ]
    candidates.sort(key=lambda m: m["avg_rating"], reverse=True)
    return candidates[:n]


def similar_to_favorite(user: dict, movies: list = MOVIES, n: int = 3) -> tuple:
    """Find the user's highest-rated watched movie and recommend similar
    (same genre or same director) unwatched titles: 'Since you liked X...'"""
    if not user["watch_history"]:
        return None, []
    favorite = max(user["watch_history"], key=lambda w: w["rating"])
    watched_titles = {w["movie"] for w in user["watch_history"]}

    similar = [
        m for m in movies
        if m["title"] not in watched_titles
        and (m["genre"] == favorite["genre"] or m["director"] == favorite["director"])
    ]
    similar.sort(key=lambda m: m["avg_rating"], reverse=True)
    return favorite, similar[:n]


# ---------------------------------------------------------------------------
# 2. Collaborative filtering (Pearson correlation)
# ---------------------------------------------------------------------------
def build_user_item_matrix(users: list, movies: list = MOVIES) -> pd.DataFrame:
    """Rows = users, columns = movie titles, values = rating (NaN if unrated)."""
    titles = [m["title"] for m in movies]
    matrix = pd.DataFrame(np.nan, index=[u["name"] for u in users], columns=titles)
    for u in users:
        for w in u["watch_history"]:
            matrix.loc[u["name"], w["movie"]] = w["rating"]
    return matrix


def pearson_similar_users(matrix: pd.DataFrame, target_user: str, min_overlap: int = 2) -> list:
    """Rank other users by Pearson correlation of ratings on commonly-watched
    movies. Returns list of (user_name, correlation) sorted descending."""
    target = matrix.loc[target_user]
    scores = []
    for other in matrix.index:
        if other == target_user:
            continue
        both_rated = matrix.loc[[target_user, other]].dropna(axis=1)
        if both_rated.shape[1] < min_overlap:
            continue
        x, y = both_rated.loc[target_user].values, both_rated.loc[other].values
        if np.std(x) == 0 or np.std(y) == 0:
            continue  # pearsonr is undefined for constant vectors
        corr, _ = pearsonr(x, y)
        if not np.isnan(corr):
            scores.append((other, corr))
    scores.sort(key=lambda s: s[1], reverse=True)
    return scores


def collaborative_recommend(matrix: pd.DataFrame, target_user: str, n: int = 5,
                             top_k_users: int = 3) -> list:
    """Recommend movies highly rated by the target user's most similar peers
    (by Pearson correlation) that the target user hasn't watched."""
    similar_users = pearson_similar_users(matrix, target_user)[:top_k_users]
    if not similar_users:
        return []

    already_rated = matrix.loc[target_user].dropna().index
    candidate_scores = {}
    for other, corr in similar_users:
        if corr <= 0:
            continue
        for movie, rating in matrix.loc[other].dropna().items():
            if movie in already_rated:
                continue
            # weight each peer's rating by how similar they are to the target
            candidate_scores.setdefault(movie, []).append(rating * corr)

    ranked = sorted(candidate_scores.items(),
                     key=lambda kv: np.mean(kv[1]), reverse=True)
    return [{"title": movie, "predicted_score": round(float(np.mean(v)), 2)}
            for movie, v in ranked[:n]]


# ---------------------------------------------------------------------------
# 3. K-means clustering of users by genre preference
# ---------------------------------------------------------------------------
def genre_preference_vector(user: dict) -> np.ndarray:
    """Average rating the user gave per genre (0 if never watched)."""
    sums = {g: [] for g in GENRES}
    for w in user["watch_history"]:
        sums[w["genre"]].append(w["rating"])
    return np.array([np.mean(sums[g]) if sums[g] else 0.0 for g in GENRES])


def cluster_users(users: list, k: int = 3, seed: int = 42) -> dict:
    """Cluster users into k groups by genre-preference vectors using
    scipy.cluster.vq.kmeans2. Returns {cluster_id: [user_names]}."""
    vectors = np.array([genre_preference_vector(u) for u in users])
    whitened = whiten(vectors)  # normalize each feature to unit variance

    rng = np.random.default_rng(seed)
    centroids, labels = kmeans2(whitened, k, seed=seed, minit="++")

    clusters = {i: [] for i in range(k)}
    for user, label in zip(users, labels):
        clusters[int(label)].append(user["name"])
    return clusters


def cluster_top_movies(clusters: dict, users: list, movies: list = MOVIES, n: int = 5) -> dict:
    """For each cluster, aggregate the ratings given by its members and
    return the top-n highest-rated movies within that cluster."""
    users_by_name = {u["name"]: u for u in users}
    results = {}
    for cluster_id, names in clusters.items():
        movie_ratings = {}
        for name in names:
            for w in users_by_name[name]["watch_history"]:
                movie_ratings.setdefault(w["movie"], []).append(w["rating"])
        ranked = sorted(movie_ratings.items(), key=lambda kv: np.mean(kv[1]), reverse=True)
        results[cluster_id] = [
            {"title": title, "avg_cluster_rating": round(float(np.mean(r)), 2), "votes": len(r)}
            for title, r in ranked[:n]
        ]
    return results
