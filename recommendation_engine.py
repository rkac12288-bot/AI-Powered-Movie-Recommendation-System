"""
recommendation_engine.py
-------------------------
Four complementary recommendation strategies, now running over the real
300-movie catalog with richer per-movie features (multi-genre, director,
year, runtime, IMDB rating, votes, metascore):
 
1. Content-based filtering  -> genre-overlap candidates ranked by a
   composite score of quality (rating + metascore), popularity (votes),
   and recency -- not just raw average rating.
2. "Since you liked X..."   -> represents every movie as a feature vector
   (genres + year + runtime + rating + metascore + popularity) and finds
   the nearest neighbors to the user's favorite by cosine similarity.
3. Collaborative filtering  -> scipy.stats.pearsonr on a user-item rating
   matrix, same as before, now over a larger and sparser catalog.
4. Cluster-based filtering  -> scipy.cluster.vq.kmeans2 groups users by
   genre-preference vectors (20 genres now instead of 7), then surfaces
   each cluster's top movies.
5. Popularity fallback       -> a simple "trending" list (votes + metascore)
   for cold-start users with no watch history yet.
"""
 
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import cosine
 
from data_simulator import MOVIES, GENRES
 
 
# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi == lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)
 
 
# ---------------------------------------------------------------------------
# 1. Content-based filtering (genre match, ranked by a composite score)
# ---------------------------------------------------------------------------
def content_based_recommend(user: dict, movies: list = MOVIES, n: int = 5) -> list:
    """Recommend unwatched movies overlapping the user's preferred genres,
    ranked by a blend of quality (IMDB rating + Metascore), popularity
    (votes), and recency -- so a well-reviewed, well-known film beats an
    obscure one even if both match the genre equally well."""
    watched_titles = {w["movie"] for w in user["watch_history"]}
    candidates = [
        m for m in movies
        if set(m["genres"]) & set(user["preferences"]) and m["title"] not in watched_titles
    ]
    if not candidates:
        return []
 
    ratings = _minmax(np.array([m["rating"] for m in candidates]))
    metascores = _minmax(np.array([m["metascore"] for m in candidates]))
    popularity = _minmax(np.log1p(np.array([m["votes"] for m in candidates])))
    recency = _minmax(np.array([m["year"] for m in candidates]))
    overlap = np.array([len(set(m["genres"]) & set(user["preferences"])) for m in candidates])
    overlap_norm = _minmax(overlap.astype(float))
 
    scores = (0.35 * overlap_norm + 0.30 * ratings + 0.20 * metascores
              + 0.10 * popularity + 0.05 * recency)
 
    ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
    return [c for c, _ in ranked[:n]]
 
 
def get_trending(movies: list = MOVIES, n: int = 5) -> list:
    """Popularity fallback for cold-start users with no watch history:
    ranks purely by votes + metascore, no personalization needed."""
    popularity = _minmax(np.log1p(np.array([m["votes"] for m in movies])))
    metascores = _minmax(np.array([m["metascore"] for m in movies]))
    scores = 0.6 * popularity + 0.4 * metascores
    ranked = sorted(zip(movies, scores), key=lambda ms: ms[1], reverse=True)
    return [m for m, _ in ranked[:n]]
 
 
# ---------------------------------------------------------------------------
# 2. "Since you liked X..." via content-vector cosine similarity
# ---------------------------------------------------------------------------
def build_content_vectors(movies: list = MOVIES) -> tuple:
    """Represent each movie as a numeric feature vector: multi-hot genres
    (weighted up, since genre is the strongest taste signal) plus
    normalized year, runtime, rating, metascore, and popularity. Returns
    (matrix, title_to_row_index)."""
    genre_index = {g: i for i, g in enumerate(GENRES)}
    n_genre = len(GENRES)
 
    years = _minmax(np.array([m["year"] for m in movies], dtype=float))
    runtimes = _minmax(np.array([m["runtime_minutes"] for m in movies], dtype=float))
    ratings = _minmax(np.array([m["rating"] for m in movies], dtype=float))
    metascores = _minmax(np.array([m["metascore"] for m in movies], dtype=float))
    popularity = _minmax(np.log1p(np.array([m["votes"] for m in movies], dtype=float)))
 
    GENRE_WEIGHT = 2.0  # genre match matters more than the numeric extras
    vectors = np.zeros((len(movies), n_genre + 5))
    for i, m in enumerate(movies):
        for g in m["genres"]:
            vectors[i, genre_index[g]] = GENRE_WEIGHT
        vectors[i, n_genre + 0] = years[i]
        vectors[i, n_genre + 1] = runtimes[i]
        vectors[i, n_genre + 2] = ratings[i]
        vectors[i, n_genre + 3] = metascores[i]
        vectors[i, n_genre + 4] = popularity[i]
 
    title_to_idx = {m["title"]: i for i, m in enumerate(movies)}
    return vectors, title_to_idx
 
 
def similar_to_favorite(user: dict, movies: list = MOVIES, n: int = 3) -> tuple:
    """Find the user's highest-rated watched movie, then return the
    unwatched movies whose content vectors are nearest by cosine
    similarity -- a real nearest-neighbor recommendation, not just an
    exact genre/director match."""
    if not user["watch_history"]:
        return None, []
    favorite = max(user["watch_history"], key=lambda w: w["rating"])
    watched_titles = {w["movie"] for w in user["watch_history"]}
 
    vectors, title_to_idx = build_content_vectors(movies)
    if favorite["movie"] not in title_to_idx:
        return favorite, []
    fav_vec = vectors[title_to_idx[favorite["movie"]]]
 
    scored = []
    for m in movies:
        if m["title"] in watched_titles:
            continue
        sim = 1 - cosine(fav_vec, vectors[title_to_idx[m["title"]]])  # cosine similarity
        scored.append((m, sim))
    scored.sort(key=lambda ms: ms[1], reverse=True)
    return favorite, [m for m, _ in scored[:n]]
 
 
# ---------------------------------------------------------------------------
# 3. Collaborative filtering (Pearson correlation)
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
            candidate_scores.setdefault(movie, []).append(rating * corr)
 
    ranked = sorted(candidate_scores.items(),
                     key=lambda kv: np.mean(kv[1]), reverse=True)
    return [{"title": movie, "predicted_score": round(float(np.mean(v)), 2)}
            for movie, v in ranked[:n]]
 
 
# ---------------------------------------------------------------------------
# 4. K-means clustering of users by genre preference
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
 
    # whiten() normalizes each genre column to unit variance; with 20 genres
    # and few users, some genre columns can be all-zero (nobody in this
    # sample rated that genre) -- whiten leaves those columns untouched
    # rather than dividing by zero, so drop them before clustering.
    nonzero_std_cols = vectors.std(axis=0) > 0
    vectors = vectors[:, nonzero_std_cols]
    whitened = whiten(vectors)  # normalize each remaining feature to unit variance
 
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
