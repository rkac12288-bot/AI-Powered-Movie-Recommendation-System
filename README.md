**AI-Powered Movie Recommendation System**

A Python movie recommendation engine that mimics the personalization layer of a real GenAI media/entertainment product — built for a hackathon demo. Runs on a real movie catalog (IMDB-Movie-Data.csv, the full 1000-movie IMDB dataset), not a hand-typed list.

What it does
Real movie catalog (IMDB-Movie-Data.csv) — the full 1000-movie dataset, with multi-genre tags (20 genres total), director, year, runtime, IMDB rating, vote count, and Metascore. Watch histories are sampled with a popularity bias (like real viewing behavior), which also keeps enough overlap between simulated users for collaborative filtering to find real matches even across the full catalog.
Simulates user data on top of it — profiles with name, age, preferred genres, and a watch history (movie, genre, director, rating, date). Each simulated rating blends genre-preference match and the movie's real IMDB quality, so taste isn't pure noise.
Recommends movies five different ways:
Content-based filtering — matches unwatched movies to a user's preferred genres, ranked by a composite score of genre overlap, IMDB rating, Metascore, popularity (votes), and recency.
"Since you liked X..." — represents every movie as a feature vector (multi-hot genres + year + runtime + rating + metascore + popularity) and finds the nearest neighbors to the user's favorite watched movie by cosine similarity (scipy.spatial.distance.cosine).
Collaborative filtering — builds a user × movie rating matrix and uses scipy.stats.pearsonr to find users with correlated taste, then recommends what those similar users rated highly.
Cluster-based filtering — uses scipy.cluster.vq.kmeans2 to group users by their genre-preference vectors, then surfaces each cluster's top-rated movies.
Trending fallback — a popularity + Metascore ranking for cold-start users with no watch history yet.
Visualizes insights:
Pie chart of genre distribution in a user's watch history (counts every genre a movie carries, not just the first-listed one).
Line plot of that user's ratings over time.
Project structure
movie_recommender/
├── IMDB-Movie-Data.csv         # raw 1000-movie IMDB export (title, genre,
│                                # director, year, runtime, rating, votes, metascore)
├── data_simulator.py          # loads + trims the catalog, synthetic user generator
├── recommendation_engine.py   # content-based, cosine, Pearson, k-means, trending
├── visualize.py                # pie chart + rating trend plots
├── main.py                     # end-to-end demo script
├── requirements.txt
└── README.md
Setup
bash
pip install -r requirements.txt
Run the demo
bash
python main.py

This prints:

A trending list for cold-start users
A sample user's profile and watch history
Content-based recommendations (genre + quality + popularity + recency)
A "since you liked X" nearest-neighbor suggestion
Users most similar to them by rating correlation, and what to recommend based on those users
K-means clusters of all simulated users, with each cluster's top movies

It also saves genre_distribution.png and rating_trend.png to the working directory.

Using it in your own code
python
from data_simulator import generate_users
from recommendation_engine import (
    content_based_recommend, similar_to_favorite, get_trending,
    build_user_item_matrix, collaborative_recommend, cluster_users,
)
from visualize import plot_genre_pie, plot_rating_trend

users = generate_users(n_users=12)
target = users[0]

# Content-based (genre + rating + metascore + popularity + recency)
recs = content_based_recommend(target, n=5)

# Nearest-neighbor by content vector, cosine similarity
favorite, similar = similar_to_favorite(target, n=3)

# Collaborative (Pearson)
matrix = build_user_item_matrix(users)
collab_recs = collaborative_recommend(matrix, target["name"], n=5)

# Clustering
clusters = cluster_users(users, k=3)

# Cold-start fallback
trending = get_trending(n=5)

# Visuals
plot_genre_pie(target, "genre_distribution.png")
plot_rating_trend(target, "rating_trend.png")
Design notes / talking points for judges
Why a real dataset? 1000 real movies (vs. a 30-title hardcoded list) make every recommendation demoable and let correlation/clustering signals actually mean something — with more titles and richer metadata, similar- taste patterns emerge instead of being coincidence.
Why popularity-weighted watch histories? With 1000 movies and each simulated user watching only ~25, uniform random sampling makes it very unlikely two users ever watched the same movie — starving Pearson correlation of anything to compare. Weighting by vote count (people mostly watch what's popular) keeps realistic overlap without hand-fixing the data.
Why cosine similarity on content vectors? Genre-string equality ("same genre = similar") is brittle. Representing each movie as a vector (genres + era + quality + popularity) and measuring the angle between vectors captures how similar, not just same or not — e.g. it correctly ranks other Fantasy/Adventure/Family films above a movie that only shares one tag.
Why Pearson correlation? It measures whether two users' ratings rise and fall together, independent of how generous each rater is overall — a stronger signal than raw rating differences for "taste similarity."
Why k-means on genre vectors? Clustering groups users who behave similarly (not just who's similar to one target user), so you can surface cluster-wide trending titles — useful for cold-start users with little history of their own.
Extending to real GenAI: swap content_based_recommend's ranking for an LLM call that explains why a movie fits ("since you liked Inception's layered plotting, you'll enjoy Arrival's structural twist..."), or embed movie synopses/descriptions and do semantic similarity search on top of (or instead of) the genre-based content vectors.
Where this would plug into a real system: generate_users is the only synthetic piece — swap it for a production ratings database, and the catalog, recommendation, and visualization layers work unchanged
