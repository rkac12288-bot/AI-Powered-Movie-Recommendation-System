# AI-Powered Movie Recommendation System

A Python movie recommendation engine that mimics the personalization layer
of a real GenAI media/entertainment product — built for a hackathon demo.

## What it does

1. **Simulates user data** — profiles with name, age, preferred genres, and
   a watch history (movie, genre, director, rating, date).
2. **Recommends movies** three different ways:
   - **Content-based filtering** — matches unwatched movies to a user's
     preferred genres, ranked by average rating.
   - **"Since you liked X..."** — finds the user's favorite watched movie
     and suggests unwatched titles sharing its genre or director.
   - **Collaborative filtering** — builds a user × movie rating matrix and
     uses `scipy.stats.pearsonr` to find users with correlated taste, then
     recommends what those similar users rated highly.
   - **Cluster-based filtering** — uses `scipy.cluster.vq.kmeans2` to group
     users by their genre-preference vectors, then surfaces each cluster's
     top-rated movies.
3. **Visualizes insights**:
   - Pie chart of genre distribution in a user's watch history.
   - Line plot of that user's ratings over time.

## Project structure

```
movie_recommender/
├── data_simulator.py         # movie catalog + synthetic user generator
├── recommendation_engine.py  # content-based, Pearson, k-means logic
├── visualize.py               # pie chart + rating trend plots
├── main.py                    # end-to-end demo script
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the demo

```bash
python main.py
```

This prints, for a sample user:
- Their profile and watch history
- Content-based recommendations
- A "since you liked X" suggestion
- Users most similar to them by rating correlation, and what to recommend
  based on those users
- K-means clusters of all simulated users, with each cluster's top movies

It also saves `genre_distribution.png` and `rating_trend.png` to the
working directory.

## Using it in your own code

```python
from data_simulator import generate_users
from recommendation_engine import content_based_recommend, collaborative_recommend, build_user_item_matrix, cluster_users
from visualize import plot_genre_pie, plot_rating_trend

users = generate_users(n_users=10)
target = users[0]

# Content-based
recs = content_based_recommend(target, n=5)

# Collaborative (Pearson)
matrix = build_user_item_matrix(users)
collab_recs = collaborative_recommend(matrix, target["name"], n=5)

# Clustering
clusters = cluster_users(users, k=3)

# Visuals
plot_genre_pie(target, "genre_distribution.png")
plot_rating_trend(target, "rating_trend.png")
```

## Design notes / talking points for judges

- **Why Pearson correlation?** It measures whether two users' ratings rise
  and fall together, independent of how generous each rater is overall —
  a stronger signal than raw rating differences for "taste similarity."
- **Why k-means on genre vectors?** Clustering groups users who *behave*
  similarly (not just who's similar to one target user), so you can surface
  cluster-wide trending titles — useful for cold-start users with little
  history of their own.
- **Extending to real GenAI**: swap `content_based_recommend`'s ranking for
  an LLM call that explains *why* a movie fits ("since you liked Inception's
  layered plotting, you'll enjoy Arrival's structural twist..."), or embed
  movie synopses and do semantic similarity search instead of genre-string
  matching.
- **Where this would plug into a real system**: `MOVIES` and `generate_users`
  are the only synthetic pieces — swap them for a real catalog (e.g. TMDB
  API) and a production ratings database, and the recommendation and
  visualization layers work unchanged.
