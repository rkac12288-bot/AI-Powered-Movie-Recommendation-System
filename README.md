**AI-Powered Movie Recommendation System**

A Python movie recommendation engine that mimics the personalization layer of a real GenAI media/entertainment product — built for a hackathon demo. Runs on a real movie catalog (IMDB-Movie-Data.csv, the full 1000-movie IMDB dataset), not a hand-typed list, and closes with a presentation-ready storytelling dashboard, not just a grid of charts.

What it does
Real movie catalog (IMDB-Movie-Data.csv) — the full 1000-movie dataset, with multi-genre tags (20 genres total), director, year, runtime, IMDB rating, vote count, and Metascore.
Missing data is repaired with statistics, not guesswork: ~6% of movies are missing a Metascore in the raw export. Rather than filling those gaps with a flat average, the loader fits a linear regression between IMDB Rating and Metascore (Pearson r = 0.63 across the ~936 movies that have both) and predicts each missing value from that movie's own Rating — a well-reviewed movie gets a plausible high Metascore, not the dataset average.
Watch histories are popularity-weighted, mirroring real viewing behavior. This also keeps meaningful overlap between simulated users for collaborative filtering — with 1000 movies and ~25 watched per user, uniform random sampling would make it very unlikely two users ever watched the same movie.
Simulates user data on top of it — profiles with name, age, preferred genres, and a watch history (movie, genre, director, rating, date). Each simulated rating blends genre-preference match and the movie's real IMDB quality, so taste isn't pure noise.
Recommends movies five different ways:
Content-based filtering — matches unwatched movies to a user's preferred genres, ranked by a composite score of genre overlap, IMDB rating, Metascore, popularity (votes), and recency.
"Since you liked X..." — represents every movie as a feature vector (multi-hot genres + year + runtime + rating + metascore + popularity) and finds the nearest neighbors to the user's favorite watched movie by cosine similarity (scipy.spatial.distance.cosine).
Collaborative filtering — builds a user × movie rating matrix and uses scipy.stats.pearsonr to find users with correlated taste, then recommends what those similar users rated highly. Requires at least 4 commonly-rated movies before trusting a correlation — with only 2 shared ratings, Pearson correlation is mathematically guaranteed to be exactly ±1 (any two points form a perfect line), which is a statistical artifact, not a real taste signal.
Cluster-based filtering — uses scipy.cluster.vq.kmeans2 to group users by their genre-preference vectors, then surfaces each cluster's most-agreed-upon movies — ranked by how many cluster members actually rated a title, not just its single highest rating (otherwise one person's lucky 5-star always outranks a movie three people agreed was a 4).
Trending fallback — a popularity + Metascore ranking for cold-start users with no watch history yet.
Visualizes insights — six charts combined into one storytelling dashboard (see below), not printed or saved separately.
The Story Dashboard

Rather than a grid of unordered charts, create_story_dashboard() in visualize.py builds a single image structured as a five-chapter narrative, meant to be presented, not just browsed:

Chapter	Charts	Question it answers
01 — The Profile	KPI strip, genre pie, "About this viewer" card	Who is this person, and what do they say they like?
02 — The Hidden Pattern	Genre satisfaction bar, user-vs-critics scatter	What do they actually rate highest — and are they a harsher or more generous rater than IMDB's audience?
03 — The Journey	Rating trend line	Has their taste shifted over time?
04 — The Tribe	Taste-correlation heatmap, k-means cluster map	Who shares this viewer's taste, and how does the whole user base group up?
05 — The Recommendation	Recommended movie cards + synthesis paragraph	The payoff: real picks for this viewer, and a plain-language summary tying every chapter together.

The headline at the top is generated dynamically from that specific user's data — it leads with whichever finding is most presentation-worthy: a stated-preference-vs-actual-behavior surprise if one exists, otherwise a strong peer correlation, otherwise a clear critics-comparison verdict. A different target user produces a genuinely different headline and closing argument, because both are computed, not templated.

Project structure
movie_recommender/
├── IMDB-Movie-Data.csv         # raw 1000-movie IMDB export (title, genre,
│                                # director, year, runtime, rating, votes, metascore)
├── data_simulator.py          # loads + cleans the catalog, synthetic user generator
├── recommendation_engine.py   # content-based, cosine, Pearson, k-means, trending
├── visualize.py                # all chart logic + the 5-chapter story dashboard
├── main.py                     # end-to-end demo script
├── requirements.txt
└── README.md
Setup
Option A — standalone script
bash
pip install -r requirements.txt
python main.py
Option B — Google Colab (cell-by-cell)

Each file's code goes in its own cell, run top to bottom, in this order:

Fetch the CSV: pd.read_csv("https://raw.githubusercontent.com/<user>/<repo>/main/IMDB-Movie-Data.csv") (or !wget it into the Colab filesystem)
data_simulator.py's code
recommendation_engine.py's code
visualize.py's code
main.py's code

Important: cells 2–5 must have their from data_simulator import ... / from recommendation_engine import ... / from visualize import ... lines removed before pasting. In separate .py files those imports reach a real file on disk; in a notebook, everything already shares one memory space once the earlier cell has run, so the import just fails looking for a file that doesn't exist (ModuleNotFoundError: No module named 'data_simulator'). Only the first import line in each file (the external library ones — pandas, numpy, scipy, matplotlib) should stay; the project's own cross-file imports should go.

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
A short plain-language insights summary

It also saves dashboard.png — the full five-chapter story dashboard — to the working directory.

Using it in your own code
python
from data_simulator import generate_users
from recommendation_engine import (
    content_based_recommend, similar_to_favorite, get_trending,
    build_user_item_matrix, collaborative_recommend, cluster_users,
)
from visualize import create_story_dashboard

users = generate_users(n_users=12, history_len=25)
target = users[0]

# Content-based (genre + rating + metascore + popularity + recency)
recs = content_based_recommend(target, n=5)

# Nearest-neighbor by content vector, cosine similarity
favorite, similar = similar_to_favorite(target, n=3)

# Collaborative (Pearson, min 4 shared ratings required)
matrix = build_user_item_matrix(users)
collab_recs = collaborative_recommend(matrix, target["name"], n=5)

# Clustering
clusters = cluster_users(users, k=3)

# Cold-start fallback
trending = get_trending(n=5)

# The full 5-chapter presentation dashboard
create_story_dashboard(target, users, matrix, clusters, recs, "dashboard.png")

Individual chart functions (plot_genre_pie, plot_rating_trend, plot_genre_satisfaction, plot_user_vs_critics, plot_taste_correlation_heatmap, plot_user_clusters) are still available in visualize.py if you want any one chart at full size on its own, outside the combined dashboard.

Design notes / talking points for judges
Why a real dataset? 1000 real movies (vs. a 30-title hardcoded list) make every recommendation demoable and let correlation/clustering signals actually mean something — with more titles and richer metadata, similar- taste patterns emerge instead of being coincidence.
Why regression-based imputation for missing Metascores? A flat average throws away information a real predictor variable already gives you for free. Rating and Metascore are correlated (r=0.63); using that relationship to fill gaps is more honest than pretending every unscored movie is exactly average.
Why popularity-weighted watch histories? With 1000 movies and each simulated user watching only ~25, uniform random sampling makes it very unlikely two users ever watched the same movie — starving Pearson correlation of anything to compare. Weighting by vote count (people mostly watch what's popular) keeps realistic overlap without hand-fixing the data.
Why cosine similarity on content vectors? Genre-string equality ("same genre = similar") is brittle. Representing each movie as a vector (genres + era + quality + popularity) and measuring the angle between vectors captures how similar, not just same or not.
Why require 4+ shared ratings for Pearson correlation, not 2? With exactly 2 data points, Pearson correlation is mathematically guaranteed to be exactly ±1 — a statistical artifact, not evidence of matching taste. Requiring more overlap before trusting a correlation is what makes the "closest taste match" KPI meaningful instead of misleading.
Why rank cluster "top movies" by vote count first, rating second? In a sparse catalog, most watched movies were only ever rated by one person in a cluster — sorting by average rating alone would let a single lucky 5-star always outrank a movie three people agreed was a 4. That's one person's opinion dressed up as group consensus.
Why a storytelling dashboard instead of a chart grid? A hackathon audience needs the "so what," not just the data — the dashboard argues toward a conclusion (the recommendation) instead of displaying six independent facts and leaving the interpretation to the viewer.
Extending to real GenAI: swap content_based_recommend's ranking for an LLM call that explains why a movie fits ("since you liked Inception's layered plotting, you'll enjoy Arrival's structural twist..."), or embed movie synopses/descriptions and do semantic similarity search on top of (or instead of) the genre-based content vectors. The dashboard's auto-generated headline/synthesis text is already doing a simple version of this — a real LLM call would make it richer.
Where this would plug into a real system: generate_users is the only synthetic piece — swap it for a production ratings database, and the catalog, recommendation, and visualization layers work unchanged.
Content
IMDB-Movie-Data.csv

CSV

IMDB-Movie-Data.csv

CSV

output1.png

PNG

Cloning into 'AI-Powered-Movie-Recommendation-System'... remote: Enumerating objects: 33, done. remote: Counting objects: 100% (33/33), done. remote: Compressing objects: 100% (33/33), done. remote: Total 33 (delta 15), reused 2 (delta 0), pack-reused 0 (from 0) Receiving objects: 100% (33/33),

PASTED

============================================================ TRENDING (popularity fallback for brand-new users) ============================================================ -> Gravity (2013, rating 7.8, metascore 96.0, 622,089 votes) -> The Dark Knight (2008, rating 9.0, metascore 82.0, 1

PASTED

============================================================ TRENDING (popularity fallback for brand-new users) ============================================================ -> Gravity (2013, rating 7.8, metascore 96.0, 622,089 votes) -> The Dark Knight (2008, rating 9.0, metascore 82.0, 1

PASTED
