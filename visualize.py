"""
visualize.py
-------------
Generates the two required visualizations:
  1. Pie chart of genre distribution in a user's watch history.
  2. Line plot of that user's ratings over time.
 
Plots are saved as PNG files rather than shown interactively, since this
is meant to run headless in a hackathon demo/CI environment.
"""
 
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime
 
from data_simulator import MOVIES
 
_TITLE_TO_GENRES = {m["title"]: m["genres"] for m in MOVIES}
 
 
def plot_genre_pie(user: dict, out_path: str = "genre_distribution.png"):
    # Real movies carry multiple genres (e.g. Stardust = Adventure/Family/
    # Fantasy/Romance). Counting only the primary genre would misrepresent
    # a user's taste, so every genre a watched movie carries gets a vote.
    genres = []
    for w in user["watch_history"]:
        genres.extend(_TITLE_TO_GENRES.get(w["movie"], [w["genre"]]))
    counts = Counter(genres)
 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts.values(),
        labels=counts.keys(),
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    ax.set_title(f"Genres Watched by {user['name']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
 
 
def plot_rating_trend(user: dict, out_path: str = "rating_trend.png"):
    history = sorted(user["watch_history"], key=lambda w: w["date"])
    dates = [datetime.strptime(w["date"], "%Y-%m-%d") for w in history]
    ratings = [w["rating"] for w in history]
    titles = [w["movie"] for w in history]
    n = len(history)
 
    # Wider figure and smaller/staggered labels as history grows, so titles
    # don't collide once a user has watched dozens of movies.
    fig_width = max(9, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.plot(dates, ratings, marker="o", linestyle="-", color="#4C72B0")
 
    font_size = 7 if n <= 15 else 6
    # Beyond ~20 points, only label a spaced-out subset (always including
    # the highest and lowest rated movie) to keep the chart legible.
    if n > 20:
        label_idx = set(range(0, n, 2))
        label_idx.add(int(np.argmax(ratings)))
        label_idx.add(int(np.argmin(ratings)))
    else:
        label_idx = set(range(n))
 
    for i, (x, y, t) in enumerate(zip(dates, ratings, titles)):
        if i not in label_idx:
            continue
        y_offset = 10 if i % 2 == 0 else -16  # alternate above/below the line
        va = "bottom" if i % 2 == 0 else "top"
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(0, y_offset),
                    fontsize=font_size, rotation=20, ha="left", va=va)
 
    ax.set_ylim(0.3, 5.7)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Rating")
    ax.set_xlabel("Date")
    ax.set_title(f"Rating Trend Over Time - {user['name']}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
