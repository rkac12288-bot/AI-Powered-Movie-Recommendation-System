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
from collections import Counter
from datetime import datetime


def plot_genre_pie(user: dict, out_path: str = "genre_distribution.png"):
    genres = [w["genre"] for w in user["watch_history"]]
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

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates, ratings, marker="o", linestyle="-", color="#4C72B0")
    for x, y, t in zip(dates, ratings, titles):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=7, rotation=20, ha="left")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Rating")
    ax.set_xlabel("Date")
    ax.set_title(f"Rating Trend Over Time - {user['name']}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
