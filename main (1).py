"""
main.py
--------
End-to-end demo of the AI-Powered Movie Recommendation System.

Run:
    python main.py

This will:
  1. Simulate a set of users with preferences and watch histories.
  2. Generate content-based, "since you liked X", collaborative (Pearson),
     and cluster-based (k-means) recommendations for a target user.
  3. Save two visualizations (genre pie chart + rating trend) as PNGs.
"""

import json

from data_simulator import generate_users
from recommendation_engine import (
    content_based_recommend,
    similar_to_favorite,
    build_user_item_matrix,
    pearson_similar_users,
    collaborative_recommend,
    cluster_users,
    cluster_top_movies,
)
from visualize import plot_genre_pie, plot_rating_trend


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    # 1. Simulate users -----------------------------------------------------
    users = generate_users(n_users=10, history_len=8)
    target = users[0]

    print_header(f"USER PROFILE: {target['name']}")
    print(json.dumps({k: v for k, v in target.items() if k != "watch_history"}, indent=2))
    print(f"\nWatch history ({len(target['watch_history'])} movies):")
    for w in target["watch_history"]:
        print(f"  - {w['movie']:<28} genre={w['genre']:<10} "
              f"director={w['director']:<22} rating={w['rating']} date={w['date']}")

    # 2. Content-based recommendations --------------------------------------
    print_header("CONTENT-BASED RECOMMENDATIONS (genre match, ranked by avg rating)")
    for m in content_based_recommend(target, n=5):
        print(f"  -> {m['title']} ({m['genre']}, avg rating {m['avg_rating']})")

    favorite, similar = similar_to_favorite(target, n=3)
    if favorite:
        print(f"\nSince you liked '{favorite['movie']}' "
              f"(you rated it {favorite['rating']}/5), you might enjoy:")
        for m in similar:
            print(f"  -> {m['title']}  (dir. {m['director']}, genre {m['genre']})")

    # 3. Collaborative filtering (Pearson correlation) ----------------------
    print_header("COLLABORATIVE FILTERING (Pearson correlation between users)")
    matrix = build_user_item_matrix(users)
    similar_users = pearson_similar_users(matrix, target["name"])
    if similar_users:
        print("Most similar users by rating correlation:")
        for name, corr in similar_users[:5]:
            print(f"  {name:<10} correlation = {corr:.2f}")
        print(f"\nRecommended for {target['name']} based on similar users:")
        for rec in collaborative_recommend(matrix, target["name"], n=5):
            print(f"  -> {rec['title']} (predicted score {rec['predicted_score']})")
    else:
        print("Not enough overlapping ratings between users to compute correlations.")

    # 4. K-means clustering ---------------------------------------------------
    print_header("K-MEANS CLUSTERING OF USERS BY GENRE PREFERENCE")
    clusters = cluster_users(users, k=3)
    top_by_cluster = cluster_top_movies(clusters, users, n=5)
    for cluster_id, names in clusters.items():
        print(f"\nCluster {cluster_id}: {names}")
        print("  Top-rated movies in this cluster:")
        for m in top_by_cluster[cluster_id]:
            print(f"    -> {m['title']} (avg {m['avg_cluster_rating']}, {m['votes']} votes)")

    # 5. Visualizations -------------------------------------------------------
    print_header("VISUALIZATIONS")
    pie_path = plot_genre_pie(target, "genre_distribution.png")
    trend_path = plot_rating_trend(target, "rating_trend.png")
    print(f"Saved: {pie_path}")
    print(f"Saved: {trend_path}")


if __name__ == "__main__":
    main()
