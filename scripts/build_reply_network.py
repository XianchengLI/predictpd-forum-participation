"""Directed reply network and the five engagement measures (Methods). Output: forum_user_centrality_full.csv"""

import networkx as nx
import pandas as pd

from paths import PROC, read_forum

forum = read_forum()

author_of = forum.set_index("IDHash")["CreatedBy"]
replies = forum[forum["ReplyingToPost"].notna()].copy()
replies["target"] = replies["ReplyingToPost"].map(author_of)
n_unmatched = int(replies["target"].isna().sum())
replies = replies.dropna(subset=["target"])
edges = replies[replies["CreatedBy"] != replies["target"]][["CreatedBy", "target"]]

G = nx.DiGraph()
G.add_nodes_from(forum["CreatedBy"].unique())
G.add_edges_from(edges.itertuples(index=False, name=None))
U = G.to_undirected()
print(f"Posts: {len(forum)}; accounts: {G.number_of_nodes()}; replies: {len(replies)} "
      f"(unmatched parent: {n_unmatched}); directed edges: {G.number_of_edges()}")

out = pd.DataFrame({"user_hash": list(G.nodes)})
out["post_count"] = out["user_hash"].map(forum.groupby("CreatedBy").size())
out["in_degree"] = out["user_hash"].map(dict(G.in_degree()))
out["betweenness"] = out["user_hash"].map(nx.betweenness_centrality(U))
out["closeness"] = out["user_hash"].map(nx.closeness_centrality(U))
out["clustering"] = out["user_hash"].map(nx.clustering(U))
out = out.sort_values("post_count", ascending=False).reset_index(drop=True)
out.to_csv(PROC / "forum_user_centrality_full.csv", index=False)
print(f"Saved {PROC / 'forum_user_centrality_full.csv'} ({len(out)} accounts)")
