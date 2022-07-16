import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from tabulate import tabulate


# List of metrics
metrics = ["IS*", "FID", "RP", "SOA-C", "SOA-I", "O-IS", "O-FID", "CA", "PA"]

# List of methods
methods = [f.split(".")[0] for f in os.listdir("methods") if f.split(".")[1] == "json"]

# Read and parse the aspect scores
scores = OrderedDict()
for method in methods:
    scores[method] = []
    with open(f"methods/{method}.json", "r") as f:
        method_scores = json.load(f)
    for metric in metrics:
        scores[method].append(float(method_scores[metric]))

# Convert to Numpy to compute ranking scores.
scores_np = []
for method in scores:
    scores_np.append(scores[method])
scores_np = np.array(scores_np)

# Get rank
# Change sign of metrics which are lower is better for ranking purpose.
scores_np[:, 1] = -scores_np[:, 1]  # FID
scores_np[:, 6] = -scores_np[:, 6]  # O-FID
scores_np[:, 7] = -scores_np[:, 7]  # CA
relative_ranking_scores = np.argsort(scores_np, 0)

# Get raking scores

num_methods = len(scores)
num_metrics = len(metrics)
ranking_scores = np.zeros([num_methods, num_metrics])
for method_idx in range(num_methods):
    for metric_idx in range(num_metrics):
        ranking_scores[method_idx, metric_idx] = np.where(relative_ranking_scores[:, metric_idx] == method_idx)[0] + 1

metrics.append("RS")
for method_idx in range(num_methods):
    method_scores = ranking_scores[method_idx]
    # Get aspect scores
    aspect_scores = [
        np.mean(method_scores[0:2]),
        method_scores[2],
        np.mean(method_scores[3:5]),
        np.mean(method_scores[5:7]),
        method_scores[7],
        method_scores[8],
    ]
    # Summing all aspect scores to get final ranking score
    rs = np.sum(aspect_scores).item()
    scores[methods[method_idx]].append(rs)

# Convert to numpy to create pandas table for saving.
scores_np = []
for method in scores:
    scores_np.append(scores[method])
scores_np = np.array(scores_np)

# Display results
df = pd.DataFrame(scores_np, columns=metrics)
df.insert(loc=0, column="Method", value=methods)
display_results = tabulate(df, headers="keys", tablefmt="psql", showindex=False)

# Save to file
with open("results/coco_benchmark_results.txt", "w") as f:
    f.write(display_results)
print(display_results)

# To markdown
# print(df.to_markdown(index=False))
