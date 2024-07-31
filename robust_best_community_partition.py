import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import community as community_louvain
from sklearn.metrics import adjusted_rand_score

df = pd.read_csv('data/Nodes_CD6huan.csv') 

# Construct a spatial index using KDTree
tree = KDTree(df[["cgcs_x","cgcs_y"]].values)

# Define distance threshold 
threshold = 50000

# Build the graph
G = nx.Graph()

# Add edges for each node
for index, row in df.iterrows():
    distances, indices = tree.query([row['cgcs_x'], row['cgcs_y']], k=len(df))
    total_distance = 0
    for dist, idx in zip(distances[1:], indices[1:]):  # Skip the first index as it is the node itself
        if total_distance + dist > threshold:
            break
        if idx != index:  # Avoid self-loops
            node1_id = row['station_id']
            node2_id = df.iloc[idx]['station_id']
            weight = (1-row['rate']) * (1-df.loc[idx, 'rate'])
            G.add_edge(node1_id, node2_id, weight=weight)
            total_distance += dist

# # Save the graph to a CSV file
# edge_data = nx.to_pandas_edgelist(G, weight='weight')
# edge_data.to_csv('data/graphedges_CD6huan50K.csv', index=False)

# Determine the number of runs
num_runs = 10
# Store the results of each run
all_partitions = []

# Run the Louvain algorithm multiple times
for _ in range(num_runs):
    partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
    # Convert partition to a list of labels
    labels = [partition[node] for node in G.nodes()]
    all_partitions.append(labels)

# Calculate the ARI between all partitions
ari_matrix = np.zeros((num_runs, num_runs))
for i in range(num_runs):
    for j in range(num_runs):
        ari_matrix[i, j] = adjusted_rand_score(all_partitions[i], all_partitions[j])

# Further analysis of community division consistency based on ari_matrix
# For example, calculate the average ARI
mean_ari = np.mean(ari_matrix)
max_ari = np.amax(ari_matrix)
print("Average Adjusted Rand Index (ARI):", mean_ari)
print("Maximum Adjusted Rand Index (ARI):", max_ari)

# Since the ARI matrix is symmetric, we only consider the upper triangular matrix
upper_tri_ari = np.triu(ari_matrix, k=1)
max_ari_index = np.unravel_index(np.argmax(upper_tri_ari, axis=None), upper_tri_ari.shape)

# Select the community division result with the highest ARI
best_partition_labels = all_partitions[max_ari_index[1]]

# Convert the result to a DataFrame
best_partition_df = pd.DataFrame({'station_id': G.nodes(), 'Community': best_partition_labels})
# Sort by Station_ID in ascending order
# best_partition_df = best_partition_df.sort_values(by='station_id')
# Save as CSV
best_partition_df.to_csv(f'data/best_community_partition_{max_ari}_{num_runs}r0.csv', index=False)

print("The community division result with the highest consistency score has been saved")
# Add community results to the DataFrame
# community_df = pd.DataFrame.from_dict(partition, orient='index', columns=['Community'])
# community_df.reset_index(inplace=True)
# community_df.rename(columns={'index': 'Station_ID'}, inplace=True)

# # Save the community results to a CSV file
# community_df.to_csv('data/community_CD6huan50K1.csv', index=False)
