import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import community as community_louvain
from sklearn.metrics import adjusted_rand_score

# Define distance threshold (in meters)
threshold = 20000
# Determine the number of runs
num_runs = 10

df = pd.read_csv('data/Nodes_CD6huan.csv') 
# Use KDTree to build a spatial index
tree = KDTree(df[["cgcs_x","cgcs_y"]].values)

# Build the graph
G = nx.Graph()

# Add edges to each node
for index, row in df.iterrows():
    distances, indices = tree.query([row['cgcs_x'], row['cgcs_y']], k=len(df))
    total_distance = 0
    for dist, idx in zip(distances[1:], indices[1:]):  # Skip the first index since it is the node itself
        if total_distance + dist > threshold:
            break
        if idx != index:  # Avoid self-loops
            node1_id = row['station_id']
            node2_id = df.iloc[idx]['station_id']
            weight = (1-row['rate']) * (1-df.loc[idx, 'rate'])
            G.add_edge(node1_id, node2_id, weight=weight)
            total_distance += dist

# # Save the graph to a CSV file
edge_data = nx.to_pandas_edgelist(G)
edge_data.to_csv('data/test_edges_20k.csv', index=False)


# Store the results of each run
all_partitions = []

# Run the Louvain algorithm multiple times
for _ in range(num_runs):
    partition = community_louvain.best_partition(G,weight='weight',resolution=1.0)
    # Convert partitions to lists of labels
    labels = [partition[node] for node in G.nodes()]
    all_partitions.append(labels)

# Convert partitions to lists of labels
ari_matrix = np.zeros((num_runs, num_runs))
for i in range(num_runs):
    for j in range(num_runs):
        ari_matrix[i, j] = adjusted_rand_score(all_partitions[i], all_partitions[j])

# You can further analyze the consistency of community partitioning based on ari_matrix
# For example, calculate the average ARI
mean_ari = np.mean(ari_matrix)
max_ari = np.amax(ari_matrix)


# Since the ARI matrix is ​​symmetric, we only consider the upper triangular matrix
upper_tri_ari = np.triu(ari_matrix, k=1)
max_ari_index = np.unravel_index(np.argmax(upper_tri_ari, axis=None), upper_tri_ari.shape)


# Select the community partition result of the highest ARI run
best_partition_labels = all_partitions[max_ari_index[1]]

# Convert the result to DataFrame
best_partition_df = pd.DataFrame({'station_id': G.nodes(), 'Community': best_partition_labels})
# Sort by Station_ID in ascending order
# best_partition_df = best_partition_df.sort_values(by='station_id')
# Save as CSV
best_partition_df.to_csv(f'data/result/community_{max_ari}_{num_runs}_20k.csv', index=False)

# Initialize the community statistics list
community_stats = []

# For each community, calculate the modularity, average center position, average edge weight, and number of community members
for community in set(best_partition_labels):
    # members = [node for node in G.nodes() if best_partition_labels[node] == community]
    # subgraph = G.subgraph(members)
    members = [node for idx, node in enumerate(G.nodes()) if best_partition_labels[idx] == community]
    subgraph = G.subgraph(members)

    total_weight = sum(weight for _, _, weight in G.edges(data='weight'))
    # Calculate the internal weight (sum of weights of edges within the community)
    internal_weight = sum(weight for _, _, weight in subgraph.edges(data='weight'))

    # Calculate the sum of degrees for the nodes in the subgraph
    subgraph_degree_sum = sum(d for _, d in subgraph.degree(weight='weight'))

    # Calculate the sum of degrees for all nodes in the original graph
    total_degree_sum = sum(d for _, d in G.degree(weight='weight'))

    # Calculate expected weight
    expected_weight = (subgraph_degree_sum ** 2) / (2 * total_weight)

    # Calculate the modularity contribution for the community
    modularity_contribution = (internal_weight - expected_weight) / total_weight

    member_count = len(members)
    avg_position = np.mean([df[df['station_id'] == member][['cgcs_x', 'cgcs_y']].values[0] for member in members], axis=0)
    avg_center_x, avg_center_y = avg_position
    # Calculate spatial dispersion (average distance of members from the community center)
    distances = [np.linalg.norm(df[df['station_id'] == member][['cgcs_x', 'cgcs_y']].values[0] - avg_position) for member in members]
    spatial_dispersion = np.mean(distances)

    avg_weight = np.mean([data['weight'] for _, _, data in subgraph.edges(data=True)]) if subgraph.size(weight='weight') > 0 else 0
    
    community_stats.append({
        'Community': community,
        'Modularity Contribution': modularity_contribution,
        'Member Count': member_count,
        'Average Position': avg_position,
        'Average Center X': avg_center_x,
        'Average Center Y': avg_center_y,
        'Spatial Dispersion': spatial_dispersion,
        'Average Weight': avg_weight
    })

community_stats_df = pd.DataFrame(community_stats)
filtered_community = community_stats_df[community_stats_df['Member Count'] > 1]

# Save community statistics to CSV file
filtered_community.to_csv(f'data/result/test_stats_{num_runs}_20k.csv', index=False)

