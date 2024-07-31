import pandas as pd
import numpy as np
import networkx as nx

# Read community discovery, node attributes, and edge CSV files
df_com = pd.read_csv('data/best_community_partition_1.0_100r.csv')
df_node = pd.read_csv('data/Nodes_CD6huan.csv')
df_edge = pd.read_csv('data/graphedges_CD6huan50K.csv')

# Create a graph from the edge list
G = nx.from_pandas_edgelist(df_edge, 'source', 'target', edge_attr='weight')

# Merge community information and node attributes
df = pd.merge(df_node[['station_id','cgcs_x','cgcs_y','rate']], df_com, on='station_id')
df['occupied_rate'] = 1 - df['rate']
# Total weight of the network
total_weight = sum(weight for _, _, weight in G.edges(data='weight'))

# Calculate metrics for each community
community_metrics = {}

# Iterate over each community
for community in df['Community'].unique():
    community_members = df[df['Community'] == community]
    member_ids = set(community_members['station_id'])

    # Only proceed if the community has more than one member
    if len(member_ids) > 1:
        # Calculate average occupied rate and spatial metrics
        avg_occupied_rate = community_members['occupied_rate'].mean()
        avg_center_x = community_members['cgcs_x'].mean()
        avg_center_y = community_members['cgcs_y'].mean()
        distances = np.sqrt((community_members['cgcs_x'] - avg_center_x)**2 + (community_members['cgcs_y'] - avg_center_y)**2)
        spatial_dispersion = distances.mean()

        # Create a partition including the current community and the rest of the graph
        other_nodes = set(G.nodes()) - member_ids
        partition = [list(member_ids), list(other_nodes)]

        # Calculate modularity contribution
        modularity_community = nx.algorithms.community.quality.modularity(G, partition)

        # Calculate average edge weight for the community
        subgraph = G.subgraph(member_ids)
        internal_edges_weight = sum(data['weight'] for _, _, data in subgraph.edges(data=True))
        # Expected weight calculation (E.g., using a random graph model)
        # This is a simplistic assumption; adjust according to your model
        expected_weight = (sum(G.degree(n, weight='weight') for n in member_ids) ** 2) / (2 * total_weight)
        # Modularity contribution calculation
        modularity_contribution = (internal_edges_weight - expected_weight) / total_weight
        if subgraph.size(weight='weight') > 0:
            avg_edge_weight = np.mean([data['weight'] for _, _, data in subgraph.edges(data=True)])
            #avg_edge_weight = sum(data['weight'] for _, _, data in subgraph.edges(data=True)) / subgraph.size(weight='weight')
        else:
            avg_edge_weight = 0

        # Store metricss
        community_metrics[community] = {
            'number_of_members': len(member_ids),
            'average_occupied_rate': avg_occupied_rate,
            'community_center_x': avg_center_x,
            'community_center_y': avg_center_y,
            'spatial_dispersion': spatial_dispersion,
            'community_modularity': modularity_community,
            'community_contribution':modularity_contribution,
            'average_edge_weight': avg_edge_weight
        }

# Convert the metrics dictionary to a DataFrame
community_metrics_df = pd.DataFrame.from_dict(community_metrics, orient='index')

# Save the DataFrame to a CSV file
file_path = 'data/result/community_metrics_best_community_partition_1.0_100r_all.csv'
community_metrics_df.to_csv(file_path, index_label='Community')

# import pandas as pd
# import numpy as np
# import networkx as nx

# # Read community discovery, node attributes, and edge CSV files
# df_com = pd.read_csv('data/best_community_partition_1.0_10r0.csv')
# df_node = pd.read_csv('data/Nodes_CD6huan.csv')
# df_edge = pd.read_csv('data/graphedges_CD6huan50K.csv')

# # Create a graph from the edge list
# G = nx.from_pandas_edgelist(df_edge, 'source', 'target', edge_attr='weight')

# # Merge community information and node attributes
# df = pd.merge(df_node[['station_id','cgcs_x','cgcs_y','rate']], df_com, on='station_id')
# df['occupied_rate'] = 1 - df['rate']

# # Calculate metrics for each community
# community_metrics = {}

# # Iterate over each community
# for community in df['Community'].unique():
#     community_members = df[df['Community'] == community]
#     member_ids = set(community_members['station_id'])
    
#     # Calculate average occupied rate and spatial metrics
#     avg_occupied_rate = community_members['occupied_rate'].mean()
#     avg_center_x = community_members['cgcs_x'].mean()
#     avg_center_y = community_members['cgcs_y'].mean()
#     distances = np.sqrt((community_members['cgcs_x'] - avg_center_x)**2 + (community_members['cgcs_y'] - avg_center_y)**2)
#     spatial_dispersion = distances.mean()
#     # Create a partition including the current community and the rest of the graph
#     other_nodes = set(G.nodes()) - member_ids
#     partition = [list(member_ids), list(other_nodes)]
#     # Calculate modularity contribution
#     modularity_contribution = nx.algorithms.community.quality.modularity(G, partition)
#     # Calculate average edge weight for the community
#     subgraph = G.subgraph(member_ids)
#     if subgraph.size(weight='weight') > 0:
#         avg_edge_weight = sum(data['weight'] for _, _, data in subgraph.edges(data=True)) / subgraph.size(weight='weight')
#     else:
#         avg_edge_weight = 0

#     # Store metrics
#     community_metrics[community] = {
#         'average_occupied_rate': avg_occupied_rate,
#         'community_center_x': avg_center_x,
#         'community_center_y': avg_center_y,
#         'spatial_dispersion': spatial_dispersion,
#         'modularity_contribution': modularity_contribution,
#         'average_edge_weight': avg_edge_weight
#     }

# # Convert the metrics dictionary to a DataFrame
# community_metrics_df = pd.DataFrame.from_dict(community_metrics, orient='index')

# # Save the DataFrame to a CSV file
# file_path = 'data/result/community_metrics_r0.csv'

# community_metrics_df.to_csv(file_path, index_label='Community')
