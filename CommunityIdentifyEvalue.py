import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import community as community_louvain
from sklearn.metrics import adjusted_rand_score

# 定义距离阈值（单位为m）
threshold = 20000
# 确定运行次数
num_runs = 10

#df = pd.read_csv('data/Nodes_CD6huan.csv') 
df = pd.read_csv(r'D:\Users\jsj\ArcGISpro\Station\test_nodes.csv')
# 使用KDTree构建空间索引
tree = KDTree(df[["cgcs_x","cgcs_y"]].values)

# 构建图
G = nx.Graph()

# 为每个节点添加边
for index, row in df.iterrows():
    distances, indices = tree.query([row['cgcs_x'], row['cgcs_y']], k=len(df))
    total_distance = 0
    for dist, idx in zip(distances[1:], indices[1:]):  # 跳过第一个索引，因为它是节点本身
        if total_distance + dist > threshold:
            break
        if idx != index:  # 避免自环
            node1_id = row['station_id']
            node2_id = df.iloc[idx]['station_id']
            weight = (1-row['rate']) * (1-df.loc[idx, 'rate'])
            G.add_edge(node1_id, node2_id, weight=weight)
            total_distance += dist

# # 将图保存到CSV文件
edge_data = nx.to_pandas_edgelist(G)
edge_data.to_csv('data/test_edges_20k.csv', index=False)


# 存储每次运行的结果
all_partitions = []

# 多次运行Louvain算法
for _ in range(num_runs):
    partition = community_louvain.best_partition(G,weight='weight',resolution=1.0)
    # 转换分区为标签列表
    labels = [partition[node] for node in G.nodes()]
    all_partitions.append(labels)

# 计算所有分区之间的ARI
ari_matrix = np.zeros((num_runs, num_runs))
for i in range(num_runs):
    for j in range(num_runs):
        ari_matrix[i, j] = adjusted_rand_score(all_partitions[i], all_partitions[j])

# 可以基于ari_matrix进一步分析社团划分的一致性
# 例如，计算平均ARI
mean_ari = np.mean(ari_matrix)
max_ari = np.amax(ari_matrix)
print("平均调整兰德指数（ARI）:", mean_ari)
print("最大调整兰德指数（ARI）:", max_ari)

# 由于ARI矩阵是对称的，我们只考虑上三角矩阵
upper_tri_ari = np.triu(ari_matrix, k=1)
max_ari_index = np.unravel_index(np.argmax(upper_tri_ari, axis=None), upper_tri_ari.shape)


# 选择ARI最高的一次运行的社团划分结果
best_partition_labels = all_partitions[max_ari_index[1]]

# 将结果转换为DataFrame
best_partition_df = pd.DataFrame({'station_id': G.nodes(), 'Community': best_partition_labels})
# 按照Station_ID升序排序
# best_partition_df = best_partition_df.sort_values(by='station_id')
# 保存为CSV
best_partition_df.to_csv(f'data/result/community_{max_ari}_{num_runs}_20k.csv', index=False)

print("已保存具有最高一致性得分的社团划分结果")

# 初始化社团统计信息列表
community_stats = []

# 对于每个社团，计算模块度、平均中心位置、平均边权重和社团成员数量
for community in set(best_partition_labels):
    # members = [node for node in G.nodes() if best_partition_labels[node] == community]
    # subgraph = G.subgraph(members)
    members = [node for idx, node in enumerate(G.nodes()) if best_partition_labels[idx] == community]
    subgraph = G.subgraph(members)
    # 计算社团模块贡献度

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
    # 计算其他统计信息
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

# 将社团统计信息保存到CSV文件
filtered_community.to_csv(f'data/result/test_stats_{num_runs}_20k.csv', index=False)

