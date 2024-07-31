import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import community as community_louvain
from sklearn.metrics import adjusted_rand_score

df = pd.read_csv('data/Nodes_CD6huan.csv') 

# 使用KDTree构建空间索引
tree = KDTree(df[["cgcs_x","cgcs_y"]].values)

# 定义距离阈值（50km）
threshold = 50000

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
# edge_data = nx.to_pandas_edgelist(G, weight='weight')
# edge_data.to_csv('data/graphedges_CD6huan50K.csv', index=False)

# 确定运行次数
num_runs = 10
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
best_partition_df.to_csv(f'data/best_community_partition_{max_ari}_{num_runs}r0.csv', index=False)

print("已保存具有最高一致性得分的社团划分结果")
# 将社团结果添加到DataFrame
# community_df = pd.DataFrame.from_dict(partition, orient='index', columns=['Community'])
# community_df.reset_index(inplace=True)
# community_df.rename(columns={'index': 'Station_ID'}, inplace=True)

# # 保存社团结果到CSV文件
# community_df.to_csv('data/community_CD6huan50K1.csv', index=False)
