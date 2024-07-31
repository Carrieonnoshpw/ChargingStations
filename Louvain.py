import networkx as nx
import numpy as np
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_weighted_graph(edges):
    """
    根据给定的边和权重创建加权图。

    参数:
    edges (list of tuples): 每个元组包含两个节点和它们之间的权重 (node1, node2, weight)。
    例:edges = [(1, 2, 0.5), (2, 3, 1.5), (3, 4, 2.5), (4, 1, 1.0)]

    返回:
    G (NetworkX Graph): 一个 NetworkX 图，其中包含给定的边和权重。
    """
    # 创建一个空的无向图
    G = nx.Graph()

    # 为图添加边和权重
    for edge in edges:
        node1, node2, weight = edge
        G.add_edge(node1, node2, weight=weight)

    return G

def louvain_community_detection(G, resolution=1.0):
    """
    对给定的图G执行Louvain社团发现算法。
    
    :param G: NetworkX图对象
    :param resolution: 分辨率参数，影响社团的大小
    :return: 字典，键为节点，值为分配给每个节点的社团编号
    """
    # 使用Louvain算法计算最佳社团划分
    partition = community_louvain.best_partition(G, resolution=resolution)
    return partition

# 示例：创建一个简单的图
# G = nx.erdos_renyi_graph(30, 0.05)

def ll(G):

    # 执行Louvain社团发现算法
    communities = louvain_community_detection(G)

    # 打印结果
    # for node, community in communities.items():
    #     print(f"Node {node} is in community {community}")

    # 绘制网络图
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G)  # 设置网络图的布局

    # Use numpy to generate a specified number of colors from the 'viridis' colormap
    num_colors = max(communities.values()) + 1
    colors = plt.cm.viridis(np.linspace(0, 1, num_colors))

    # Create a new ListedColormap with the specified number of colors
    cmap = mcolors.ListedColormap(colors)

    # Draw nodes with colors based on their community membership
    node_colors = [communities[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, cmap=cmap, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_colors-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Community ID')

    plt.title('Community Detection using Louvain Algorithm')
    plt.show()