import pandas as pd
from sklearn.metrics import adjusted_rand_score
# 读取两个CSV文件
df1 = pd.read_csv('data/best_community_partition_1.0_10r0.csv')
df2 = pd.read_csv('data/best_community_partition_1.0_10r.csv')
df1.sort_values(by='station_id',inplace=True)
df2.sort_values(by='Station_ID',inplace=True)
# 假设社团划分结果是按节点顺序排列的，我们只需要社团的编号
labels1 = df1['Community'].values
labels2 = df2['Community'].values

# 计算调整兰德指数
ari_score = adjusted_rand_score(labels1, labels2)

# 判断两次社团划分结果是否一致
if ari_score == 1.0:
    print("两次社团划分结果完全一致。")
elif ari_score > 0:
    print("两次社团划分结果大体上相似。",ari_score)
else:
    print("两次社团划分结果不相似。")
