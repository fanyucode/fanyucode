import os
import pandas as pd
import numpy as np
import matrixprofile.algorithms as mpg
from sklearn.cluster import KMeans
import networkx as nx
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import davies_bouldin_score
# from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# 读取文件夹中的所有csv文件，每个csv代表一条时间序列
folder_path = './data/apartment_month copy'
data = []
windowsize=96
n_files=108
n_cluster=3
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)
        data.append(df['load'].values)

train_size=int(0.8*len(data))
train_data=data[:train_size]
test_data = data[train_size:]

# 计算MatrixProfile
train_mp = []
for d in train_data:
    train_mp.append(mpg.stomp(d, windowsize))
# 将MatrixProfile转换为NumPy数组
train_mp_array = [np.array(m['mp']) for m in train_mp]

# 计算距离矩阵
dist_matrix = np.zeros((len(train_mp_array), len(train_mp_array)))
for i in range(len(train_mp_array)):
    for j in range(i+1, len(train_mp_array)):
        dist = mpg.mpdist(train_mp_array[i], train_mp_array[j],windowsize)
        dist_matrix[i, j] = dist
        
# train_data聚类
kmeans = KMeans(n_clusters=n_cluster, random_state=None,max_iter=300,n_init=42,init='k-means++')
kmeans.fit(dist_matrix)

#在测试集上计算召回率和F1 score
test_mp=[]
for d in test_data:
    test_mp.append(mpg.stomp(d, windowsize))
test_mp_array = [np.array(m['mp']) for m in test_mp]

test_labels = []
for i in range(len(test_mp_array)):
    distances = []
    for j in range(len(train_mp_array)):
        dist = mpg.mpdist(test_mp_array[i], train_mp_array[j],windowsize)
        distances.append(dist)
    label = kmeans.predict(np.array(distances).reshape(1, -1))[0]
    test_labels.append(label)


true_labels = [i for i in range(n_cluster)] * (len(test_data) // n_cluster)
recall = recall_score(true_labels, test_labels, average="macro")
f1 = f1_score(true_labels, test_labels, average="macro")
print("recall：", recall)
print("f1：", f1)