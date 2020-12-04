from sklearn.cluster import KMeans
from copy import deepcopy
from matplotlib import  pyplot as plt
import pandas as pd
import seaborn as sns
def kMeans_scree_plot(dataset):
    dataset = dataset[:,0:10]
    costs = []
    kmeans_runs = []
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15, 20, 25, 30, 35, 40]
    for k in k_list:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dataset)
        costs.append(kmeans.inertia_)
        kmeans_runs.append(deepcopy(kmeans))

    plt.figure(figsize=(10, 5))
    plt.plot(k_list, costs, marker="o")
    plt.xlabel('Number of Clusters')
    plt.ylabel('KMeans loss')
    plt.title('Search for the elbow to determine the optimal k')
    plt.show();


