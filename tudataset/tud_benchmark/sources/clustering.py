from sklearn.cluster import KMeans
from copy import deepcopy
from matplotlib import  pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import pandas as pd
import seaborn as sns


def kMeans_scree_plot(dataset):
    '''
    Scree plot is a line plot of the eigenvalues of factors or principal components analysis.
    A scree plot always displays the eigenvalues in a downward curve, ordering the eigenvalues
    from largest to smallest. According to the scree test, the "elbow" of the graph where the
    eigenvalues seem to level off is found and factors or components to the left of this point
    should be retained as significant.
    Source: https://en.wikipedia.org/wiki/Scree_plot
    '''
    dataset = np.delete(dataset, range(11, 99), 1)
    costs = []
    kmeans_runs = []
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40]
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



def kMeans(dataset,n_clusters):
    '''
    k-means clustering is a method of vector quantization, originally from signal processing,
    that aims to partition n observations into k clusters in which each observation belongs to
    the cluster with the nearest mean, serving as a prototype of the cluster.
    Source: https://en.wikipedia.org/wiki/K-means_clustering
    '''
    kmeans = KMeans(n_clusters=n_clusters).fit(dataset)
    return kmeans.labels_




def calculateNMI(labels1,labels2):
    '''
    Calulates the NMI between to label arrays, adds the labels of the absolute difference to
    the shorter array so both arrays are the same length
    '''

    label_diff = abs(labels1.shape[0]-labels2.shape[0])
    if labels1.shape[0]>labels2.shape[0]:
        labels2 = np.append(labels2, labels1[-label_diff:])
    else:
        labels1 = np.append(labels1, labels2[-label_diff:])

    return normalized_mutual_info_score(labels1, labels2)


def showPairPlot(dataset,labels,title):
    '''
    A pairplot plot a pairwise relationships in a dataset. The pairplot function creates a grid
    of Axes such that each variable in data will by shared in the y-axis across a single row
    and in the x-axis across a single column.
    Source: https://pythonbasics.org/seaborn-pairplot/
    '''
    df = pd.DataFrame(dataset)
    df['Cluster'] = labels
    g = sns.pairplot(df,hue="Cluster",diag_kind='hist',palette='Accent')
    g.fig.suptitle(title)
    plt.show()

