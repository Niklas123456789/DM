import os
import numpy as np
import sys
import networkx as nx
import math as m
from matplotlib import pyplot as plt
from nrkmeans import NrKmeans
from sklearn.decomposition import KernelPCA, TruncatedSVD

sys.path.append('.')
from tudataset.tud_benchmark.auxiliarymethods import datasets as dp
from tudataset.tud_benchmark.auxiliarymethods.reader import tud_to_networkx
from tudataset.tud_benchmark.auxiliarymethods import auxiliary_methods as aux

#from tudataset.tud_benchmark.auxiliarymethods import datasets as dp
#from tudataset.tud_benchmark import auxiliarymethods
base_path = os.path.join("graph_representations", "without_labels")
ds_name = "IMDB-BINARY"
classes = dp.get_dataset("IMDB-BINARY")



### OTHER FUNCTIONS ###
# utility functions
def load_csv(path):
    return np.loadtxt(path, delimiter=";")

#def load_sparse(path):
#    return load_npz(path)

def select_from_list(l, indices):
    return [l[i] for i in indices]

def visualize(G, color=None, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, 
                     pos=nx.spring_layout(G, seed=42),
                     with_labels=True,
                     node_color=color,
                     cmap="Set2")
    plt.show();

# Cosine normalization for a gram matrix.
def normalize_gram_matrix(gram_matrix):
    n = gram_matrix.shape[0]
    gram_matrix_norm = np.zeros([n, n], dtype=np.float64)

    for i in range(0, n):
        for j in range(i, n):
            if not (gram_matrix[i][i] == 0.0 or gram_matrix[j][j] == 0.0):
                g = gram_matrix[i][j] / m.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
                gram_matrix_norm[i][j] = g
                gram_matrix_norm[j][i] = g

    return gram_matrix_norm


print(classes)
G = tud_to_networkx(ds_name)
print(f"Number of graphs in data set is {len(G)}")
print(f"Number of classes {len(set(classes.tolist()))}")

#Let's start by plotting a single graph of our data set
idx = 0
visualize(G[idx])

#Plot the vector representation with KernelPCA
iterations = 5
gram = load_csv(os.path.join(base_path,f"{ds_name}_gram_matrix_wl{iterations}.csv"))
gram = aux.normalize_gram_matrix(gram)


kpca = KernelPCA(n_components=100, kernel="precomputed")
reduced_kpca = kpca.fit_transform(gram)
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(reduced_kpca[:,0], reduced_kpca[:,1], c=classes, s=1)
plt.show();

#print(os.path.isfile("tudataset/tud_benchmark/datasets/IMDB-BINARY/IMDB-BINARY/raw/IMDB-BINARY_graph_labels.txt"))
#gram = np.loadtxt("tudataset/tud_benchmark/datasets/IMDB-BINARY/IMDB-BINARY/raw/IMDB-BINARY_graph_labels.txt")
#print(gram.shape)