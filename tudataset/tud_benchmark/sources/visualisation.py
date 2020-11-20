import os
import sys
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.decomposition import KernelPCA, TruncatedSVD
from mpl_toolkits import mplot3d



def kernelPCA(dataset,classes,title):
    kpca = KernelPCA(n_components=100, kernel="precomputed")
    reduced_kpca = kpca.fit_transform(dataset)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    ax.scatter(reduced_kpca[:, 0], reduced_kpca[:, 1], c=classes, s=1)
    plt.show()



def truncatedSVD(dataset,classes,title):
    tsvd = TruncatedSVD(n_components=100)
    reduced_tsvd = tsvd.fit_transform(dataset)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    ax.scatter(reduced_tsvd[:, 0], reduced_tsvd[:, 1], c=classes, s=1)
    plt.show();


def kernelPCA3D(dataset,classes,title):
    print("3d")
    kpca = KernelPCA(n_components=100, kernel="precomputed")
    reduced_kpca = kpca.fit_transform(dataset)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter3D(reduced_kpca[:,0], reduced_kpca[:,1], reduced_kpca[:,2], c=classes)
    plt.show()


