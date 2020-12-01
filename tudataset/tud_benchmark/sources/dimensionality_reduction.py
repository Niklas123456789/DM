
from matplotlib import pyplot as plt
from sklearn.decomposition import KernelPCA, TruncatedSVD
import numpy as np



def kernelPCA(dataset):
    kpca = KernelPCA(n_components=2, kernel="precomputed")
    reduced_kpca = kpca.fit_transform(dataset)
    return reduced_kpca




def truncatedSVD(dataset):
    tsvd = TruncatedSVD(n_components=100)
    reduced_tsvd = tsvd.fit_transform(dataset)
    return reduced_tsvd




def svd_energy(dataset):
    tsvd = TruncatedSVD(n_components=100)
    bla = tsvd.fit_transform(dataset)
    x = np.round((tsvd.singular_values_**2)/np.sum(tsvd.singular_values_**2),decimals=3)
    x = np.add.accumulate(x)
    return np.argmax(x>=0.9)


def kpca_energy(dataset,n):
    print('test')
