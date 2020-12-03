
from matplotlib import pyplot as plt
from sklearn.decomposition import KernelPCA, TruncatedSVD
import numpy as np



def kernelPCA(dataset, n_components):
    kpca = KernelPCA(n_components=n_components, kernel="precomputed")
    reduced_kpca = kpca.fit_transform(dataset)
    return reduced_kpca




def truncatedSVD(dataset, n_components):
    tsvd = TruncatedSVD(n_components=n_components)
    reduced_tsvd = tsvd.fit_transform(dataset)
    energyIndex = svd_energy(dataset, n_components)
    return reduced_tsvd




def svd_energy(dataset, n_components):
    tsvd = TruncatedSVD(n_components=n_components)
    bla = tsvd.fit_transform(dataset)
    x = np.round((tsvd.singular_values_**2)/np.sum(tsvd.singular_values_**2),decimals=3)
    x = np.add.accumulate(x)
    return np.argmax(x>=0.9)


def kpca_energy(dataset,n):
    print('test')
