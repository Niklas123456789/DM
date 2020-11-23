
from matplotlib import pyplot as plt
from sklearn.decomposition import KernelPCA, TruncatedSVD
import numpy as np



def kernelPCA(dataset,classes,title):
    kpca = KernelPCA(n_components=10, kernel="precomputed")
    reduced_kpca = kpca.fit_transform(dataset)
    return reduced_kpca




def truncatedSVD(dataset,classes,title):
    tsvd = TruncatedSVD(n_components=10)
    reduced_tsvd = tsvd.fit_transform(dataset)
    return reduced_tsvd




def svd_test(dataset):
    tsvd = TruncatedSVD(n_components=10)
    bla = tsvd.fit_transform(dataset)
    print(tsvd.singular_values_)
    x = np.round((tsvd.singular_values_**2)/np.sum(tsvd.singular_values_**2),decimals=3),
    print(x[0:10])
