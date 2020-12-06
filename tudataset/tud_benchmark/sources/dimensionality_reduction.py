from sklearn.decomposition import KernelPCA, TruncatedSVD
import numpy as np


def kernelPCA(dataset, n_components):
    '''
    Kernel principal component analysis (kernel PCA)[1] is an extension of principal component
    analysis (PCA) using techniques of kernel methods. Using a kernel, the originally linear
    operations of PCA are performed in a reproducing kernel Hilbert space.
    Source: https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
    '''
    kpca = KernelPCA(n_components=n_components, kernel="precomputed")
    reduced_kpca = kpca.fit_transform(dataset)
    return reduced_kpca


def truncatedSVD(dataset, n_components):
    '''
    Truncated SVD is approximating a rectangular matrix requires using something more general
    than eigenvalues and eigenvectors, and that is singular values and singular vectors.
    Source: https://www.johndcook.com/blog/2018/05/07/truncated-svd/
    '''
    tsvd = TruncatedSVD(n_components=n_components)
    reduced_tsvd = tsvd.fit_transform(dataset)
    return reduced_tsvd


def svd_energy(dataset, n_components):
    '''
    After applying truncated SVD each principal component has its own energy value (singular
    values identify this energy). This function returns only that many principal components,
    that 90% of the total energy is retained.
    '''
    tsvd = TruncatedSVD(n_components=n_components)
    bla = tsvd.fit_transform(dataset)
    x = np.round((tsvd.singular_values_ ** 2) / np.sum(tsvd.singular_values_ ** 2), decimals=3)
    x = np.add.accumulate(x)
    return np.argmax(x >= 0.9)
