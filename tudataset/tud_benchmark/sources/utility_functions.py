from scipy.sparse import load_npz
import numpy as np


def load_csv(path):
    """
    This function loads an csv file with a given path
    """
    return np.loadtxt(path, delimiter=";")


def load_sparse(path):
    """
    This function loads an npz file with a given path
    """
    return load_npz(path)

