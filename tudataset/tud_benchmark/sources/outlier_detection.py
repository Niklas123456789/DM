
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
def local_outlier_factor(data):
    print('test')
    clf = LocalOutlierFactor(n_neighbors=2)
    labels = clf.fit_predict(data)
    return labels



def elliptic_envelope(data):
    cov = EllipticEnvelope(random_state=0).fit_predict(data)
    return cov


def isolation_forest(data,contamination):
    forest = IsolationForest(contamination=contamination)
    labels = forest.fit_predict(data)
    return labels



def one_class_svm(data,nu):
    labels = OneClassSVM(nu=nu).fit_predict(data)
    return labels


def pandas_outlier_detection(data):
    df = pd.DataFrame(data)

    from scipy import stats
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df.values


def db_scan_outlier_detection():
    print('test')