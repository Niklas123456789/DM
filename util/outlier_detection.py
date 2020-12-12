from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
import collections


def local_outlier_factor(data):
    """
    This function returns the labels by an given outlier factor
    """
    clf = LocalOutlierFactor(n_neighbors=2)
    labels = clf.fit_predict(data)
    return labels


def elliptic_envelope(data):
    """
    This function returns the result of the Elliptic Envelope
    """
    cov = EllipticEnvelope(random_state=0).fit_predict(data)
    return cov


def isolation_forest(data, contamination):
    """
    This function returns the labels of the isolation forest of the given data
    """
    forest = IsolationForest(contamination=contamination)
    labels = forest.fit_predict(data)
    return labels


def one_class_svm(data, nu):
    """
    This function returns the labels of svm of the given data
    """
    labels = OneClassSVM(nu=nu).fit_predict(data)
    return labels


def pandas_outlier_detection(data):
    """
    This function returns the labels of the data frame outlaier detection of the given data
    """
    df = pd.DataFrame(data)

    from scipy import stats
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df.values


def detect_outlier(data):
    """
    This function finds the different bounds of a dataset, to determine, which points are outliers.
    Q1 and Q3 are the position where 25% and 75% of the data are placed.
    Iqr is the median of the given datapoints.
    The lower and upper bounds are a reference point, which determines weather a point is an outlier of not.
    """
    # find q1 and q3 values
    q1, q3 = np.percentile(sorted(data), [25, 75])

    # compute IRQ
    iqr = q3 - q1

    # find lower and upper bounds
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = [x for x in data if x <= lower_bound or x >= upper_bound]

    return outliers, lower_bound, upper_bound


def seperateOutliersWithRange(testDataset, minX, maxX, minY, maxY, returnOutliers):
    """
    This function seperates the outliers from the given datapoints of given ranges.
    It returns either the only outliers or the data without outliers, depending on returnOutliers
    """
    outlierIndex = [0] * len(testDataset)

    for i in range(len(testDataset)):
        for j in range(2):
            if j == 0:
                if testDataset[i][0] <= minX or testDataset[i][0] >= maxX:
                    outlierIndex[i] = 1
            if j == 0:
                if testDataset[i][1] <= minY or testDataset[i][1] >= maxY:
                    outlierIndex[i] = 1

    if returnOutliers:
        data = removeData(dataset=testDataset, outliers=outlierIndex)
        return data, outlierIndex
    else:
        data = removeOutliers(dataset=testDataset, outliers=outlierIndex)
        return data, outlierIndex


def seperateOutliers(testDataset, returnOutliers):
    """
    This function seperates the outliers from the given datapoints of caldulated ranges.
    It returns either the only outliers or the data without outliers, depending on returnOutliers
    """
    outlier0, lower_bound0, upper_bound0 = detect_outlier(testDataset[0])
    outlier1, lower_bound1, upper_bound1 = detect_outlier(testDataset[1])
    outlierIndex = [0] * len(testDataset)

    for j in range(2):
        if j == 0:
            lower_bound = lower_bound0
            upper_bound = upper_bound0
        else:
            lower_bound = lower_bound1
            upper_bound = upper_bound1

        for index in range(len(testDataset)):
            if testDataset[index][j] <= lower_bound * 2 or testDataset[index][j] >= upper_bound * 2:
                outlierIndex[index] = 1

    if returnOutliers:
        data = removeData(dataset=testDataset, outliers=outlierIndex)
        return data, outlierIndex
    else:
        data = removeOutliers(dataset=testDataset, outliers=outlierIndex)
        return data, outlierIndex


def printOutlierCount(outliers, title):
    """
    This function prints the amount of outliers with an outlier index
    """
    elements_count = collections.Counter(outliers)
    # printing the element and the frequency
    print()
    print('\033[1m' + title + ': ' + '\033[0m')
    for key, value in elements_count.items():

        if key == 0:
            print(f"Number of datapoints without outliers:  {value}")
        else:
            print(f"Number of outliers:                     {value}")


def removeOutliers(dataset, outliers):
    """
    This function removes the outliers (value == None) of some dataset
    """
    dataWithoutOutliers = np.where(outliers == 1, None, dataset)
    for i in range(len(dataWithoutOutliers)):
        if outliers[i] == 1:
            dataWithoutOutliers[i] = None
    return dataWithoutOutliers


def removeData(dataset, outliers):
    """
    This function removes the data (value != None) of some dataset
    """
    dataWithoutOutliers = np.where(outliers == 0, None, dataset)
    for i in range(len(dataWithoutOutliers)):
        if outliers[i] == 0:
            dataWithoutOutliers[i] = None
    return dataWithoutOutliers


def deleteOutliers(dataset, outliers):
    """
    This function deletes the data (value == None) of some dataset
    """
    dataWithoutOutliers = np.array([])
    for index in range(0, len(dataset)):
        if outliers[index] == 0:
            dataWithoutOutliers = np.append(dataWithoutOutliers, dataset[index])
    return dataWithoutOutliers
