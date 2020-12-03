from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
import collections



def local_outlier_factor(data):
    print('test')
    clf = LocalOutlierFactor(n_neighbors=2)
    labels = clf.fit_predict(data)
    return labels


def elliptic_envelope(data):
    cov = EllipticEnvelope(random_state=0).fit_predict(data)
    return cov


def isolation_forest(data, contamination):
    forest = IsolationForest(contamination=contamination)
    labels = forest.fit_predict(data)
    return labels


def one_class_svm(data, nu):
    labels = OneClassSVM(nu=nu).fit_predict(data)
    return labels


def pandas_outlier_detection(data):
    df = pd.DataFrame(data)

    from scipy import stats
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df.values


def db_scan_outlier_detection():
    print('test')


def detect_outlier(data):
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
    outlier0, lower_bound0, upper_bound0 = detect_outlier(testDataset[0])
    outlier1, lower_bound1, upper_bound1 = detect_outlier(testDataset[1])
    outlierIndex = [0] * len(testDataset)
    # print(testDataset[0])
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

def printOutlierCount(outliers):
    elements_count = collections.Counter(outliers)
    # printing the element and the frequency
    for key, value in elements_count.items():
        print()
        if key == 0:
            print(f"Number of datapoints without outliers:  {value}")
        else:
            print(f"Number of outliers:                     {value}")


def removeOutliers(dataset, outliers):
    # data without outliers
    testDataWitoutOuliers = np.where(outliers == 1, None, dataset)
    for i in range(len(testDataWitoutOuliers)):
        if outliers[i] == 1:
            testDataWitoutOuliers[i] = None
    return testDataWitoutOuliers


def removeData(dataset, outliers):
    # data only outliers
    testDataOnlyOuliers = np.where(outliers == 0, None, dataset)
    for i in range(len(testDataOnlyOuliers)):
        if outliers[i] == 0:
            testDataOnlyOuliers[i] = None
    return testDataOnlyOuliers

