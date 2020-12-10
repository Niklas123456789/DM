# Script for running an elki algorithm and save its results, here with an example for the COPAC algorithm
import os
import re
from subprocess import Popen, PIPE
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score

DATA_FILE_NAME = "data.tsv"
# Install Java and download the elki bundle https://elki-project.github.io/releases/release0.7.5/elki-bundle-0.7.5.jar
ELKI_JAR = "elki-bundle-0.7.5.jar"


def elki_copac(X, k=10, mu=5, eps=0.5, alpha=0.85):
    """Perform COPAC clustering implemented by ELKI package.
       The function calls jar package, which must be accessible through the
       path stated in ELKI_JAR constant.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            A feature array.
        k : int, optional, default=10
            Size of local neighborhood for local correlation dimensionality.
            The paper suggests k >= 3 * n_features.
        mu : int, optional, default=5
            Minimum number of points in a cluster with mu <= k.
        eps : float, optional, default=0.5
            Neighborhood predicate, so that neighbors are closer than `eps`.
        alpha : float in ]0,1[, optional, default=0.85
            Threshold of how much variance needs to be explained by Eigenvalues.
            Assumed to be robust in range 0.8 <= alpha <= 0.9

        Returns
        -------
        labels : array [n_samples]
            Cluster labels for each point.
    """
    # write data into tsv file
    np.savetxt(DATA_FILE_NAME, X, delimiter=",", fmt="%.6f")
    print("Run elki")
    # run elki with java
    # You can find the read of the names of the parameters for the COPAC algorithm from the elki GUI
    process = Popen(["java", "-cp", ELKI_JAR, "de.lmu.ifi.dbs.elki.application.KDDCLIApplication",
                     "-algorithm", "clustering.correlation.COPAC",
                     "-dbc.in", "data.tsv",
                     "-parser.colsep", ",",
                     "-copac.knn", str(k),
                     "-dbscan.epsilon", str(eps),
                     "-dbscan.minpts", str(mu),
                     "-pca.filter.alpha", str(alpha)],
                    stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        raise IOError("Elki implementation failed to execute: \n {}".format(output.decode("utf-8")))

    # remove data file
    os.remove(DATA_FILE_NAME)

    # parse output
    elki_output = output.decode("utf-8")
    # initialize array of ids and labels
    Y_pred = np.array([]).reshape(0, 2)
    # for each cluster, split by regex from output
    for i, cluster in enumerate(elki_output.split("Cluster: Cluster")[1:]):
        # find point coordinates in output
        IDs_list = re.findall(r"ID=(\d+)", cluster)
        # create a numpy array
        IDs = np.array(IDs_list, dtype="i").reshape(-1, 1)
        # append label
        IDs_and_labels = np.hstack((IDs, np.repeat(i, len(IDs_list)).reshape(-1, 1)))
        # append to matrix
        Y_pred = np.vstack((Y_pred, IDs_and_labels))
    # sort by ID, so that the points correspond to the original X matrix
    Y_pred = Y_pred[Y_pred[:, 0].argsort()]
    # remove ID
    return Y_pred[:, 1]

if __name__ == "__main__":
    x, y = make_blobs(n_samples=200,
                      n_features=2,
                      centers=2,
                      cluster_std=0.1,
                      )
    pred = elki_copac(x)

    nmi = normalized_mutual_info_score(pred, y)
    print(f"NMI: {nmi:.4f}")
    pred_and_y = np.concatenate([pred[:,None],y[:,None]],axis=1)
    np.savetxt("results.csv", pred_and_y, delimiter=",", fmt="%.1f")
