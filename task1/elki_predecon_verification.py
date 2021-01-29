# Adapted Script for running an elki algorithm and save its results for the PreDeCon algorithm
import os
import re
from subprocess import Popen, PIPE
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI

# Install Java and download the elki bundle https://elki-project.github.io/releases/release0.7.5/elki-bundle-0.7.5.jar
ELKI_JAR = "elki-bundle-0.7.5.jar"


def elki_predecon(data_file_name, X, minPts=3, eps=1.0, delta = 0.25, lambda_ = 1, kappa = 100):
    """Perform PreDeCon clustering implemented by ELKI package.
       The function calls jar package, which must be accessible through the
       path stated in ELKI_JAR constant.

        args:
            data_file_name : string - file name of the dataset X

            X : array of shape (n_samples, n_features) - A feature array.
                
            minPts : int - The minimum number of points required of a data-point p's epsilon neighborhood so that p is a core point.

            eps : float - The maximum distance between a data-point p and any other data-point q in p's epsilon neighborhood.

            delta : float - Threshold for the variance of an attribute inside an epsilon neighborhood

            lambda_ : int - The maximum value of the subspace preference dimensionality of an epsilon neighborhood
                            of a data-point p so that p can still be a preference weighted core point.

            kappa : int - The factor which weights low variances of an attribute.

        Returns
        -------
        labels : array [n_samples]
            Cluster labels for each point.
    """
    print(data_file_name)
    np.savetxt(data_file_name, X, delimiter=",", fmt="%.6f")
    print("Run elki")
    # run elki with java
    # You can find the read of the names of the parameters for the PreDeCon algorithm from the elki GUI
    process = Popen(["java", "-cp", ELKI_JAR, "de.lmu.ifi.dbs.elki.application.KDDCLIApplication",
                     "-algorithm", "clustering.subspace.PreDeCon",
                     "-dbc.in", data_file_name,
                     "-dbscan.epsilon", str(eps),
                     "-dbscan.minpts", str(minPts),
                     "-predecon.delta", str(delta),
                     "-predecon.kappa", str(kappa),
                     "-predecon.lambda", str(lambda_)],
                    stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        raise IOError("Elki implementation failed to execute: \n {}".format(output.decode("utf-8")))

    # remove data file
    os.remove(data_file_name)

    # parse output
    elki_output = output.decode("utf-8")
    # initialize array of ids and labels
    Y_pred = np.array([]).reshape(0, 2)
    # for each cluster, split by regex from output
    for i, cluster in enumerate(elki_output.split("Cluster: Cluster")):
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

def nmi_comparison(dataset, elki_labels, own_labels, true_labels, save_in_txtfile=False):
    '''
    Compares the elki labels with the labels of our own implementation.
    '''
    elki_nmi = NMI(elki_labels,true_labels)
    own_nmi  = NMI(own_labels,true_labels)

    print()    
    print(f"ELKI-NMI: {elki_nmi}")
    print(f"Own-NMI: {own_nmi}")
    print('ELKI-NMI == Own-NMI:', elki_nmi == own_nmi)

    if save_in_txtfile:
        with open('NMIs.txt','a') as f:
            f.write(f'{dataset},\n  elki_nmi == own_nmi: {elki_nmi == own_nmi}\n')

    # preds_and_true_labels = np.concatenate([elki_labels[:,None],own_labels[:,None]],true_labels[:,None]],axis=1)
    # np.savetxt("results.csv", preds_and_true_labels', delimiter=",", fmt="%.1f")
