import time
from functools import wraps

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as NMI
from predecon import PreDeCon
from IPython.display import Image
import os

'''
Used to measure the time of every function call in the PreDeCon object.
'''
def timed(attribute, key):
    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            t0 = time.perf_counter_ns()
            result = function(self, *args, **kwargs)
            getattr(self, attribute)[key] += time.perf_counter_ns() - t0 
            return result
        return wrapper
    return decorator

#################################

base_path_testing = os.path.join('..','algorithm_verification','datasets')
base_path_imdb = os.path.join('..','graph_representations','without_labels')

'''
Used to visualise the clusterings and show some statistics of the clustering.
'''
def algo_test(dataset, labelcolumn, minPts, eps, delta, lambda_, kappa, show_performance=True, show_elki=True):
    X = np.loadtxt(os.path.join(base_path_testing, dataset, dataset + '.csv'), delimiter =' ')
    X, true_labels = X[:,:labelcolumn], X[:,labelcolumn]

    predecon = PreDeCon(minPts=minPts, eps=eps, delta=delta, lambda_=lambda_, kappa=kappa)
    predecon.fit(X)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].scatter(X[:, 0], X[:, 1], c=true_labels)
    ax[0].set_title("True labels")

    ax[1].scatter(X[:,0], X[:,1], c=predecon.labels)
    ax[1].set_title("PreDeCon labels")


    if show_performance:
        print("\nDifferent ClusterIDs:", len(set(predecon.labels)))
        print("\nNMI:", NMI(true_labels,predecon.labels))
        print("\nPerformance:")
        print(predecon.performance())

    # display results of elki
    if show_elki:
        eps_str = str(eps).replace('.','_')
        delta_str = str(delta).replace('.','_')
        print()
        display(Image(os.path.join(base_path_testing, dataset,f'1__eps{eps_str}__minPts{minPts}__delta{delta_str}__kappa{kappa}__lambda{lambda_}.png')))
        display(Image(os.path.join(base_path_testing, dataset,f'1__eps{eps_str}__minPts{minPts}__delta{delta_str}__kappa{kappa}__lambda{lambda_}__legend.png')))
        