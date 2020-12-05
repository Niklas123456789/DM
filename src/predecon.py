import numpy as np
from timing import timed
from collections import defaultdict
from queue import Queue

class PreDeCon():
    def __init__(self, minPts=3, eps=1.0, delta = 0.25, lambda_ = 1, kappa = 100):
        """
        args:
            minPts : int - The minimum number of points required of a data-point p's epsilon neighborhood so that p is a core point.
            eps : float - The maximum distance between a data-point p and any other data-point q in p's epsilon neighborhood.
            delta : float - Threshold for the variance of an attribute inside an epsilon neighborhood
            lambda_ : int - The maximum value of the subspace preference dimensionality of an epsilon neighborhood of a data-point p so that p can still be a preference weighted core point.
            kappa : int - The factor which weights low variances of an attribute.
        """

        self.minPts = minPts
        self.eps = eps
        self.delta = delta
        self.lambda_ = lambda_
        self.kappa = kappa

        self.num_points = 0
        self.num_features = 0
        self.X = None
        self._neighborhoods = None
        self._pref_weighted_neighborhoods = None
        self._cluster_of_points = None

        self._NOISE = -1 # cluster ID for all noise points

        self._performance = defaultdict(int)

    @timed('_performance', 'fit')
    def fit(self, X):
        """
        Apply the PreDeCon algorithm on X.

        args:
            X : numpy.ndarray - The to-be clustered data-points.
        """
        self.num_points = X.shape[0]
        self.num_features = X.shape[1]
        self.X = X

        # caching neighborhoods in dicts
        neighborhoods = {}
        for p in range(self.num_points):
            N = self._eps_neighborhood(p)
            neighborhoods[p] = N
        self._neighborhoods = neighborhoods

        self._subspace_preference_matrix()

        pref_weighted_neighborhoods = {}
        for p in range(self.num_points):
            N_w = self._preference_weighted_eps_neighborhood(p)
            pref_weighted_neighborhoods[p] = N_w
        self._pref_weighted_neighborhoods = pref_weighted_neighborhoods

    	# see Figure 4 of the PreDeCon_Paper.pdf for the Pseudocode
        self._cluster_of_points = {}
        clusterID = 0

        for i in range(self.num_points):
            if self._is_core_point(i):
                # ensures IDs that only increase by 1
                try:
                    self._cluster_of_points[i]
                except KeyError:
                    clusterID += 1

                queue = Queue()

                for n in self._pref_weighted_neighborhoods[i]:
                    queue.put(n)

                while not queue.empty():
                    q = queue.get()
                    R = [x for x in range(self.num_points) if self._is_directly_preference_weighted_reachable(q,x)]

                    for x in R:
                        try:
                            if self._cluster_of_points[x] == self._NOISE:
                                self._cluster_of_points[x] = clusterID

                        # if a KeyError occured, x was unclassified
                        except KeyError:
                            self._cluster_of_points[x] = clusterID
                            queue.put(x)

            else: # point is noise
                self._cluster_of_points[i] = self._NOISE
        
        self.labels = []
        for i in range(self.num_points):
            self.labels.append(self._cluster_of_points[i])
        
        self._cluster_of_points = None

    @timed('_performance', 'vm')
    def _variance_matrix(self):
        """
        Computes the variances where the values in row i correspond to the variances of the attributes 0,...,j of data-point self.X[i] (see Definition 1 of the PreDeCon_Paper.pdf).
        """
        vars = np.zeros(self.X.shape)
        for i in range(self.num_points):
            # https://numpy.org/doc/stable/user/theory.broadcasting.html#example-3
            vars[i] = np.sum(np.abs(self.X[self._neighborhoods[i]] - self.X[i]),axis=0) / len(self.X[self._neighborhoods[i]])
        return vars

    @timed('_performance', 'spm')
    def _subspace_preference_matrix(self):
        """
        Constructs the subspace preference matrix where row i corresponds to the subspace preference vector of data-point self.X[i] (see Definition 3 of the PreDeCon_Paper.pdf).
        """
        vars = self._variance_matrix()
        self._subspace_preference_matrix = np.ones(self.X.shape)
        # where the variance is smaller or equal to delta, set the preference to kappa
        # https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slicing#boolean-or-mask-index-arrays
        self._subspace_preference_matrix[vars <= self.delta] = self.kappa

    @timed('_performance', 'spd')
    def _subspace_preference_dimensionality(self, p):
        """
        Computes the number of dimensions with low enough variance of a data-point self.X[p] (see Definition 2 of the PreDeCon_Paper.pdf).

        args:
            p : int
        """
        return np.count_nonzero(self._subspace_preference_matrix[p] == self.kappa)

    @timed('_performance', 'pwsm')
    def _preference_weighted_similarity_measure(self, p, q):
        """
        Computes a distance between data-points self.X[p] and self.X[q] based on self.X[p]'s subspace preference vector (see Definition 3 of the PreDeCon_Paper.pdf).

        args:
            p : int
            q : int
        """
        return np.sqrt(np.sum(self._subspace_preference_matrix[p] * (self.X[p]-self.X[q])**2))

    @timed('_performance', 'gpwsm')
    def _general_preference_weighted_similarity_measure(self, p, q):
        """
        Determines the maximum distance between data-points self.X[p] and self.X[q] (see Definition 4 of the PreDeCon_Paper.pdf).

        args:
            p : int
            q : int
        """
        dist = self._preference_weighted_similarity_measure
        return np.maximum(dist(p,q), dist(q,p))

    @timed('_performance', 'en')
    def _eps_neighborhood(self, p):
        """
        Computes an index list for the epsilon neighborhood of a data-point self.X[p] based on this objects eps-value where every entry corresponds to another data-point (i.e. a row in self.X).

        e.g. for a returned list [1,2,5], the epsilon neighborhood consists of self.X[1], self.X[2], self.X[5]

        args:
            p : int
        """
        return np.array([q for q in range(self.num_points) if np.linalg.norm(self.X[p]-self.X[q]) <= self.eps])

    @timed('_performance', 'pwen')
    def _preference_weighted_eps_neighborhood(self, o):
        """
        Computes an index list for the preference weighted epsilon neighborhood of a data-point self.X[o] based on this objects eps-value and the general preference weighted similarity measure (see Definition 5 of the PreDeCon_Paper.pdf) where every entry corresponds to another data-point (i.e. a row in self.X).

        e.g. for a returned list [1,2,5], the prefernce weighted epsilon neighborhood consists of self.X[1], self.X[2], self.X[5]

        args:
            o : int
        """
        dist_pref = self._general_preference_weighted_similarity_measure
        return np.array([x for x in range(self.num_points) if dist_pref(o,x) <= self.eps])

    @timed('_performance', 'icp')
    def _is_core_point(self, p):
        """
        Checks if a data-point self.X[p] is a preference weighted core point (see Definition 6 of the PreDeCon_Paper.pdf).

        args:
            p : int
        """
        pdim = self._subspace_preference_dimensionality(p)
        N_w = self._pref_weighted_neighborhoods[p]
        return pdim <= self.lambda_ and len(N_w) >= self.minPts

    @timed('_performance', 'idpwr')
    def _is_directly_preference_weighted_reachable(self, q, p):
        """
        Checks if a data-point self.X[p] is directly preference weighted reachable from a data-point self.X[q] (see Definition 7 of the PreDeCon_Paper.pdf).

        args:
            q : int
            p : int
        """
        # this order of condition checking was the fastest
        return p in self._pref_weighted_neighborhoods[q] \
                and self._is_core_point(q) \
                and self._subspace_preference_dimensionality(p) <= self.lambda_
    
    def performance(self):
        """Returns performance statistics for selected instance methods."""
        perf = ""
        for key, value in self._performance.items():
            perf += f"{value / 1000_000_000:>8.4f}s {key}\n"
        return perf
