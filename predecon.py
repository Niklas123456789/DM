import numpy as np
from util.timing import timed
from collections import defaultdict

class PreDeCon():
    def __init__(self, minPts=3, eps=1.0, delta = 0.25, lambda_ = 1, kappa = 100):
        """
        args:
            minPts : int - The minimum number of points required of a data-point p's epsilon neighborhood so that p is a core point.

            eps : float - The maximum distance between a data-point p and any other data-point q in p's epsilon neighborhood.

            delta : float - Threshold for the variance of an attribute inside an epsilon neighborhood

            lambda_ : int - The maximum value of the subspace preference dimensionality of an epsilon neighborhood
                            of a data-point p so that p can still be a preference weighted core point.

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

        self._compute_neighborhoods()
        self._compute_subspace_preference_matrix()
        self._compute_similarity_matrix()
        self._compute_weighted_neighborhoods()
        self._compute_core_points()
        self._compute_reachability()
        self._compute_clusters()

    @timed('_performance', 'cn')
    def _compute_neighborhoods(self):
        """
        Computes the epsilon neighborhood for all data-points p in self.X
        """

        self._neighborhoods = {}
        for p in range(self.num_points):
            # Computes an index list for the epsilon neighborhood of a data-point self.X[p] based on this objects eps-value
            # where every entry corresponds to another data-point (i.e. a row in self.X).

            # e.g. if for a data-point p the entry self._neighborhoods[p] is [1,2,5],
            # the epsilon neighborhood consists of self.X[1], self.X[2], self.X[5]
            self._neighborhoods[p] = np.flatnonzero(np.linalg.norm(self.X - self.X[p], axis=1, ord=2) <= self.eps)
    
    @timed('_performance', 'cwn')
    def _compute_weighted_neighborhoods(self):
        """
        Computes the preference weighted epsilon neighborhood for all data-points p in self.X
        """

        self._pref_weighted_neighborhoods = {}
        for p in range(self.num_points):
            # Computes an index list for the preference weighted epsilon neighborhood of a data-point self.X[o] based on this objects eps-value
            # and the general preference weighted similarity measure (see Definition 5 of the PreDeCon_Paper.pdf)
            # where every entry corresponds to another data-point (i.e. a row in self.X).

            # e.g. if for a data-point p the entry self._pref_weighted_neighborhoods[p] is [1,2,5],
            # the prefernce weighted epsilon neighborhood consists of self.X[1], self.X[2], self.X[5]
            self._pref_weighted_neighborhoods[p] = np.flatnonzero(self._similarity[p] <= self.eps)
    
    @timed('_performance', 'cspm')
    def _compute_subspace_preference_matrix(self):
        """
        Constructs the subspace preference matrix where row i corresponds to the subspace preference
        vector of data-point self.X[i] (see Definition 3 of the PreDeCon_Paper.pdf) and a vector of subspace preference
        dimensionalities where entry i is the subspace preference dimensionality of self.X[i] (i.e. the number of penalized row-elements
        in the subspace preference matrix, see Definition 2 of the PreDeCon_Paper.pdf).
        """

        # variances where the values in row i correspond to the variances of the attributes 0,...,j
        # of data-point self.X[i] (see Definition 1 of the PreDeCon_Paper.pdf).

        # Initially assume a large variance.
        variance_matrix = np.full(self.X.shape, fill_value=np.inf)

        for i in range(self.num_points):
            # https://numpy.org/doc/stable/user/theory.broadcasting.html#example-3
            if len(self._neighborhoods[i]) > 0:
                variance_matrix[i] = np.sum((self.X[self._neighborhoods[i]] - self.X[i])**2,axis=0) / len(self._neighborhoods[i])
        
        self._subspace_preference_matrix = np.ones(self.X.shape)

        # where the variance is smaller or equal to delta, set the preference to kappa
        # https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slicing#boolean-or-mask-index-arrays
        self._subspace_preference_matrix[variance_matrix <= self.delta] = self.kappa
        self._subspace_preference_dimensionality = np.count_nonzero(self._subspace_preference_matrix == self.kappa, axis=1)

    @timed('_performance', 'csm')
    def _compute_similarity_matrix(self):
        """
        Computes a symmetric matrix where row i corresponds to the maximum distance between self.X[i]
        and every other data-point in self.X (see Definition 4 of the PreDeCon_Paper.pdf).

        e.g. if the maximum distance between self.X[p] and self.X[q] is 7, then
        self._similarity[p,q] == self._similarity[q,p] == 7
        """

        similarity = np.zeros((self.num_points, self.num_points))
        for p in range(self.num_points):
            w = self._subspace_preference_matrix[p]
            similarity[p] = np.sqrt(np.sum(w * (self.X - self.X[p])**2, axis=1))

        self._similarity = np.maximum(similarity, similarity.T)

    @timed('_performance', 'ccp')
    def _compute_core_points(self):
        """
        Computes the preference weighted core points of self.X (see Definition 6 of the PreDeCon_Paper.pdf).
        """

        pdim = self._subspace_preference_dimensionality
        num_N_w = np.array([len(self._pref_weighted_neighborhoods[x]) for x in range(self.num_points)])
        self._core_points = np.logical_and(pdim <= self.lambda_, num_N_w >= self.minPts)

    @timed('_performance', 'cr')
    def _compute_reachability(self):
        """
        Computes all directly preference weighted reachable data-points for every data-point in self.X
        (see Definition 7 of the PreDeCon_Paper.pdf).
        """

        reachable = {}
        for q in range(self.num_points):
            cond1 = self._core_points[q]
            cond2 = self._subspace_preference_dimensionality <= self.lambda_
            cond3 = np.full((self.num_points,), False)
            cond3[self._pref_weighted_neighborhoods[q]] = True

            reachable[q] = np.flatnonzero(np.logical_and(cond1, np.logical_and(cond2, cond3)))
        self._directly_reachable_points = reachable
    
    @timed('_performance', 'cc')
    def _compute_clusters(self):
        """
        Computes the clustering for self.X, see Figure 4 of the PreDeCon_Paper.pdf for the Pseudocode.
        """

        # set all labels to noise by default
        clusters = [self._NOISE] * self.num_points
        clusterID = 0

        for i in range(self.num_points):
            # only start a cluster if the current point is a core point which
            # is not yet assigned to a cluster
            if self._core_points[i] and clusters[i] == self._NOISE:
                clusterID += 1

                connected = set(self._pref_weighted_neighborhoods[i])
                
                while connected:
                    q = connected.pop()
                    R = self._directly_reachable_points[q]

                    # keep only unclassified points
                    R = [x for x in R if clusters[x] == self._NOISE]

                    for x in R:
                        clusters[x] = clusterID

                        # only core points may connect the current point (i.e. point i) with
                        # points outside of i's epsilon-neighborhood (and therefore add these 
                        # points to the cluster)
                        if self._core_points[x]:
                            connected.add(x)

        self.labels = clusters
    
    def performance(self):
        """Returns performance statistics for selected instance methods."""

        perf = ""
        for key, value in self._performance.items():
            perf += f"{value / 1000_000_000:>8.4f}s {key}\n"
        return perf
