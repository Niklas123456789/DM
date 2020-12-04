import numpy as np
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
        for p in self.X:
            N = self._eps_neighborhood(p)
            neighborhoods[p.tobytes()] = N
        self._neighborhoods = neighborhoods

        pref_weighted_neighborhoods = {}
        for p in self.X:
            N_w = self._preference_weighted_eps_neighborhood(p)
            pref_weighted_neighborhoods[p.tobytes()] = N_w
        self._pref_weighted_neighborhoods = pref_weighted_neighborhoods

    	# see Figure 4 of the PreDeCon_Paper.pdf for the Pseudocode
        self._cluster_of_points = {}
        clusterID = 0

        for point in self.X:
            if self._is_core_point(point):
                # ensures IDs that only increase by 1
                try:
                    self._cluster_of_points[point.tobytes()]
                except KeyError:
                    clusterID += 1

                queue = Queue()

                for n in self._pref_neighborhood_of_point(point):
                    queue.put(n)

                while not queue.empty():
                    q = queue.get()
                    R = [x for x in self.X if self._is_directly_preference_weighted_reachable(q,x)]

                    for x in R:
                        try:
                            if self._cluster_of_points[x.tobytes()] == self._NOISE:
                                self._cluster_of_points[x.tobytes()] = clusterID

                        # if a KeyError occured, x was unclassified
                        except KeyError:
                            self._cluster_of_points[x.tobytes()] = clusterID
                            queue.put(x)

            else: # point is noise
                self._cluster_of_points[point.tobytes()] = self._NOISE
        
        self.labels = []
        for point in self.X:
            self.labels.append(self._cluster_of_points[point.tobytes()])
        
        # self._cluster_of_points = None

    def _neighborhood_of_point(self, p):
        """
        Convenience method, either computes the epsilon neighborhood of a data-point p if it is not already stored in the PreDeCon object or returns the stored neighborhood.

        args:
            p : numpy.ndarray
        """
        # return cached neighborhood for points in X, calculate for unknown points
        try:
            return self._neighborhoods[p.tobytes()]
        except KeyError:
            return self._eps_neighborhood(p)

    def _pref_neighborhood_of_point(self, p):
        """
        Convenience method, either computes the preference weighted epsilon neighborhood of a data-point p if it is not already stored in the PreDeCon object or returns the stored neighborhood.

        args:
            p : numpy.ndarray
        """
        # return cached neighborhood for points in X, calculate for unknown points
        try:
            return self._pref_weighted_neighborhoods[p.tobytes()]
        except KeyError:
            self._preference_weighted_eps_neighborhood(p)

    def _variance_along_attribute(self, p, j):
        """
        Computes the variance along an attribute j inside the epsilon neighborhood of a data-point p (see Definition 1 of the PreDeCon_Paper.pdf).

        args:
            p : numpy.ndarray
            j : int - Specifies the attribute (i.e. the column) whichs variance will be computed.
        """
        N = self._neighborhood_of_point(p)
        sum = np.sum([np.abs(p[j] - q[j]) for q in N])
        return sum / len(N)

    def _subspace_preference_vector(self, p):
        """
        Constructs the subspace preference vector pf a data-point p (see Definition 3 of the PreDeCon_Paper.pdf).

        args:
            p : numpy.ndarray
        """
        var = self._variance_along_attribute
        d = self.num_features
        w = [(1 if var(p,j) > self.delta else self.kappa) for j in range(d)]
        return np.array(w)

    def _subspace_preference_dimensionality(self, p):
        """
        Computes the number of dimensions with low enough variance of a data-point p (see Definition 2 of the PreDeCon_Paper.pdf).

        args:
            p : numpy.ndarray
        """
        return np.count_nonzero(self._subspace_preference_vector(p) == self.kappa)

    def _preference_weighted_similarity_measure(self, p, q):
        """
        Computes a distance between data-points p and q based on p's subspace preference vector (see Definition 3 of the PreDeCon_Paper.pdf).

        p : numpy.ndarray

        q : numpy.ndarray
        """
        return np.sqrt(np.sum(self._subspace_preference_vector(p) * (p - q)**2))

    def _general_preference_weighted_similarity_measure(self, p, q):
        """
        Determines the maximum distance between data-points p and q (see Definition 4 of the PreDeCon_Paper.pdf).

        args:
            p : numpy.ndarray
            q : numpy.ndarray
        """
        dist = self._preference_weighted_similarity_measure
        return np.maximum(dist(p,q), dist(q,p))

    def _eps_neighborhood(self, p):
        """
        Computes the epsilon neighborhood of a data-point p based on this objects eps-value.

        args:
            p : numpy.ndarray
        """
        return np.array([q for q in self.X if np.linalg.norm(p-q) <= self.eps])

    def _preference_weighted_eps_neighborhood(self, o):
        """
        Computes the preference weighted epsilon neighborhood of a data-point o based on this objects eps-value and the general preference weighted similarity measure (see Definition 5 of the PreDeCon_Paper.pdf).

        args:
            o : numpy.ndarray
        """
        dist_pref = self._general_preference_weighted_similarity_measure
        return np.array([x for x in self.X if dist_pref(o,x) <= self.eps])

    def _is_core_point(self, p):
        """
        Checks if a data-point p is a preference weighted core point (see Definition 6 of the PreDeCon_Paper.pdf).

        args:
            p : numpy.ndarray
        """
        pdim = self._subspace_preference_dimensionality(p)
        N_w = self._pref_neighborhood_of_point(p)
        return pdim <= self.lambda_ and len(N_w) >= self.minPts

    def _is_directly_preference_weighted_reachable(self, q, p):
        """
        Checks if a data-point p is directly preference weighted reachable from a data-point q (see Definition 7 of the PreDeCon_Paper.pdf).

        args:
            q : numpy.ndarray
            p : numpy.ndarray
        """
        for pt in self._pref_neighborhood_of_point(q):
            # if the point is in the neighborhood (condition 3), check if condition 1 and 2 are fulfilled
            if np.array_equal(p,pt):
                return self._is_core_point(q) \
                       and self._subspace_preference_dimensionality(p) <= self.lambda_

        # the point cannot be directly reachable since condition 3 was not fulfilled, therefore return False
        return False

    def _is_noise_point(self, p):
        """
        Checks if a data-point p of is a noise point.

        args:
            p : numpy.ndarray

        raises:
            KeyError : if p is not a data point of the fitted data-points X
        """
        return self._cluster_of_points[p.tobytes()] == self._NOISE
