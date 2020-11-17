import numpy as np

class PreDeCon():
    def __init__(self, minPts=3, eps=1, delta = 0.25, lambda_ = 1, kappa = 100):
        self.minPts = minPts
        self.eps = eps
        self.delta = delta
        self.lambda_ = lambda_
        self.kappa = kappa

        self.num_points = 0
        self.num_features = 0
        self.X = None
        self._neighborhoods = None
    
    def fit(self, X):
        self.num_points = X.shape[0]
        self.num_features = X.shape[1]
        self.X = X

        # caching neighborhoods in dicts
        neighborhoods = {}
        for p in X:
            N = self._eps_neighborhood(p)
            neighborhoods[p.tobytes()] = N
        self._neighborhoods = neighborhoods

        pref_weighted_neighborhoods = {}
        for p in X:
            N_w = self._preference_weighted_eps_neighborhood(p)
            pref_weighted_neighborhoods[p.tobytes()] = N_w
        self._pref_weighted_neighborhoods = pref_weighted_neighborhoods
    
    def _neighborhood_of_point(self, p):
        # return cached neighborhood for points in X, calculate for unknown points
        try:
            return self._neighborhoods[p.tobytes()]
        except IndexError:
            return self._eps_neighborhood(p)
    
    def _pref_neighborhood_of_point(self, p):
        # return cached neighborhood for points in X, calculate for unknown points
        try:
            return self._pref_weighted_neighborhoods[p.tobytes()]
        except IndexError:
            self._preference_weighted_eps_neighborhood(p)
    
    def _variance_along_attribute(self, p, j):
        N = self._neighborhood_of_point(p)
        sum = np.sum([np.abs(p[j] - q[j]) for q in N])
        return sum / len(N)
    
    def _subspace_preference_vector(self, p):
        var = self._variance_along_attribute
        d = self.num_features
        w = [(1 if var(p, j) > self.delta else self.kappa) for j in range(d)],
        return np.array(w)
    
    def _subspace_preference_dimensionality(self, p):
        return np.count_nonzero(self._subspace_preference_vector(p) == self.kappa)
    
    def _preference_weighted_dimensionality_measure(self, p, q):
        return np.sqrt(np.sum(self._subspace_preference_vector(p) * (p - q)**2))
    
    def _general_preference_weighted_dimensionality_measure(self, p, q):
        dist = self._preference_weighted_dimensionality_measure
        return np.maximum(dist(p, q), dist(q, p))
    
    def _eps_neighborhood(self, p):
        return np.array([q for q in X if np.linalg.norm(p-q) <= self.eps])

    def _preference_weighted_eps_neighborhood(self, o):
        dist_pref = self._general_preference_weighted_dimensionality_measure
        return np.array([x for x in self.X if dist_pref(o, x) <= self.eps])
    
    def _is_core_point(self, p):
        pdim = self._subspace_preference_dimensionality(p)
        N_w = self._pref_neighborhood_of_point(p)
        return pdim <= self.lambda_ and len(N_w) >= self.minPts

if __name__ == "__main__":
    X = np.array([
        [0, 3],
        [1, 3],
        [2, 3], # p_3
        [3, 3],
        [4, 3],
        [5, 3], # p_6
        [6, 5],
        [6, 4],
        [6, 3],
        [6, 2],
        [6, 1],
        [6, 0]
    ])
    print(X.shape)

    predecon = PreDeCon()
    predecon.fit(X)

    p3 = np.array([2, 3])
    p6 = np.array([5, 3])
    p9 = np.array([6, 3])

    N_p3 = predecon._neighborhood_of_point(p3)
    print("p3:   ", p3)
    print("N(p3):", N_p3)

    var_A0 = predecon._variance_along_attribute(p3, 0)
    var_A1 = predecon._variance_along_attribute(p3, 1)
    print("VAR_A0 for p3's neighborhood:", var_A0)
    print("VAR_A1 for p3's neighborhood:", var_A1)

    print("w_p3:", predecon._subspace_preference_vector(p3))
    print("w_p6:", predecon._subspace_preference_vector(p6))

    print("PDim for p3:", predecon._subspace_preference_dimensionality(p3))
    print("PDim for p6:", predecon._subspace_preference_dimensionality(p6))

    dist = predecon._preference_weighted_dimensionality_measure
    print("dist(p6, p9) =", dist(p6, p9))
    print("dist(p9, p6) =", dist(p9, p6))

    dist_pref = predecon._general_preference_weighted_dimensionality_measure
    print("dist_pref(p6, p9) =", dist_pref(p6, p9))

    print("N_w for p3:", predecon._pref_neighborhood_of_point(p3), sep='\n')
    print("N_w for p6:", predecon._pref_neighborhood_of_point(p6), sep='\n')

    print("Is p3 a core point?", predecon._is_core_point(p3))
    print("Is p6 a core point?", predecon._is_core_point(p6))
