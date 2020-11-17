import numpy as np

minPts = 3
eps = 1
delta = 0.25
lambda_ = 1
kappa = 100

coords = np.array([
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

# TODO: delete, can be inferred from coords
neighborhood = [[coords[n] for n in neighbors] for neighbors in np.array([
    [0, 1],
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 8],
    [6, 7],
    [6, 7, 8],
    [5, 7, 8, 9],
    [8, 9, 10],
    [9, 10, 11],
    [10, 11]
])]

p = list(zip(coords, neighborhood))

p3, N_p3 = p[2]

print("p3:   ", p3)
print("N(p3):", N_p3)

def var(j, p):
    """variance along an attribute"""
    pi, N = p
    sum = 0
    for pi_q in N:
        sum += np.abs(pi[j] - pi_q[j])
    return sum / len(N)

print("VAR_A0 for p3's neighborhood:", var(0, p[2]))
print("VAR_A1 for p3's neighborhood:", var(1, p[2]))

def w(p):
    """subspace preference vector"""
    return np.array([
        1 if var(0, p) > delta else kappa,
        1 if var(1, p) > delta else kappa
    ])

print("w_p3:", w(p[2]))
print("w_p6:", w(p[5]))

def pdim(p):
    """subspace preference dimensionality"""
    return np.count_nonzero(w(p) == kappa)

print("PDim for p3:", pdim(p[2]))
print("PDim for p6:", pdim(p[5]))

def dist(p, q):
    """preference weighted similarity measure"""
    pi_p, N_p = p
    pi_q, N_q = q
    return np.sqrt(np.sum(w(p) * (pi_p - pi_q)**2))

print("dist(p6, p9) =", dist(p[5], p[8]))
print("dist(p9, p6) =", dist(p[8], p[5]))

def dist_pref(p, q):
    """general preference weighted similarity"""
    return np.maximum(dist(p, q), dist(q, p))

print("dist_pref(p6, p9) =", dist_pref(p[5], p[8]))

def N_w(o):
    """preference weighted ε-neighborhood"""
    return [x[0] for x in p if dist_pref(o, x) <= eps]

print("N_w for p3:", N_w(p[2]))
print("N_w for p6:", N_w(p[5]))

def core(p):
    return pdim(p) <= lambda_ and len(N_w(p)) >= minPts

print("Is p3 a core point?", core(p[2]))
print("Is p6 a core point?", core(p[5]))
