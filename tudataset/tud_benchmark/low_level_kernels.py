import numpy as np

import networkx as nx
from networkx.algorithms import approximation as nx_approximation
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
  
def normalize(prop_list,max_val,min_val):
  return [(x - min_val) / (max_val - min_val) for x in prop_list]


def community_modularity_kernel(G):
  ''' 
  Uses community structure inside a graph to create the kernel.
  
  args:
    G : list of networkx.Graph
  '''
  mods = [modularity(g,greedy_modularity_communities(g)) for g in G]

  mat = np.zeros((1000,1000))

  for i in range(1000):
    for j in range(1000):
      mat[i,j] = np.minimum(mods[i],mods[j]) / np.maximum(mods[i],mods[j])

  return np.nan_to_num(mat)

def density_kernel(G):
  '''
  Uses the densities of the graphs for a simple kernel.
  
  args:
    G : list of networkx.Graph
  '''
  dens = [2*g.number_of_edges()/(g.number_of_nodes() * (g.number_of_nodes()-1)) for g in G]
  return np.array([dens]).T.dot(np.array([dens]))

def eigencentrality_kernel(G,topN=5):
  '''
  Uses the eigencentrality (a measure of influence of a node in a network) to compute a kernel.

  args:
    G : list of networkx.Graph
  '''
  eigencents = []
  mat = np.zeros((1000,1000))
  
  for g in G:
    vals = nx.eigenvector_centrality(g)
    eigencents.append(np.array([sorted(vals.values(),reverse=True)[:topN]]))
  
  for i in range(1000):
    for j in range(i,1000):
      mat[i,j] = eigencents[i].dot(eigencents[j].T)
  
  return np.maximum(mat, mat.T)

def node_edge_kernel(G):
  '''
  Uses nodes and edges of a graph to compute a kernel.
  
  args:
    G : list of networkx.Graph
  '''
  vals = [g.number_of_edges()/g.number_of_nodes() for g in G]
  vals = normalize(vals,max_val=max(vals),min_val=min(vals))
  mat = np.zeros((1000,1000))

  for i in range(1000):
    for j in range(i,1000):
      mat[i,j] = vals[i]*vals[j]
  
  return np.maximum(mat, mat.T)

def treewidth_kernel(G):
  '''
  Uses the treewidth of a graph to compute a kernel.
  
  args:
    G : list of networkx.Graph
  '''
  widths = [nx_approximation.treewidth.treewidth_min_degree(g)[0] for g in G]
  widths = normalize(widths,max_val=max(widths),min_val=min(widths))
  mat = np.zeros((1000,1000))

  for i in range(1000):
    for j in range(i,1000):
      mat[i,j] = widths[i]*widths[j]

  return np.maximum(mat, mat.T)

def isomorphism_kernel_old(G):
  '''
  Returns the number of isomorphic pairs of a network with several ego-graphs.

  args:
    G : list of networkx.Graph
  '''
  iso = np.diag(np.full(len(G),1)) # graphs are always isomorphic to themselves

  for i in range(1, 1000):
    for j in range(i+1,1000):
      iso[i,j] = np.where(nx.faster_could_be_isomorphic(G[i], G[j]),1,0)
  
  return np.maximum(iso, iso.T)

# normalization might not improve since minimum is 0
def isomorphism_kernel(G, normalize=True):
  '''
  Returns the number of isomorphic pairs of a network with several ego-graphs.
  
  args:
    G : list of networkx.Graph
  '''
  iso = np.diag(np.full(len(G),1)) # graphs are always isomorphic to themselves

  for i in range(1, 1000):
    for j in range(i+1,1000):
      iso[i,j] = np.where(nx.faster_could_be_isomorphic(G[i], G[j]),1,0)

  iso = np.maximum(iso, iso.T)

  if normalize:
    for i in range(1000):
      iso_to_i = sum(iso[i])
      iso[i][iso[i] == 1] = iso_to_i
    iso = iso / np.amax(iso)

  return iso
