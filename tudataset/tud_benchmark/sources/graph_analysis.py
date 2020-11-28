import networkx as nx
from itertools import combinations

def getAverageNumberOfEdges(G):
    number_of_graphs = len(G)
    number_of_edges = sum([nx.number_of_edges(graph) for graph in G])
    return number_of_edges / number_of_graphs


def getAverageNumberOfNodes(G):
    number_of_graphs = len(G)
    number_of_nodes = sum([nx.number_of_nodes(graph) for graph in G])
    return number_of_nodes/number_of_graphs



def getAverageNumberOfEdgesPerNode(G):
    return getAverageNumberOfEdges(G)/getAverageNumberOfNodes(G)


def getDensity(G):
    number_of_graphs = len(G)
    sum_density = sum([nx.density(graph) for graph in G])
    return sum_density / number_of_graphs



def getNumberOfIsomorphicGraphs(G):
    return len([pair for pair in combinations(G,2) if nx.is_isomorphic(pair[0],pair[1])])


def getGraphDataByClass(graphs,movie_topic):
    avg_number_edges = getAverageNumberOfEdges(graphs)
    avg_number_nodes = getAverageNumberOfNodes(graphs)
    avg_density = getDensity(graphs)
    avg_number_edges_per_node = getAverageNumberOfEdgesPerNode(graphs)
    print(movie_topic+': ')
    print('Average number of edges: ',avg_number_edges)
    print('Average number of nodes: ',avg_number_nodes)
    print('Average number of edges per nodes: ',avg_number_edges_per_node)
    print('Average density: ',avg_density)
    print('Number of isomorphic pairs: ',getNumberOfIsomorphicGraphs(graphs))


