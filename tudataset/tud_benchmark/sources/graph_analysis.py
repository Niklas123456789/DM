import networkx as nx


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