import networkx as nx
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt


def getAverageNumberOfEdges(G):
    '''
    This function returns the average number of edges of a network with several ego-graphs.
    '''
    number_of_graphs = len(G)
    number_of_edges = sum([nx.number_of_edges(graph) for graph in G])
    return number_of_edges / number_of_graphs


def getAverageNumberOfNodes(G):
    '''
    This function returns the average number of nodes of a network with several ego-graphs.
    '''
    number_of_graphs = len(G)
    number_of_nodes = sum([nx.number_of_nodes(graph) for graph in G])
    return number_of_nodes / number_of_graphs


def getAverageNumberOfEdgesPerNode(G):
    '''
    This function returns the average number of edges per node of a network with several ego-graphs.
    '''
    return getAverageNumberOfEdges(G) / getAverageNumberOfNodes(G)


def getDensity(G):
    '''
    This function returns the average number of edges of a network with several ego-graphs.
    '''
    number_of_graphs = len(G)
    sum_density = sum([nx.density(graph) for graph in G])
    return sum_density / number_of_graphs


def getNumberOfIsomorphicGraphs(G):
    '''
    This function returns the number of isomorphic pairs of a network with several ego-graphs.
    '''
    return len([pair for pair in combinations(G, 2) if nx.faster_could_be_isomorphic(pair[0], pair[1])])


def getExtremeGraphs(network):
    '''
    This function analyzes the given network, to find extreme ego-graphs (Minimum and Maximum in several
    categories) inside the network. Then it calls the visualizeWithTitle function to plot each result.
    '''
    allEdges = [nx.number_of_edges(graph) for graph in network]
    allEdges = np.reshape(allEdges, (1000, 1))
    index = np.argmin(allEdges)
    visualizeWithTitle("Ego-Graph with the lowest number of edges", network[index])
    index = np.argmax(allEdges)
    visualizeWithTitle("Ego-Graph with the highest number of edges", network[index])

    allNodes = [nx.number_of_nodes(graph) for graph in network]
    allNodes = np.reshape(allNodes, (1000, 1))
    index = np.argmin(allNodes)
    visualizeWithTitle("Ego-Graph with the lowest number of nodes", network[index])
    index = np.argmax(allNodes)
    visualizeWithTitle("Ego-Graph with the highest number of nodes", network[index])

    allDensities = [nx.density(graph) for graph in network]
    allDensities = np.reshape(allDensities, (1000, 1))
    index = np.argmin(allDensities)
    visualizeWithTitle("Ego-Graph with the lowest density", network[index])
    index = np.argmax(allDensities)
    visualizeWithTitle("Ego-Graph with the highest density", network[index])


def visualizeWithTitle(title, G, color=None, figsize=(5, 5)):
    '''
    This function draws/visualizes a graph with a title.
    '''
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G,
                     pos=nx.spring_layout(G, seed=42),
                     with_labels=True,
                     node_color=color,
                     cmap="Set2")
    plt.title(title)
    plt.show();


def getGraphDataByClass(graphs, movie_topic):
    '''
    This function prints several analytical results of a network.
    '''
    avg_number_edges = getAverageNumberOfEdges(graphs)
    avg_number_nodes = getAverageNumberOfNodes(graphs)
    avg_density = getDensity(graphs)
    avg_number_edges_per_node = getAverageNumberOfEdgesPerNode(graphs)
    print('\033[1m' + movie_topic + ': ' + '\033[0m')
    print('Average number of edges: ', avg_number_edges)
    print('Average number of nodes: ', avg_number_nodes)
    print('Average number of edges per nodes: ', avg_number_edges_per_node)
    print('Average density: ', avg_density)
    print('Number of isomorphic pairs: ', getNumberOfIsomorphicGraphs(graphs))
