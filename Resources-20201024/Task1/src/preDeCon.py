# 2.2 Density-based Projected-Clustering (PreDeCon)
# Source: https://blog.xa0.de/post/PreDeCon%20---%20Density-based-Projected-Subspace-Clustering/

import numpy as np

class PreDeCon:
    minpts = 3
    epsilon = 1
    lam = 1  # lamda
    delta = 0.25
    weightPunishment = 100 #punishment for the weights

    def __init__(self, minimumPoints, epsilon, lam, delta, weightPunishment):
        self.minpts = minimumPoints
        self.epsilon = epsilon
        self.lam = lam
        self.delta = delta
        self.weightPunishment = weightPunishment


    def get_neighbors(self, Data_Points, selected_point):
        neighbors = []  # init empty neighbours list

        for singlePoint in Data_Points:  # go threw every point in the Points_Array
            temp = (np.square(singlePoint - selected_point)).sum()  # calcualte distance
            if np.sqrt(temp) <= self.epsilon:  # if distance is smaller than epsilon
                neighbors.append(singlePoint)  # append to neighbors list if point is close enough
        print('For', selected_point, 'theese are all neighbors:', neighbors)
        return neighbors  # return the epsilon of the neighbors of selected_point


    def calculate_if_point_is_core_point(self, Data_Points, p_candidate):
        eps_neighborhood = []  # ε-neighborhood
        for q_neighbor in self.get_neighbors(Data_Points, p_candidate):  # go threw all the neighbors
            dist_pref = max(self.pref_weighted_distance(Data_Points, q_neighbor, p_candidate),
                            self.pref_weighted_distance(Data_Points, p_candidate, q_neighbor))  # calc distance_pref

            if dist_pref <= self.epsilon:  # fill preference weighted ε-neighborhood
                eps_neighborhood.append(dist_pref)
                print('For point at position', p_candidate, 'the neighbour', q_neighbor, 'has the prefered distance',
                      dist_pref, 'which is ≤ epsilon ')
                print()
        print('The epsilon-neighborhood of', p_candidate, 'is', eps_neighborhood)
        print('The length of the epsilon-neighborhood < than minpts ===> the point is NOT a core point!') if len(
            eps_neighborhood) < self.minpts else print(
            'The length of the epsilon-neighborhood ≥ minpts ===> the point is a core point!')
        return len(eps_neighborhood) >= self.minpts


    def pref_weighted_distance(self, Data_Points, neighbor, p_candidate):  # Preference weighted distance function
        weights = self.get_weights(Data_Points, neighbor)
        dist = 0  # init dist
        for i in range(2):
            dist += weights[i] * np.square(neighbor[i] - p_candidate[i])  # calc preference weighted distance
        dist = np.sqrt(dist)
        print('The point', p_candidate, 'has the neighbor', neighbor, 'which has the distance:', dist,
              'calculated from the weights', weights)
        print()
        return dist

    def get_weights(self, Data_points, candidate):
        x_distances = []  # distances in x direction
        y_distances = []  # distances in y direction
        for neighbor in self.get_neighbors(Data_points, candidate):
            x_distances.append(np.square(neighbor[0] - candidate[0]))  # [0] = x coordinate
            y_distances.append(np.square(neighbor[1] - candidate[1]))  # [1] = y coordinate

        x_var = sum(x_distances) / len(
            x_distances)  # sum of distances devided by the number of neighbors to get the average
        y_var = sum(y_distances) / len(
            y_distances)  # sum of distances devided by the number of neighbors to get the average

        # for each point p (x-values only), it defines its subspace preference vector with eigher 1 or K:
        if x_var > self.delta:
            x_weight = 1
        else:
            x_weight = self.weightPunishment

        # for each point p (y-values only), it defines its subspace preference vector with eigher 1 or K:
        if y_var > self.delta:
            y_weight = 1
        else:
            y_weight = self.weightPunishment
        return x_weight, y_weight  # return x and y vectors

