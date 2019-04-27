import math

class KNeighbors:
    def __init__(self, n_neighbors):
        self.k_value = n_neighbors
        
    def find_neighbors(self, curr_index, X):
        index = 0
        neighbors_vector = []
        for inp in X:
            if(index != curr_index):
                dist = []
                dist.append(self.find_distance(x_1=X[curr_index], x_2=inp))
                dist.append(index)
                neighbors_vector.append(dist)
            index += 1
        
        neighbors_vector = sorted(neighbors_vector, key=lambda a_entry: a_entry[0], reverse=False)[:self.k_value]
        return neighbors_vector

    def find_distance(self, x_1, x_2):
        return self.euclidean_distance(x_1, x_2)
         
    def euclidean_distance(self, x_1, x_2):
        dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x_1, x_2)]))
        return dist

