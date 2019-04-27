import random
import copy
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import distance


class KMeansCluster:
    def __init__(self, cluster_number):
        self.k_value = cluster_number
        self.X_w_index = []

    def fit(self, X):
        # Copy array to class array which contains index
        self.X_w_index = copy.deepcopy(X)
        for point in self.X_w_index:
            point.append(-1)

        # find random cluster centers index
        clt_centers_index = self.generate_n_random_number(n=self.k_value, upper_range=len(X))

        # Set initial cluster centers
        init_clt_centers = []
        for index in clt_centers_index:
            init_clt_centers.append(X[index])
            
        # Start clustering progress
        return self.clustering_progress(X=X, clt_centers=init_clt_centers)
        
    def clustering_progress(self, X, clt_centers):
        is_under_threshold = False
        while (not is_under_threshold):
            count = 0
            for point in X:
                # set point to correct group index
                self.X_w_index[count][self.k_value] = self.find_nearest_clt_centers(point, clt_centers)
                count += 1 
            new_clt_centers = self.calculate_new_clt_centers(clt_centers)
            is_under_threshold = self.check_distance_change_threshold(clt_centers, new_clt_centers)
            clt_centers = new_clt_centers
        return [row[self.k_value] for row in self.X_w_index]

    def calculate_objective_function_value(self, clt_centers):
        group_difference = [0] * len(clt_centers)
        result = 0
        for point in self.X_w_index:
            group_difference[point[2]] = group_difference[point[2]] + self.square_of_distance([point[0],point[1]],clt_centers[point[2]])
        
        for diff in group_difference:
            result += diff

        return result

    def calculate_new_clt_centers(self, clt_centers):
        total_values = [ [0] * len(clt_centers) for _ in range(self.k_value)]
        group_element_count = [0] * len(clt_centers)
        new_clt_centers = [0] * len(clt_centers)
        count = 0
        for point in self.X_w_index:
            group_element_count[point[self.k_value]] += 1
            for i in range(len(point)-1):
                total_values[point[self.k_value]][i] = total_values[point[self.k_value]][i] + point[i]
            
        
        for values in total_values:
            new_clt_centers[count] = [x / group_element_count[count] for x in values]
            count += 1
        return new_clt_centers
            
    def find_nearest_clt_centers(self, point, clt_centers):
        distances = []
        for clt_center in clt_centers:
            distances.append(self.euclidean_distance(x_1=point, x_2=clt_center))

        min_index = 0
        min_distance = distances[0]
        count = 0
        for distance in distances:
            if distance < min_distance:
                min_distance = distance
                min_index = count
            count +=1

        return min_index

    def check_distance_change_threshold(self,first_clt_centers, last_clt_centers):
        distances = []
        count = 0
        for f_center in first_clt_centers:
            distances.append(self.euclidean_distance(x_1=f_center, x_2=last_clt_centers[count]))
            count += 1
        for distance in distances:
            if distance < 0.0001:
                return True
        return False

    def euclidean_distance(self, x_1, x_2):
        dist = distance.euclidean(x_1, x_2)
        return dist

    def generate_n_random_number(self, n, upper_range):
        rand_nums = []
        for i in range(n):
            while True:
                num = random.randint(0, upper_range-1)
                if num not in rand_nums:
                    rand_nums.append(num)
                    break
        return rand_nums
                 
    def square_of_distance(self, point_1, point_2):
        return pow(self.euclidean_distance(point_1,point_2),2)
        

        
