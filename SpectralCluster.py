from KNeighbors import KNeighbors
import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import eigh
from KMeansCluster import KMeansCluster

class SpectralCluster:
    def __init__(self, clusters_number):
        self.k_value = clusters_number 
    
    def fit(self, X):
        adjacency_matrix = self.generate_adjacency_matrix(X=X)
        laplacian_matrix = self.generate_laplacian_matrix(adjacency_matrix=adjacency_matrix)
        laplacian_matrix = self.normalize_matrix(matrix=laplacian_matrix)
        eigenvectors = self.find_first_k_eigen_values_vectors(matrix=laplacian_matrix)
        feature_matrix = self.generate_feature_matrix(vectors=eigenvectors)
        counter = 0
        for i in feature_matrix:
            if i[0] == 0 and i[1] == 0 and i[2] == 0:
                counter =counter + 1
        print ("feature vector empty = " + str(counter))
        clt = KMeansCluster(cluster_number=self.k_value)
        clt.fit(X=feature_matrix)
        

    def generate_adjacency_matrix(self, X):
        adjacency_matrix = [ [0] * len(X) for _ in range(len(X))]
        neighbors_finder = KNeighbors(n_neighbors=self.k_value)
        for counter in range(len(X)):
            neighbors_list = neighbors_finder.find_neighbors(curr_index=counter, X=X)
            for i in range(len(X)):
                if i in [row[1] for row in neighbors_list]:
                    index = [row[1] for row in neighbors_list].index(i)
                    adjacency_matrix[counter][i] = 1 #neighbors_list[index][0]
                    adjacency_matrix[i][counter] = 1 #neighbors_list[index][0]
        return adjacency_matrix

    def generate_laplacian_matrix(self, adjacency_matrix):
        laplacian_matrix = [ [0] * len(adjacency_matrix) for _ in range(len(adjacency_matrix))]
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix)):
                if i == j:
                    laplacian_matrix[i][j] = self.calculate_diagonal_value(list_row=adjacency_matrix[i])
                elif adjacency_matrix[i][j] != 0:
                    laplacian_matrix[i][j] = -adjacency_matrix[i][j]
        
        return laplacian_matrix

    def calculate_diagonal_value(self, list_row):
        sum = 0
        for value in list_row:
            sum = sum + value
        return sum 

    def find_first_k_eigen_values_vectors(self, matrix):
        tmp_arr = np.asarray(matrix)
        eigenvalue, eigenvectors = eigh(matrix)
        eigenvalue_list = eigenvalue.tolist()
        eigenvectors_list = eigenvectors.tolist()
        result_list = list(zip(eigenvalue_list,eigenvectors_list))
        result_list = sorted(result_list, key=lambda a_entry: a_entry[0], reverse=False)[:self.k_value]
        return [row[1] for row in result_list]

    def normalize_matrix(self, matrix):
        normalized = normalize(matrix, norm='l1', axis=1)
        return normalized.tolist()

    def generate_feature_matrix(self, vectors):
        feature_array = np.column_stack(vectors)
        return feature_array.tolist()
