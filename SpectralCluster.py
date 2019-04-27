from KNeighbors import KNeighbors
import numpy as np
from sklearn.preprocessing import normalize
import scipy
from KMeansCluster import KMeansCluster
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class SpectralCluster:
    def __init__(self, clusters_number):
        self.k_value = clusters_number 
    
    def fit(self, X):
        adjacency_matrix = self.generate_adjacency_matrix(X=X)
        laplacian_matrix = self.generate_laplacian_matrix(adjacency_matrix=adjacency_matrix)
        laplacian_matrix = self.normalize_matrix(matrix=laplacian_matrix)

        feature_matrix = self.find_first_k_eigen_values_vectors(matrix=laplacian_matrix)
        
        # Create KMeans Cluster and start clustering process on feature matrix
        clt = KMeansCluster(cluster_number=self.k_value)
        labels = clt.fit(X=feature_matrix)

        # Merge data and labels and plot
        X = list(zip(X,labels))
        self.plot_data(X=X)
        
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
        tmp_arr = tmp_arr.astype(float)
        eig_val, eig_vect = scipy.sparse.linalg.eigs(tmp_arr, self.k_value)
        X = eig_vect.real
        rows_norm = numpy.linalg.norm(X, axis=1, ord=2)
        Y = (X.T / rows_norm).T
        return Y.tolist()

    def normalize_matrix(self, matrix):
        normalized = normalize(matrix, norm='l1', axis=1)
        return normalized.tolist()

    def generate_feature_matrix(self, vectors):
        feature_array = np.column_stack(vectors)
        return feature_array.tolist()

    def plot_data(self, X):
        # Create color maps
        cmap_bold = ListedColormap(['#9400D3', '#4B0082', '#00FF00', '#0000FF', '#FFFF00','#FF7F00','#FF0000','#A9A9A9'])

        x_min, x_max = min([row[0][0] for row in X]) - 1, max([row[0][0] for row in X]) + 1
        y_min, y_max = min([row[0][1] for row in X]) - 1, max([row[0][1] for row in X]) + 1
        
        plt.scatter([row[0][0] for row in X], [row[0][1] for row in X], c=[row[1] for row in X], cmap=cmap_bold, s=0.3)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        #plt.title("k=%i\n%i. Iteration" % (self.k_value, count))
        plt.xlabel("x1", fontsize=8)
        plt.ylabel("x2", fontsize=8)
        plt.show()