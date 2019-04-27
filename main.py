import matplotlib.pyplot as plt
from SpectralCluster import SpectralCluster

def load_dataset(txt_name):
    try:
        f = open("code/datasets/" + txt_name, 'r')
        inputs = []

        for line in f:
            # Split on any whitespace (including tab characters)
            row = line.split(",")
            inp = []

            # Convert strings to numeric values:
            inp.append(float(row[0]))
            inp.append(float(row[1]))
            
            # Append to our list of lists:
            inputs.append(inp)
        return inputs
    except Exception as e:
        print(e)

def plot_dataset(X):
    x_min, x_max = int(min([row[0] for row in X])) - 5, int(max([row[0] for row in X]) + 5)
    y_min, y_max = int(min([row[1] for row in X])) - 5, int(max([row[1] for row in X]) + 5)

    plt.scatter([row[0] for row in X], [row[1] for row in X], s=1, c='#00BFFF')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Dataset")
    plt.xlabel("Feature 1", fontsize=8)
    plt.ylabel("Feature 2", fontsize=8)
    plt.show()

def main():
    # Load dataset with given txt file
    X = load_dataset(txt_name="toy_dataset2.txt")
    # Plot initial dataset
    #plot_dataset(X)

    # Create cluster object with clusters_number and start clustering process
    clt = SpectralCluster(clusters_number=3)
    clt.fit(X=X)
    
if __name__ == "__main__":
    main()
