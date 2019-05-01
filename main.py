import matplotlib.pyplot as plt
from SpectralCluster import SpectralCluster
import csv

def load_csv_dataset(csv_name, x1_index, x2_index):
    with open("datasets/" + csv_name) as csv_file:
        inputs = []
        names = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i,row in enumerate(csv_reader):
            if i != 0:
                inp = []
                if row[x1_index] != "":
                    inp.append(float(row[x1_index]))
                else:
                    inp.append(0)
                if row[x2_index] != "":
                    inp.append(float(row[x2_index]))
                else:
                    inp.append(0)
                inputs.append(inp)
            else:
                names.append(row[x1_index])
                names.append(row[x2_index])
                
    return inputs, names

def load_dataset(txt_name):
    try:
        f = open("datasets/" + txt_name, 'r')
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

def plot_dataset(X,f1_name, f2_name):
    x_min, x_max = int(min([row[0] for row in X])) - 0.1, int(max([row[0] for row in X]) +  0.1)
    y_min, y_max = int(min([row[1] for row in X])) - 0.1, int(max([row[1] for row in X]) +  0.1)

    plt.scatter([row[0] for row in X], [row[1] for row in X], s=4, c='#00BFFF')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Dataset")
    plt.xlabel(f1_name, fontsize=8)
    plt.ylabel(f2_name, fontsize=8)
    plt.show()

def main():
    # Load dataset with given txt file
    X = load_dataset(txt_name="toy_dataset2.txt")
    
    # Load dataset with given CSV file(give index between 15 - 44)
    X, names = load_csv_dataset("nba_players_stats_2017.csv", x1_index=17, x2_index=18)
    # Plot initial dataset
    plot_dataset(X,f1_name=names[0],f2_name=names[1])

    # Create cluster object with clusters_number and start clustering process
    clt = SpectralCluster(clusters_number=5)
    clt.fit(X=X)
    
if __name__ == "__main__":
    main()
