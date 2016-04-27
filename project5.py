import csv
import matplotlib.pyplot as plt
import numpy as np

from numpy import linalg as LA

def read_data():
    county_dict = {}
    # County dict
    with open("county_facts_dictionary.csv", "r") as f:
        csv_reader = csv.DictReader(f)
        for line in csv_reader:
            county_dict[line['column_name']] = line['description']

    county_results = []
    results_cols_names = []
    # County results
    with open("county_results.csv", "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        count = 0
        for line in csv_reader:
            if count == 0:
                results_cols_names = dict(zip(np.arange(58), line))
            else:
                county_results.append(line[2:])
            count += 1

    county_mat = np.array(county_results).astype(np.float32)


    return county_dict, results_cols_names, county_mat


# part b
def partb(county_mat):
    mean_cols = county_mat.mean(axis=0)

    # Mean 0 across cols
    county_mat -= mean_cols

    # Normalize across cols
    l2_norms = LA.norm(county_mat, axis=0)
    l2_norms = l2_norms.reshape(1, -1)
    county_mat /= l2_norms

    U, S, V = np.linalg.svd(county_mat, full_matrices=False)

    return U, S, V


def partc(U, S, V):
    v1 = V[:, 0]



if __name__ == "__main__":
    county_dict, results_cols_names, county_mat = read_data()


    # Part b
    U, S, V = partb(county_mat)
    plt.plot(np.arange(58), S)
    plt.xlabel("Singular Value Number")
    plt.ylabel("Singular Value")
    plt.title("Singule Value vs. Number")
    plt.show()

    # Part c
