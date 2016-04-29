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

    # This is a correct normalization!!
    normed = (county_mat - county_mat.mean(axis=0)) / county_mat.std(axis=0)

    county_mat = normed
    print "sums: ", county_mat.sum(axis=0)

    U, S, V = np.linalg.svd(county_mat, full_matrices=False)

    return U, S, V


def partc(U, S, V, results_col_names, county_dict):
    v1 = V[0, :] # Note we are taking rows because SVD returns V transpose
    absv1 = np.abs(v1)
    sorted_idx = absv1.argsort()[::-1][:15]
    print "Sorted idx: ", sorted_idx

    for idx in sorted_idx:
        if idx <= 8:
            print results_col_names[idx], " ", v1[idx]
        else:
            print county_dict[results_col_names[idx]], " ", v1[idx]


def partd(U, S, V, results_col_names, county_dict):
    v2 = V[1, :] # Note we are taking rows because SVD returns V transpose
    absv2 = np.abs(v2)
    sorted_idx = absv2.argsort()[::-1][:15]
    print "Sorted idx: ", sorted_idx

    for idx in sorted_idx:
        if idx <= 8:
            print results_col_names[idx], " ", v2[idx]
        else:
            print county_dict[results_col_names[idx]], " ", v2[idx]


if __name__ == "__main__":
    county_dict, results_cols_names, county_mat = read_data()


    # Part b
    U, S, V = partb(county_mat)
    # plt.plot(np.arange(58), S)
    # plt.xlabel("Singular Value Number")
    # plt.ylabel("Singular Value")
    # plt.title("Singule Value vs. Number")
    # plt.show()

    # Part c
    #partc(U, S, V, results_cols_names, county_dict)

    # Part d

    partd(U, S, V, results_cols_names, county_dict)