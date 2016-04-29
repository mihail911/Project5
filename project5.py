import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from numpy import linalg 
import copy

def read_data():
    county_dict = {}
    # County dict
    with open("county_facts_dictionary.csv", "r") as f:
        csv_reader = csv.DictReader(f)
        for line in csv_reader:
            county_dict[line['column_name']] = line['description']

    county_results = []
    results_cols_names = []
    states = []
    # County results
    with open("county_results.csv", "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        count = 0
        for line in csv_reader:
            if count == 0:
                results_cols_names = dict(zip(np.arange(58), line[2:]))
                colnames_to_indices = dict(zip(line[2:], np.arange(58)))
            else:
                county_results.append(line[2:])
                states.append(line[0])
            count += 1

    county_mat = np.array(county_results).astype(np.float32)


    return county_dict, results_cols_names, county_mat, colnames_to_indices, states


# part b
def partb(county_mat):
    mean_cols = county_mat.mean(axis=0)

    # This is a correct normalization!!
    normed = (county_mat - county_mat.mean(axis=0)) / county_mat.std(axis=0)

    county_mat = normed
    #print "sums: ", county_mat.sum(axis=0)

    U, S, V = np.linalg.svd(county_mat, full_matrices=False)

    return U, S, V


def partc(U, S, V, results_col_names, county_dict):
    v1 = V[0, :] # Note we are taking rows because SVD returns V transpose
    absv1 = np.abs(v1)
    sorted_idx = absv1.argsort()[::-1][:15]
    print "Sorted idx: ", sorted_idx

    for idx in sorted_idx:
        if idx <= 6:
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

def parte(U, S, V, results_col_names, county_dict, county_mat, colnames):
    v1 = V[0, :]; v2 = V[1, :]; v3 = V[2, :]
    proj_clinton = []; proj_sanders = []
    proj_trump = []; proj_cruz = []; proj_rubio = []
    for county in U:
        proj1 = np.dot(county, v1); proj2 = np.dot(county, v2); proj3 = np.dot(county, v3)
        proj1 = county[0]; proj2 = county[1]; proj3 = county[2]
        if (county[colnames["Clinton"]] > county[colnames["Sanders"]]):
            proj_clinton.append((proj1, proj2))
        else:
            proj_sanders.append((proj1, proj2))

        if (county[colnames["Trump"]] > county[colnames["Rubio"]] and 
            county[colnames["Trump"]] > county[colnames["Cruz"]]):
            proj_trump.append((proj1, proj3))
        elif (county[colnames["Rubio"]] > county[colnames["Trump"]] and
            county[colnames["Rubio"]] > county[colnames["Cruz"]]):
            proj_rubio.append((proj1, proj3))
        elif (county[colnames["Cruz"]] > county[colnames["Rubio"]] and
            county[colnames["Cruz"]] > county[colnames["Trump"]]):
            proj_cruz.append((proj1, proj3))

    fig, plot = plt.subplots()
    a = plot.plot([x[1] for x in proj_sanders], [x[0] for x in proj_sanders], 'bo', label="Sanders")
    b = plot.plot([x[1] for x in proj_clinton], [x[0] for x in proj_clinton], 'ro', label="Clinton")
    plt.title("Clinton vs Sanders")
    legend = plot.legend()
    plt.ylabel('2nd right singular vector')
    plt.xlabel('1st right singular vector')
    plt.show()


    fig, plot = plt.subplots()
    plot.plot([x[1] for x in proj_trump], [x[0] for x in proj_trump], 'bo', label="Trump")
    plot.plot([x[1] for x in proj_rubio], [x[0] for x in proj_rubio], 'ro', label="Rubio")
    plot.plot([x[1] for x in proj_cruz], [x[0] for x in proj_cruz], 'go', label="Cruz")
    plt.title("Trump vs Cruz vs Rubio")
    legend = plot.legend()
    plt.ylabel('2nd right singular vector')
    plt.xlabel('1st right singular vector')
    plt.show()

states_desired = ["Georgia", "South Carolina", "Iowa", "Oklahoma"]#, "Texas"]
colors = ['bo', 'ro', 'go', 'co', 'mo']
def partf(U, S, V, states):
    # georgia_counties = [i for i in xrange(len(states)) if states[i] == "Georgia"]
    # sc_counties = [i for i in xrange(len(states)) if states[i] == "South Carolina"]
    # iowa_counties = [i for i in xrange(len(states)) if states[i] == "Iowa"]
    # oklahoma_counties = [i for i in xrange(len(states)) if states[i] == "Oklahoma"]
    # texas_counties = [i for i in xrange(len(states)) if states[i] == "Texas"]
    county_projections = []
    plt.figure(figsize=(20, 20))
    for i in xrange(5):
        for j in xrange(i + 1, 5):
            plt.subplot(5, 5, i * 5 + j)
            for k, state in enumerate(states):
                if state == "Texas":
                    plt.plot([U[k][i]], [U[k][j]], 'mo')
            for k, state in enumerate(states):
                #print state, k
                if state not in states_desired:
                    continue
                #print "HELLO"
                plt.plot([U[k][i]], [U[k][j]], colors[states_desired.index(state)])
            plt.xlabel('#' + str(i + 1) + " right singular vector")
            plt.ylabel('#' + str(j + 1) + " right singular vector")
    plt.show()

def partg(county_mat):
    # random_indices = np.random.randint(0, high=1008, size=100)
    # missing_entries_mat = copy.deepcopy(county_mat)
    # for i in random_indices:
    #     missing_entries_mat[i] = np.zeros(58)
    
    # mean_cols = missing_entries_mat.mean(axis = 0)
    # for i in random_indices:
    #     missing_entries_mat[i] -= missing_entries_mat.mean(axis = 0)
    
    # norm_cols = missing_entries_mat.std(axis=0)
    # missing_entries_mat = (missing_entries_mat - mean_cols) / norm_cols

    # U, s, V = np.linalg.svd(missing_entries_mat, full_matrices=False)
    ks = [5, 10, 20, 25]
    accuracy_map = defaultdict(int)
    for iteration in xrange(500):
        random_indices = np.random.randint(0, high=1008, size=100)
        missing_entries_mat = copy.deepcopy(county_mat)
        for i in random_indices:
            missing_entries_mat[i] = np.zeros(58)
        
        mean_cols = missing_entries_mat.mean(axis = 0)
        for i in random_indices:
            missing_entries_mat[i] -= missing_entries_mat.mean(axis = 0)
        
        norm_cols = missing_entries_mat.std(axis=0)
        missing_entries_mat = missing_entries_mat / norm_cols

        U, s, V = np.linalg.svd(missing_entries_mat, full_matrices=False)

        for k in ks:
            low_rank_approx = np.dot(np.dot(np.array(U[:, :k]), np.diag(s[:k])), np.array(V[:k, :]))
            rescaled_approx = (low_rank_approx * norm_cols) + mean_cols

            num_correct_d = 0; num_correct_r = 0

            for idx in random_indices:
                if (np.argmax(rescaled_approx[idx][0:5]) == np.argmax(county_mat[idx][0:5])):
                    num_correct_r += 1
                if (np.argmax(rescaled_approx[idx][5:7]) == np.argmax(county_mat[idx][5:7])):
                    num_correct_d += 1
            accuracy_map[(k, "Republican")] += num_correct_r
            accuracy_map[(k, "Democrat")] += num_correct_d


        accuracy_map[("STUPID REPULICANS")] += sum([1 for i in random_indices if np.argmax(county_mat[idx][0:5]) == 0])
        accuracy_map[("STUPID DEMOCRATS")] += sum([1 for i in random_indices if np.argmax(county_mat[idx][5:7]) == 0])
    for key in accuracy_map.keys():
        accuracy_map[key] /= (500.0 * 100) 

    print accuracy_map

    # 2, 3? 2, 4?
if __name__ == "__main__":
    county_dict, results_cols_names, county_mat, colnames_to_indices, states = read_data()


    # Part b
    U, S, V = partb(county_mat)
    # plt.plot(np.arange(58), S)
    # plt.xlabel("Singular Value Number")
    # plt.ylabel("Syingular Value")
    # plt.title("Singule Value vs. Number")
    # plt.show()

    # Part c
#    partc(U, S, V, results_cols_names, county_dict)

    # Part d

    #partd(U, S, V, results_cols_names, county_dict)

    #parte(U, S, V, results_cols_names, county_dict, county_mat, colnames_to_indices)

    #partf(U, S, V, states)