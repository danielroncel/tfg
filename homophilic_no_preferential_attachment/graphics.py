""" Network formation model with homophily parameter without preferential attachment

@Author: Daniel Roncel Díaz

Script with functions to run load data and create plots.
"""

import numpy as np
import statistics
import math
from collections import Counter
import pickle
import matplotlib.pyplot as plt

## Utils
base_colors = ['orange', 'blue']


def flatten(t):
    
    """
    e.g. [[1, 2, 3], [4, 5, 6]] --> [1, 2, 3, 4, 5, 6]
    """
    
    return [item for sublist in t for item in sublist]


def get_folder_name(n, m, c0, c1, h):

    """
    Given the parameters of the model, return the name of the folder
    that contains (or will contain) it.
    """

    return 'n_'+str(n)+'_m_'+str(m)+'_c0_'+str(c0)+'_c1_'+str(c1)+'_h_'+str(h)


def load_simulation_data(n, m, n_simulations, folder_name):

    """
    Given the directory where the results of an experiment are saved (see 
    compute_simulation function in model.py) returns a list with the degrees 
    of the nodes of each class for each simulation.

    Parameters:
        n: Number of nodes of the output graph.
        m: Edges attached to the new node in each step.
        n_simulations: Number of simulations of the experiment.
        folder_name: Path of the directory to save the graph and the node classes.

    Return:
        G: Example of networkx graph.
        degrees_majority: List of lists of the degrees of the nodes of the majority class
                          for each simulation.
        degrees_minority: List of lists of the degrees of the nodes of the minority class
                          for each simulation.
        sorted_degree_per_class: List of one list for each simulation of the class of each
                                 node sorted by degree non-descending.
    """

    # load the list of graphs created during at the end of the experiment
    G_simulation = pickle.load(open(folder_name + '/G_simulation.pickle', 'rb'))
    # load the list of list of the class of each node at the end of the experiment
    c_simulation = pickle.load(open(folder_name + '/c_simulation.pickle', 'rb'))
    
    # degrees_majority[i] = list of the degrees of the nodes of the majority class
    # at the end of the i-th simulation. Analogous definition for degrees_minority[i]
    degrees_majority = []
    degrees_minority = []

    # sorted_degree_per_class[i][j] = 0 if the j-th node with largest degree in the
    # i-th simulation belongs to the majority class.
    # sorted_degree_per_class[i][j] = 1 otherwise.
    # sorted_degree_per_class[i][0] is the class of the node with largest degree
    # in the i-th simulation
    sorted_degree_per_class = np.zeros((n_simulations, n))

    for i in range(n_simulations):

        G, c = G_simulation[i], c_simulation[i]
        nodes_degree = G.degree()
        # list with the degrees of only the nodes of the minority class
        d_majority = sorted([degree for node, degree in nodes_degree if c[node] == 0])
        # analogous for the minority class
        d_minority = sorted([degree for node, degree in nodes_degree if c[node] == 1])
        
        # store them
        degrees_majority.append(d_majority)
        degrees_minority.append(d_minority)

        # node_degree = list of tuples of the form (node name, degree)
        # node_degree is sorted non-descending by degree
        node_degree = sorted( dict(G.degree()).items(), key=lambda item: item[1], reverse=True)
        # for each node in node_degree, replace it by its class
        degree_per_class = np.array([c[nd[0]] for nd in node_degree])
        sorted_degree_per_class[i:,] = degree_per_class

    return G, degrees_majority, degrees_minority, sorted_degree_per_class


def load_data_for_plots(n, m, c0, c1, h, n_simulations, folder_name):

    """
    Load the data of several experiments at the same time to simplify creating the plots.

    Parameters:
        n: Number of nodes of the output graph.
        m: Edges attached to the new node in each step.
        c0: List with the probability of a node to belong to the majority class in each experiment.
        c1: List with the probability of a node to belong to the minority class in each experiment.
        h: List with the homophily parameter for each execution.
        n_simulations: Number of simulations of the experiment.
        folder_name: Path of the directory to save the graph and the node classes.

    Returns:
        avg_degree_majority: Average normalized degree of the majority class.
        avg_degree_minority: Average normalized degree of the minority class.
        degree_dist_majority: Data for the degree distribution plot of the majority class.
        degree_dist_minority: Data for the degree distribution plot of the minority class.
        results: Percentage of nodes of the minority class within the top nodes with largest degree.
        step: Defines which percentage of nodes with largest degree has been used to create 'results' variable.
    """

    # avg_degree_majority[i] = mean value of majority_degree / (majority_degree + minority_degree)
    # across all the simulations of the i-th experiment.
    # Analogous definition for avg_degree_minority[i]
    avg_degree_majority = []
    avg_degree_minority = []

    # degree_dist_majority[i][j] = mean number of nodes of the majority class with degree j in the i-th experiment
    # Analogous definition for degree_dist_minority
    degree_dist_majority = []
    degree_dist_minority = []

    # e.g. if steps[0] = 0.1 --> we will measure the percentage of nodes in the minority class in the 10%
    # nodes with largest degree 
    step = list(np.linspace(0, 1, 11))[1:]
    # results[i] = for the specified value of h, returns the percentage of nodes in the minority class
    # with largest degree within the top defined by variable 'step'
    results = np.zeros((len(h), len(step)))

    for i in range(len(c0)):
        
        # get the full path where the results of the i-th experiment are stored
        final_path = folder_name + '/' + get_folder_name(n=n, m=m, c0=c0[i], c1=c1[i], h=h[i])

        ## Get the degrees of the simulations
        _, degrees_majority, degrees_minority, sorted_degree_per_class = load_simulation_data(n, m, n_simulations, final_path)
        
        # lists of the average degree of the majority class on each simulation of this experiment
        k_majority = []
        # lists of the average degree of the minority class on each simulation of this experiment
        k_minority = []
        
        # For each simulation, compute the mean average degree of each class.
        for j in range(n_simulations):
            k = sum(degrees_majority[j]) + sum(degrees_minority[j])
            k_majority.append( sum(degrees_majority[j]) / k)
            k_minority.append( sum(degrees_minority[j]) / k)
            
        # Compute the mean degree of the majority class across all simulations of this experiment.
        avg_degree_majority.append(statistics.mean(k_majority))
        # Compute the mean degree of the minority class across all simulations of this experiment.
        avg_degree_minority.append(statistics.mean(k_minority))
        
        ##Data for the degree distribution plot
        
        # merge all the degrees obtained by nodes of the majority class in all the simulations
        # of this experiment
        degrees_majority = flatten(degrees_majority)
        max_majority = max(degrees_majority)
        ctr_max = Counter(degrees_majority)
        
        # data_majority[i] = number of nodes with degree i in some simulation
        data_majority = [ctr_max[i] if i in ctr_max else 0 for i in range(max_majority)]
        degree_dist_majority.append(data_majority)
        
        # merge all the degrees obtained by nodes of the minority class in all the simulations
        # of this experiment
        degrees_minority = flatten(degrees_minority)
        max_minority = max(degrees_minority)
        ctr_max = Counter(degrees_minority)
        
        # data_minority[i] = number of nodes with degree i in some simulation
        data_minority = [ctr_max[i] if i in ctr_max else 0 for i in range(max_minority)]
        degree_dist_minority.append(data_minority)

        # sorted_degree_per_class[i][j] = mean number of nodes of the minority class between the (j+1)-th nodes
        # with largest degree in this experiment.
        sorted_degree_per_class = np.cumsum(sorted_degree_per_class,axis=1).mean(axis=0)
    
        # results[i][j] = mean number of nodes of the minority class in the (step[j]*n) top nodes with largest degree
        for j in range(len(step)):
            results[i, j] = sorted_degree_per_class[math.floor(step[j]*n) - 1] / math.floor(step[j] * n)

    return avg_degree_majority, avg_degree_minority, degree_dist_majority, degree_dist_minority, results, step

## Functions for plot

def class_degree_barplot(avg_degree_minority, avg_degree_majority, c0, c1, h, figsize=(16,5)):

    global base_colors

    # Average degree barplot
    fig, axes = plt.subplots(1, len(c0), figsize=figsize, constrained_layout=True)
    fig.suptitle('Grau de cada classe', fontsize=15)

    for i in range(len(c0)):
        x = ['Minoritària', 'Majoritària']
        y = [avg_degree_minority[i], avg_degree_majority[i]]
        axes[i].bar(x, 
                    y,
                    alpha=0.8,
                    color=base_colors)

        axes[i].set_title('h=%.2f' % (h[i]))
        axes[i].set_ylabel('K_i / K')
        axes[i].set_ylim(0.0, 1.0)
        # Reference lines
        axes[i].axhline(c0[i], linestyle='--', color='b')
        axes[i].axhline(c1[i], linestyle='--', color='y')
        axes[i].grid()
        
    plt.show()


def degree_distribution(degree_dist_minority, degree_dist_majority, c0, c1, h, figsize=(16,5)): 

    global base_colors

    # Degree distribution plot
    fig, axes = plt.subplots(1, len(c0), figsize=figsize, constrained_layout=True)
    fig.suptitle('Distribució de grau de cada classe', fontsize=16)

    for i in range(len(c0)):
        # Plot the dist. of the minority class
        degree_hist = np.array(degree_dist_minority[i], dtype=float)
        degree_prob = degree_hist / len(degree_dist_minority[i])
        axes[i].loglog(np.arange(degree_prob.shape[0]),degree_prob,'.', color=base_colors[0], alpha=0.8)
        
        # Plot the dist. of the majority class
        degree_hist = np.array(degree_dist_majority[i], dtype=float)
        degree_prob = degree_hist / len(degree_dist_majority[i])
        axes[i].loglog(np.arange(degree_prob.shape[0]),degree_prob,'.', color=base_colors[1], alpha=0.8)
        
        axes[i].set_title('h=%.2f' % (h[i]))
        axes[i].grid()
        axes[i].set_xlabel('k')
        axes[i].set_ylabel('prob(k)')


def minority_in_top_d(results, step, c0, c1, h):

    for i in range(len(h)):
        plt.plot(step, results[i,:], alpha=0.8)

    plt.xlabel("% d")
    plt.ylabel("Num. of nodes of the minority class in the top d% with larger degree")
    plt.legend(["h=%.1f" % (h[i]) for i in range(len(h))])
    plt.axis([0, 1, -0.01, 1.01])
    # Reference line
    plt.axhline(y=c1[i], xmin=0, xmax=1, linestyle='--', color='k', linewidth=1.5)
    plt.grid()