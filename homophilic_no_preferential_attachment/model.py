""" Network formation model with homophily parameter without preferential attachment

@Author: Daniel Roncel Díaz

This script implements the first network formation model analyzed in my Final Degree 
Project, i.e. the network formation model inspired by [Karimi et al., 2018] in which
the preferential attachment mechanism has been removed.
"""

import numpy as np
import networkx as nx
import math
import random
import pickle
import os

## Util functions

def flatten(t):
    
    """
    e.g. [[1, 2, 3], [4, 5, 6]] --> [1, 2, 3, 4, 5, 6]
    """
    
    return [item for sublist in t for item in sublist]

def fix_probabilities(p):

    """
    Simple function to deal with numeric errors.
    If a probability is negative, assign it to 0.

    Parameters:
        p: List of probabilities.
    """
    
    for i in range(len(p)):
        if p[i] < 0:
            p[i] = 0
            
    return p


## Model

def model(n, m, c0, c1, h):
    
    """
    Parameters:
        n: Number of nodes of the output graph.
        m: Edges attached to the new node in each step.
        c0: Probability of a node to belong to the majority class.
        c1: Probability of a node to belong to the minority class.
        h: Homophily parameter, where h=1 corresponds to a completely homophilic network and h=0
           corresponds to a completely homophilic network.

    Return:
        G: networkx graph.
        node_class: list with the class of each node.
    """  

    # Define the class of each node.
    # node_class[i] = 0 if node i belongs to the majority class.
    # node_class[i] = 1 if node i belongs to the minority class.
    node_class = [0]*math.floor(c0 * n) + [1]*math.ceil(c1 * n)
    random.shuffle(node_class)
    
    # Create an empty graph with n nodes
    G = nx.empty_graph(n)
    G.name="model(%s,%s)"%(n,m)
    
    nodes = list(G.nodes)
    
    # probs_source_minority[i] ∝ Prob[node i recives an edge when a node from the minority class is generating edges]
    # probs_source_majority[i] ∝ Prob[node i recives an edge when a node from the majority class is generating edges]
    probs_source_minority = np.zeros(n)
    probs_source_majority = np.zeros(n)
    
    ## Fill probs_source_minority and probs_source_majority
    for j in range(n):
        # if node j belongs to the majority class
        if node_class[j] == 0:
            probs_source_minority[j] = 1-h
            probs_source_majority[j] = h
        # if node j belongs to the minority class
        else:
            probs_source_minority[j] = h
            probs_source_majority[j] = 1-h
    
    ## For each node i, create m edges with m different nodes
    for i in range(n):
        # Get the probability of each node of recieve an edge from node i
        if node_class[i] == 0:
            probs = probs_source_majority.copy()
        else:
            probs = probs_source_minority.copy()
        # Do not allow to create edges with itself
        probs[i] = 0
        # Normalize probabilities s.t. probabilities add up to 1
        probs = probs / sum(probs)
        
        # Sample m nodes at random
        neighbours = np.random.choice(nodes, replace=False, size=m, p=probs)
    
        # Add an edge between node i and each of the m sampled nodes
        G.add_edges_from(zip([i]*len(neighbours), neighbours))
        
    return G, node_class


def compute_simulation(n=5000, m=5, c0=0.5, c1=0.5, h=0.5,
                       n_simulations=1, folder_name='folder_name'):
    
    """
    Run n simulation of the model with the defined parameters. Return the last created graph and 
    a list of lists, with te degrees of the nodes in each of the two classes.

    Save in the specified directory a list with each of the graphs generated and a list with the
    class of each node in each of the executions.

    Parameters:
        n: Number of nodes of the output graph.
        m: Edges attached to the new node in each step.
        c0: Probability of a node to belong to the majority class.
        c1: Probability of a node to belong to the minority class.
        h: Homophily parameter, where h=1 corresponds to a completely homophilic network and h=0
            corresponds to a completely homophilic network.
        n_simulations: Number of simulations of the experiment.
        folder_name: Path of the directory to save the graph and the node classes.
    """

    # If this experiment has been already executed, skip it
    if os.path.exists(folder_name):
        print("file %s already exists" % folder_name)
        return

    ## variables to store all the data generated by the model
    G_simulation = []
    c_simulation = []
    
    for i in range(n_simulations):
        # get the resulting graph, the list with the class of each node,
        # and the degree evolution deagreggated by class
        G, c = model(n, m, c0, c1, h)

        # save all the data
        G_simulation.append(G)
        c_simulation.append(c)

    # Save the data from the executions
    os.makedirs(folder_name)
    pickle.dump(G_simulation, open(folder_name + '/G_simulation.pickle', 'wb'))
    pickle.dump(c_simulation, open(folder_name + '/c_simulation.pickle', 'wb'))

    return