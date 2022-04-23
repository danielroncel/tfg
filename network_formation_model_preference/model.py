""" Network formation model with preference parameter

@Author: Daniel Roncel Díaz

This script implements the second network formation model analyzed in my Final Degree 
Project.
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

def model(n, m, m0, c0, p0, p1, p0i, p1i, beta=1):
    
    """
    Parameters:
        n: Number of nodes of the output graph
        m: Edges attached to the new node in each step
        m0: Number of nodes of the initial graph
        c0: Probability of a node to belong to the majority class
        p0: Probability of a node of having prior preference for the majority class
        p1: Probability of a node of having prior preference for the minority class
        p0i: Ratio of nodes with prior preference for the majority class that finally
             have no preference for any class.
        p1i: Ratio of nodes with prior preference for the minority class that finally
             have no preference for any class.
        beta: Parameter to calibrate the probability of attracting an edge.

    Return:
        G: networkx graph.
        node_class: List with the class of each node.
        k_majority: List with the degree of the majority class at each timestep.
        k_minority: List with the degree of the minority class at each timestep.
    """
    
    # Probability of a node to belong to the minority class
    c1 = 1 - c0
    
    # Fix possible numeric error
    p0, p1, p0i, p1i = fix_probabilities([p0, p1, p0i, p1i])
    
    # p0 = Probability that node i has preference of the majority class
    # p1 = Probability that node i has preference of the minority class
    # p2 = Probability that node i has no preferece for any class
    p2 = p0 * p0i + p1 * p1i
    p0 = p0 * (1-p0i)
    p1 = p1 * (1-p1i)
    
    # Assign random classes to the first m0 nodes
    node_class = [0]*math.floor(c0 * m0) + [1]*math.ceil(c1 * m0)
    random.shuffle(node_class)

    # Assign a random class to the rest of the nodes
    rest_node_class = [0]*math.floor(c0 * (n-m0)) + [1]*math.ceil(c1 * (n-m0))
    random.shuffle(rest_node_class)
    # Pull the class of all nodes in a single list. node_class[i] = class of the node i 
    node_class = np.array(node_class + rest_node_class)
    
    # Assign preferences at random for the first m0 nodes
    n_pref_majority = math.floor(p0*m0)
    n_pref_minority = math.ceil(p1*m0)
    n_no_pref = m0 - n_pref_majority - n_pref_minority
    node_preference = [0]*n_pref_majority + [1]*n_pref_minority + [2]*n_no_pref
    random.shuffle(node_preference)

    # Assign preferences at random for the rest of the nodes
    n_pref_majority = math.floor(p0 * (n-m0))
    n_pref_minority = math.ceil(p1 * (n-m0))
    n_no_pref = n - m0 - n_pref_majority - n_pref_minority
    rest_node_preference = [0]*n_pref_majority + [1]*n_pref_minority + [2]*n_no_pref
    random.shuffle(rest_node_preference)
    # Pull the class of all nodes in a single list. node_class[i] = class of the node i 
    node_preference = np.array(node_preference + rest_node_preference)

    # List to save the degree of each node
    node_degree = np.zeros(n)
    
    # Create an empty graph
    G = nx.empty_graph(n)
    
    G.name="custom_model(%s,%s)"%(n,m)

    
    # k_majority[i] = degree of the majority class at timestep i
    # k_minority[i] = degree of the minority class at timestep i
    k_majority = np.zeros(n)
    k_minority = np.zeros(n)
    
    # Next node to be added
    i = 0

    while i < n:
        
        class_i = node_class[i]
        preference_i = node_preference[i]
        
        nodes = None
        degrees = None
        # if the user has a preferece, get the nodes (and their degrees) only from that class
        if preference_i != 2:
            # if it is a node of the initial graph, retrieve all the other nodes
            # of the initial graph
            if i < m0:
                nodes = np.where(node_class[:m0] == preference_i)[0]
                # delete node i (a node cannot create an edge with itself)
                nodes = np.delete(nodes, np.where(nodes == i))
                degrees = node_degree[nodes]
            # otherwise, retrieve all the nodes already added to the graph
            else:
                # delete node i (a node cannot create an edge with itself)
                nodes = np.where(node_class[:i] == preference_i)[0]
                degrees = node_degree[nodes]
            
        # Otherwise, if the user has no preference, take into account all nodes
        # already added to the graph
        else:
            # if it is a node of the initial graph, retrieve all the other nodes
            # of the initial graph
            if i < m0:
                nodes = np.array([j for j in range(m0)])
                # delete node i (a node cannot create an edge with itself)
                nodes = np.delete(nodes, np.where(nodes == i))
                degrees = node_degree[:m0]
                degrees = np.delete(degrees, i)
            else:
                nodes = np.array([j for j in range(i)])
                degrees = node_degree[:i]

        # Select the m nodes to be attached to by preferential attachment
        # if there are less than m candidates nodes to be attached, add a link to all them
        if len(nodes) < m:
            neighbours = nodes
        else:
            # compute the probability of each node of receiving and edge
            exp_degrees = np.power(beta, degrees)
            probs = exp_degrees / np.sum(exp_degrees)
            # sample m nodes at random according to these probabilities
            neighbours = np.random.choice(nodes, replace=False, size=m, p=probs)
        
        # Add the new edges to the graph
        G.add_edges_from(zip([i]*len(neighbours), neighbours))
        
        # Update the degree of the neighbours
        node_degree[neighbours] = node_degree[neighbours] + 1
        # Update the degree of the new node
        node_degree[i] = len(neighbours)
        
        # compute class degree as the class degree in the previous step
        # plus the number of nodes of that class that received an edge
        k_majority[i] = k_majority[i-1] + len(np.where(node_class[neighbours] == 0)[0])
        k_minority[i] = k_minority[i-1] + len(np.where(node_class[neighbours] == 1)[0])
        # take into account also the degree of the new node
        if class_i == 0:
            k_majority[i] += len(neighbours)
        else:
            k_minority[i] += len(neighbours)
            
        # add the next node
        i += 1
    
    return G, node_class, k_majority, k_minority


def compute_simulation(n=5000, m=9, m0=10, c0=0.5, p0=0.5, p1=0.5, p0i=0, p1i=0, beta=1,
                       n_simulations=1, folder_name='folder_name'):
    
    """
    Run n simulation of the model with the defined parameters. Return the last created graph and 
    a list of lists, with te degrees of the nodes in each of the two classes.

    Save in the specified directory a list with each of the graphs generated and a list with the
    class of each node in each of the executions.

    Parameters:
        n: Number of nodes of the output graph
        m: Edges attached to the new node in each step
        m0: Number of nodes of the initial graph
        c0: Probability of a node to belong to the majority class
        p0: Probability of a node of having prior preference for the majority class
        p1: Probability of a node of having prior preference for the minority class
        p0i: Ratio of nodes with prior preference for the majority class that finally
             have no preference for any class.
        p1i: Ratio of nodes with prior preference for the minority class that finally
             have no preference for any class.
        beta: Parameter to calibrate the probability of attracting an edge.
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
    k_majority_simulation = []
    k_minority_simulation = []
    
    for i in range(n_simulations):
        # get the resulting graph, the list with the class of each node,
        # and the degree evolution deagreggated by class
        G, c, k_majority, k_minority = model(n=n, m=m, m0=m0, c0=c0, p0=p0, p1=p1, p0i=p0i, p1i=p1i, beta=beta)

        # save all the data
        G_simulation.append(G)
        c_simulation.append(c)
        k_majority_simulation.append(k_majority)
        k_minority_simulation.append(k_minority)

    # Save the data from the executions
    os.makedirs(folder_name)
    pickle.dump(G_simulation, open(folder_name + '/G_simulation.pickle', 'wb'))
    pickle.dump(c_simulation, open(folder_name + '/c_simulation.pickle', 'wb'))
    pickle.dump(k_majority_simulation, open(folder_name + '/k_majority_simulation.pickle', 'wb'))
    pickle.dump(k_minority_simulation, open(folder_name + '/k_minority_simulation.pickle', 'wb'))

    return