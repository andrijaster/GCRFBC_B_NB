# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:15:22 2019

@author: Andrija Master
"""

def Strukturni(x_train, y_train, x_test, y_test):
        
    import itertools
    import time

    import numpy as np
    from scipy import sparse
    
    from sklearn.metrics import hamming_loss
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mutual_info_score
    from scipy.sparse.csgraph import minimum_spanning_tree
    
    from pystruct.learners import OneSlackSSVM
    from pystruct.models import MultiLabelClf
#    from pystruct.models import GraphCRF
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    
    
    x_train = x_train.values
    y_train = y_train.values
    y_test = y_test.values
    x_test = x_test.values
    
    
    """ CRF chain """
    
    """ SSVM, MLP - pystruct """
    """CREATE DATASET FOR GNN """
    def chow_liu_tree(y_):
        n_labels = y_.shape[1]
        mi = np.zeros((n_labels, n_labels))
        for i in range(n_labels):
            for j in range(n_labels):
                mi[i, j] = mutual_info_score(y_[:, i], y_[:, j])
        mst = minimum_spanning_tree(sparse.csr_matrix(-mi))
        edges = np.vstack(mst.nonzero()).T
        edges.sort(axis=1)
        return edges
    
    n_labels = y_train.shape[1]
    full = np.vstack([x for x in itertools.combinations(range(n_labels), 2)])
    tree = chow_liu_tree(y_train)
    
    """ Define models """
    full_model = MultiLabelClf(edges=full)
    independent_model = MultiLabelClf()
    tree_model = MultiLabelClf(edges=tree, inference_method='max-product')
    
    """ Define learn algorithm """
    full_ssvm = OneSlackSSVM(full_model, inference_cache=50, C=.1, tol=0.01, max_iter=150)
    tree_ssvm = OneSlackSSVM(tree_model, inference_cache=50, C=.1, tol=0.01, max_iter=150)
    independent_ssvm = OneSlackSSVM(independent_model, C=.1, tol=0.01, max_iter=150)
    
    MLP = MLPClassifier()
    DT = DecisionTreeClassifier()
    
    """ Fit models """
    
    time_ST = np.zeros(5)

    start_time = time.time()
    DT.fit(x_train, y_train)
    y_DT = DT.predict(x_test)
    time_ST[4] = time.time() - start_time
    
    start_time = time.time()
    MLP.fit(x_train, y_train)
    y_MLP = MLP.predict(x_test)
    time_ST[3] = time.time() - start_time
    
    start_time = time.time()
    independent_ssvm.fit(x_train, y_train)
    y_ind = independent_ssvm.predict(x_test)
    time_ST[0] = time.time() - start_time

    start_time = time.time()
    full_ssvm.fit(x_train, y_train)
    y_full = full_ssvm.predict(x_test)
    time_ST[1] = time.time() - start_time


    start_time = time.time()
    tree_ssvm.fit(x_train, y_train)
    y_tree = tree_ssvm.predict(x_test)    
    time_ST[2] = time.time() - start_time
  
    """ EVALUATE models """
    HL = np.zeros(5)
    ACC = np.zeros(5)
    
    y_full = np.asarray(y_full)
    y_ind = np.asarray(y_ind)
    y_tree = np.asarray(y_tree)
    
    HL[0] = hamming_loss(y_test,y_ind)
    HL[1] = hamming_loss(y_test,y_full)
    HL[2] = hamming_loss(y_test,y_tree)
    HL[3] = hamming_loss(y_test,y_MLP)
    HL[4] = hamming_loss(y_test,y_DT)
    
    y_ind =  y_ind.reshape([y_ind.shape[0]*y_ind.shape[1]])
    y_full =  y_full.reshape([y_full.shape[0]*y_full.shape[1]])
    y_tree =  y_tree.reshape([y_tree.shape[0]*y_tree.shape[1]])
    y_MLP = y_MLP.reshape([y_MLP.shape[0]*y_MLP.shape[1]])
    y_DT = y_DT.reshape([y_DT.shape[0]*y_DT.shape[1]])
    y_test = y_test.reshape([y_test.shape[0]*y_test.shape[1]])
    
    ACC[0] = accuracy_score(y_test,y_ind)
    ACC[1] = accuracy_score(y_test,y_full)
    ACC[2] = accuracy_score(y_test,y_tree)
    ACC[3] = accuracy_score(y_test,y_MLP)
    ACC[4] = accuracy_score(y_test,y_DT)
    
    return ACC, HL, time_ST

    
    






