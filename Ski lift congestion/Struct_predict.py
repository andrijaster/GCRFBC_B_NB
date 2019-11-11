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
#    from pystruct.learners import FrankWolfeSSVM
    from pystruct.models import MultiLabelClf
    from pystruct.models import GraphCRF
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    
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
    
    x_train = x_train.values
    y_train = y_train.values
    y_train = y_train.astype(int)
    y_test = y_test.values
    y_test = y_test.astype(int)
    x_test = x_test.values
    
    time_ST = np.zeros(7)
    HL = np.zeros(7)
    ACC = np.zeros(7)
    
    n_labels = y_train.shape[1]
    
    full = np.vstack([x for x in itertools.combinations(range(n_labels), 2)])
    tree = chow_liu_tree(y_train)
        
    """ CRF chain """
    train_tree = []
    train_full = []
    test_tree = []
    test_full = []
    for k in range(y_train.shape[0]):        
        X_train_CRF = np.zeros([y_train.shape[1], 18])
        for i in range(y_train.shape[1]):
            kolone = np.array([x for x in range(i*18,18*(i+1))])
            X_train_CRF[i,:] = x_train[k,kolone] 
        train_tree.append((X_train_CRF.copy(), tree.T))
        train_full.append((X_train_CRF.copy(), full.T))
        
    for k in range(y_test.shape[0]):        
        X_test_CRF = np.zeros([y_test.shape[1], 18])
        for i in range(y_test.shape[1]):
            kolone = np.array([x for x in range(i*18,18*(i+1))])
            X_test_CRF[i,:] = x_test[k,kolone] 
        test_tree.append((X_test_CRF.copy(), tree.T))
        test_full.append((X_test_CRF.copy(), full.T))

    """ SSVM, MLP, CRF-graph, DT - pystruct """
    """CREATE DATASET FOR GNN """  
    
    """ Define models """
    full_model = MultiLabelClf(edges=full)
    independent_model = MultiLabelClf()
    tree_model = MultiLabelClf(edges=tree, inference_method='max-product')
    
    modelCRF_tree = GraphCRF(directed=False, inference_method="max-product")
    modelCRF_full = GraphCRF(directed=False, inference_method="max-product")

    
    
    """ Define learn algorithm """
    full_ssvm = OneSlackSSVM(full_model, inference_cache=50, C=.1, tol=0.01, max_iter=150)
    tree_ssvm = OneSlackSSVM(tree_model, inference_cache=50, C=.1, tol=0.01, max_iter=150)
    independent_ssvm = OneSlackSSVM(independent_model, C=.1, tol=0.01, max_iter=150)
    MLP = MLPClassifier()
    DT = DecisionTreeClassifier()
    CRF_tree = OneSlackSSVM(model = modelCRF_tree, C=.1, max_iter=150)      
    CRF_full = OneSlackSSVM(model = modelCRF_full, C=.1, max_iter=150) 
    
    
    """ Fit models """
 
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
    
    start_time = time.time()
    MLP.fit(x_train, y_train)
    y_MLP = MLP.predict(x_test)
    time_ST[3] = time.time() - start_time
    
    start_time = time.time()
    DT.fit(x_train, y_train)
    y_DT = DT.predict(x_test)
    time_ST[4] = time.time() - start_time
    
    start_time = time.time()
    CRF_tree.fit(train_tree, y_train)
    yCRF_tree = np.asarray(CRF_tree.predict(test_tree))
    time_ST[5] = time.time() - start_time
    
    start_time = time.time()    
    CRF_full.fit(train_full, y_train)
    yCRF_full = np.asarray(CRF_full.predict(test_full))
    time_ST[6] = time.time() - start_time
    
    
  
    """ EVALUATE models """    
    y_full = np.asarray(y_full)
    y_ind = np.asarray(y_ind)
    y_tree = np.asarray(y_tree)
    
    HL[0] = hamming_loss(y_test,y_ind)
    HL[1] = hamming_loss(y_test,y_full)
    HL[2] = hamming_loss(y_test,y_tree)
    HL[3] = hamming_loss(y_test,y_MLP)
    HL[4] = hamming_loss(y_test,y_DT)
    HL[5] = hamming_loss(y_test,yCRF_tree)
    HL[6] = hamming_loss(y_test,yCRF_full)
    
    y_ind =  y_ind.reshape([y_ind.shape[0]*y_ind.shape[1]])
    y_full =  y_full.reshape([y_full.shape[0]*y_full.shape[1]])
    y_tree =  y_tree.reshape([y_tree.shape[0]*y_tree.shape[1]])
    y_MLP = y_MLP.reshape([y_MLP.shape[0]*y_MLP.shape[1]])
    y_DT = y_DT.reshape([y_DT.shape[0]*y_DT.shape[1]])
    yCRF_tree = yCRF_tree.reshape([yCRF_tree.shape[0]*yCRF_tree.shape[1]])
    yCRF_full = yCRF_full.reshape([yCRF_full.shape[0]*yCRF_full.shape[1]])
    y_test = y_test.reshape([y_test.shape[0]*y_test.shape[1]])
    
    
    ACC[0] = accuracy_score(y_test,y_ind)
    ACC[1] = accuracy_score(y_test,y_full)
    ACC[2] = accuracy_score(y_test,y_tree)
    ACC[3] = accuracy_score(y_test,y_MLP)
    ACC[4] = accuracy_score(y_test,y_DT)
    ACC[5] = accuracy_score(y_test,y_MLP)
    ACC[6] = accuracy_score(y_test,y_DT)
    
    return ACC, HL, time_ST

    
    






