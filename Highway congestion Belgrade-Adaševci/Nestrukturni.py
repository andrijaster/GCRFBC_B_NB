# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:04:06 2018

@author: Andrija Master
"""
""" Unstructured predictors """
def Nestrukturni_fun(x_train_un, y_train_un, x_train_st, y_train_st, x_test, y_test, No_class):
    
    import warnings
    import time
    warnings.filterwarnings('ignore')
    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import hamming_loss
    from keras.models import Sequential
    from keras.layers import Dense
    import keras
    import math
    
    plt.close('all')
    
    def evZ(x):
        return -np.log(1/x-1)
    
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    std_scl = StandardScaler()
    std_scl.fit(x_train_un)
    x_train_un1 = std_scl.transform(x_train_un)
    x_test1 = std_scl.transform(x_test)
    x_train_st1 = std_scl.transform(x_train_st)
    no_train = x_train_st.shape[0]
    no_test = x_test.shape[0]
    
    predictions_test = np.zeros([no_test, No_class])
    predictions1_test = np.zeros([no_test, No_class])
    predictions_rand_test = np.zeros([no_test, No_class])
    predictions_nn = np.zeros([no_test, No_class])
    
    Y_test_RF = np.zeros([no_test, No_class])
    Y_test_NN = np.zeros([no_test, No_class])
    Y_test_L2 = np.zeros([no_test, No_class])
    Y_test_L1 = np.zeros([no_test, No_class])
    
    Z_train = np.zeros([no_train, No_class]) 
    Z_test = np.zeros([no_test, No_class])
    Z1_train = np.zeros([no_train, No_class])
    Z1_test = np.zeros([no_test, No_class])
    Z2_train = np.zeros([no_train, No_class])
    Z2_test = np.zeros([no_test, No_class])
    Z2_train_un = np.zeros([no_train,No_class])
    Z3_test = np.zeros([no_test, No_class])
    Z3_train = np.zeros([no_train, No_class])
     
    skorAUC = np.zeros([1,4])
    skorAUC2 = np.zeros([No_class,4])
    ACC = np.zeros([1,4])
    HL = np.zeros([1,4])
    ACC2 = np.zeros([No_class,4])
    timeRF = np.zeros(No_class)
    timeNN = np.zeros(No_class)
    timeL2 = np.zeros(No_class)
    timeL1 = np.zeros(No_class)
    
    
    for i in range(No_class):
        
        kolone = np.array([x for x in range(13)])
        kolone1 = np.array([12+6*x+i+1 for x in range(9)])
        kolone = np.append(kolone,kolone1)
        x_train_un = x_train_un1[:,kolone]
        x_test = x_test1[:,kolone]
        x_train_st = x_train_st1[:,kolone]
        
        """ Random forest """
        start_time = time.time()
        rand_for = RandomForestClassifier(n_estimators=100)
        rand_for.fit(x_train_un, y_train_un.iloc[:,i])
        predictions_rand_test[:,i] = rand_for.predict_proba(x_test)[:,1]
        Z3_train[:,i] = evZ(rand_for.predict_proba(x_train_st)[:,1])
        Z3_test[:,i] = evZ(rand_for.predict_proba(x_test)[:,1])
        Y_test_RF[:,i] = rand_for.predict(x_test)
        timeRF[i] = time.time() - start_time
        
        
        """ Neural Network overfitted for strucured predictor """
        modelOF = Sequential()
        modelOF.add(Dense(30, input_dim = x_train_st.shape[1], activation='relu'))
        modelOF.add(Dense(15, activation='relu'))
        modelOF.add(Dense(8, activation='relu'))
        modelOF.add(Dense(1, activation='sigmoid'))
        modelOF.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
        modelOF.fit(x_train_st, y_train_st.iloc[:,i], epochs=600, batch_size=x_train_st.shape[0],validation_data=(x_train_st, y_train_st.iloc[:,i]))
        
        model2OF = Sequential()
        model2OF.add(Dense(30, input_dim=x_train_st.shape[1], weights = modelOF.layers[0].get_weights() ,activation='relu'))
        model2OF.add(Dense(15,weights = modelOF.layers[1].get_weights() , activation='relu'))
        model2OF.add(Dense(8,weights = modelOF.layers[2].get_weights() , activation='relu'))
        model2OF.add(Dense(1 , weights = modelOF.layers[3].get_weights(), activation='linear'))

        """ Neural Network - unstructured """        
        start_time = time.time()
        model = Sequential()
        model.add(Dense(20, input_dim = x_train_un.shape[1], activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
        ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto', baseline=None)
        model.fit(x_train_un, y_train_un.iloc[:,i], epochs=500, batch_size=250,validation_data=(x_test, y_test.iloc[:,i]), callbacks=[ES])
        Y_test_NN[:,i] = np.round(model.predict(x_test).T)
        timeNN[i] = time.time() - start_time
        
        model2 = Sequential()
        model2.add(Dense(20, input_dim=x_train_un.shape[1], weights = model.layers[0].get_weights() ,activation='relu'))
        model2.add(Dense(10,weights = model.layers[1].get_weights() , activation='relu'))
        model2.add(Dense(5,weights = model.layers[2].get_weights() , activation='relu'))
        model2.add(Dense(1 , weights = model.layers[3].get_weights(), activation='linear'))
        
        Z2_train[:,i] = model2.predict(x_train_st).reshape(no_train)
        Z2_train_un[:,i] = model2OF.predict(x_train_st).reshape(no_train)
        Z2_test[:,i] = model2.predict(x_test).reshape(no_test)
        
        predictions_nn[:,i] = sigmoid(Z2_test[:,i])
        
        """ Logistic regression L2 and L1"""
        logRegression = LogisticRegression(C = 1, penalty = 'l2')
        logRegression1 = LogisticRegression(C = 1, penalty = 'l1', solver='saga')
        
        start_time = time.time()
        logRegression.fit(x_train_un, y_train_un.iloc[:,i])
        Y_test_L2[:,i] = logRegression.predict(x_test)
        timeL2[i] = time.time() - start_time
        
        start_time = time.time()
        logRegression1.fit(x_train_un, y_train_un.iloc[:,i])
        Y_test_L1[:,i] = logRegression1.predict(x_test)
        timeL1[i] = time.time() - start_time
        
        predictions_test[:,i] = logRegression.predict_proba(x_test)[:,1]
        predictions1_test[:,i] = logRegression1.predict_proba(x_test)[:,1]
        
        Z_train[:,i] = logRegression.decision_function(x_train_st)
        Z_test[:,i] = logRegression.decision_function(x_test)
        Z1_train[:,i] = logRegression1.decision_function(x_train_st)
        Z1_test[:,i] = logRegression1.decision_function(x_test)        
        

               
        """ METRIC evaluation for each class """
        skorAUC2[i,0] = roc_auc_score(y_test.values[:,i],predictions_test[:,i])
        skorAUC2[i,1] = roc_auc_score(y_test.values[:,i],predictions1_test[:,i])    
        skorAUC2[i,2] = roc_auc_score(y_test.values[:,i],predictions_nn[:,i])
        skorAUC2[i,3] = roc_auc_score(y_test.values[:,i],predictions_rand_test[:,i])    
        
        ACC2[i,0] = accuracy_score(y_test.values[:,i], Y_test_L2[:,i]) 
        ACC2[i,1] = accuracy_score(y_test.values[:,i], Y_test_L1[:,i]) 
        ACC2[i,2] = accuracy_score(y_test.values[:,i], Y_test_NN[:,i]) 
        ACC2[i,3] = accuracy_score(y_test.values[:,i], Y_test_RF[:,i]) 
    
    """ METRICS EVALUATION """
    y_test = y_test.values
    
    skorAUC[:,0] = roc_auc_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),predictions_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorAUC[:,1] = roc_auc_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),predictions1_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    skorAUC[:,2] = roc_auc_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),predictions_nn.reshape([Z_test.shape[0]*Z2_test.shape[1]]))
    skorAUC[:,3] = roc_auc_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),predictions_rand_test.reshape([y_test.shape[0]*y_test.shape[1]]))
    
    HL[:,0] = hamming_loss(y_test,Y_test_L2)
    HL[:,1] = hamming_loss(y_test,Y_test_L1)
    HL[:,2] = hamming_loss(y_test,Y_test_NN)
    HL[:,3] = hamming_loss(y_test,Y_test_RF)
    
    ACC[:,0] = accuracy_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),Y_test_L2.reshape([y_test.shape[0]*y_test.shape[1]]))
    ACC[:,1] = accuracy_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),Y_test_L1.reshape([y_test.shape[0]*y_test.shape[1]]))
    ACC[:,2] = accuracy_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),Y_test_NN.reshape([y_test.shape[0]*y_test.shape[1]]))
    ACC[:,3] = accuracy_score(y_test.reshape([y_test.shape[0]*y_test.shape[1]]),Y_test_RF.reshape([y_test.shape[0]*y_test.shape[1]]))
    
    skorAUC2com = np.mean(skorAUC2, axis = 0)
    ACC2com = np.mean(ACC2, axis = 0)
    
    """ Unstructured predictors evaluation """
    Z_train_fin = np.concatenate((Z_train.reshape([Z_train.shape[0]*Z_train.shape[1],1]), \
                        Z1_train.reshape([Z1_train.shape[0]*Z1_train.shape[1],1])),axis=1)
    Z_test_fin = np.concatenate((Z_test.reshape([Z_test.shape[0]*Z_test.shape[1],1]), \
                        Z1_test.reshape([Z1_test.shape[0]*Z1_test.shape[1],1])),axis=1)
    Z_train_fin = np.concatenate((Z_train_fin, Z2_train.reshape([Z2_train.shape[0]*Z2_train.shape[1],1])),axis = 1)
    Z_test_fin = np.concatenate((Z_test_fin, Z2_test.reshape([Z2_test.shape[0]*Z2_test.shape[1],1])), axis = 1)
    Z_train_com = np.concatenate((Z_train_fin, Z3_train.reshape([Z3_train.shape[0]*Z3_train.shape[1],1])),axis = 1)
    Z_test_com = np.concatenate((Z_test_fin, Z3_test.reshape([Z3_test.shape[0]*Z3_test.shape[1],1])), axis = 1)    
    
    Noinst_train = np.round(Z_train_com.shape[0]/No_class).astype(int)
    Noinst_test = np.round(Z_test_com.shape[0]/No_class).astype(int)
    
    """ Standardization of unstructured predictors """
    Z_train_com[Z_train_com == -np.inf] = -10
    Z_train_com[Z_train_com == -10] = np.min(Z_train_com)-100
    Z_test_com[Z_test_com == -np.inf] = -10
    Z_test_com[Z_test_com == -10] = np.min(Z_test_com)-100
    Z2_train_un[Z2_train_un == -np.inf] = -10
    Z2_train_un[Z2_train_un == -10] = np.min(Z2_train_un)-100
    
    Z_train_com[Z_train_com == np.inf] = 10
    Z_train_com[Z_train_com == 10] = np.max(Z_train_com)+100
    Z_test_com[Z_test_com == np.inf] = 10
    Z_test_com[Z_test_com == 10] = np.max(Z_test_com)+100
    Z2_train_un[Z2_train_un == np.inf] = 10
    Z2_train_un[Z2_train_un == 10] = np.max(Z2_train_un)+100
    
    """ Time Unstructured """
    
    time = np.array([np.mean(timeL2), np.mean(timeL1), np.mean(timeNN), np.mean(timeRF)])

    for i in range(Z_train_com.shape[1]):
        Range = np.abs(np.max(Z_train_com[:,i]) + np.min(Z_train_com[:,i]))
        faktor = int(math.log10(Range))
        Z_train_com[:,i] = Z_train_com[:,i]*10**(-faktor)
        Z_test_com[:,i] = Z_test_com[:,i]*10**(-faktor)
    
    return skorAUC, skorAUC2com, ACC, ACC2com, HL, Z_train_com, Z_test_com, Z2_train_un, Noinst_train, Noinst_test, time

    
