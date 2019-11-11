# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 09:14:13 2018

@author: Andrija Master
"""

import pandas as pd
import scipy.stats as sp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score


""" Structured matrix definition """

def Struktura_fun(No_class,NoGraph,R2,y_train_com, Noinst_train, Noinst_test, koef1 = 0.5):

    Se = np.zeros([NoGraph,No_class,No_class])
    y_train_com = y_train_com.values
    
    for i in range(No_class):
        for j in range(i+1,No_class):
            Mut_info = mutual_info_score(y_train_com[:45000,i].astype(int),y_train_com[:45000,j].astype(int))
            Mat = pd.crosstab(y_train_com[:,i],y_train_com[:,j])
            chi2, pvalue, dof, ex = sp.chi2_contingency(Mat)
            Se[0,i,j] = chi2
            print([chi2,pvalue])
            Se[0,j,i] = Se[0,i,j]
            Se[1,i,j] = Mut_info
            Se[1,j,i] = Se[1,i,j]  
            Se[2,i,j] = np.exp(-koef1*np.sum(np.abs(y_train_com[:,i]-y_train_com[:,j])))
            Se[2,j,i] = Se[2,i,j]
            
    scaler = StandardScaler()
    R2 = R2.reshape([R2.shape[0]*R2.shape[1],1])
    R2[R2==-np.inf] = -10
    R2[R2==np.inf] = 10
    R2[R2==-np.inf] = np.min(R2) - 10
    R2[R2==np.inf] = np.max(R2) + 10
    scaler.fit(R2)
    R2 = scaler.transform(R2)
    R2 = R2.reshape([int(R2.shape[0]/No_class),No_class])
    
    Corelation_mat = np.corrcoef(R2.T)
    Corelation_mat[Corelation_mat<0] = 0
    np.fill_diagonal(Corelation_mat,0)
    
    Se[3,:,:] = Corelation_mat
    
    Se_train = np.zeros([Noinst_train,NoGraph,No_class,No_class])
    Se_test = np.zeros([Noinst_test,NoGraph,No_class,No_class])
    
    for i in range(Noinst_train):
        Se_train[i,:,:,:] = Se
    
    for i in range(Noinst_test):
        Se_test[i,:,:,:] = Se  
    
    return Se_train, Se_test