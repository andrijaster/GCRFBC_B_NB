# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 08:20:07 2018

@author: Andrija Master
"""

""" Packages"""
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.model_selection import KFold

from Nestrukturni import Nestrukturni_fun
from Struct_predict import Strukturni
from Struktura import Struktura_fun 
from GCRFCNB import GCRFCNB
from GCRFC import GCRFC
from Emocije_dataset import output, atribute

""" Initialization """
No_class = 6
NoGraph = 4
ModelUNNo = 4
ModelSTNo = 5
testsize2 = 0.2
broj_fold = 10
iteracija = 400

output = output.iloc[:,:No_class]

AUCNB = np.zeros(broj_fold)
AUCB = np.zeros(broj_fold)


ACCNB = np.zeros(broj_fold)
ACCB = np.zeros(broj_fold)

HLNB = np.zeros(broj_fold)
HLB = np.zeros(broj_fold)

logProbNB = np.zeros(broj_fold)
logProbB = np.zeros(broj_fold)

timeNB = np.zeros(broj_fold)
timeB = np.zeros(broj_fold)

timeUN = np.zeros([broj_fold,ModelUNNo])


Skor_com_AUC = np.zeros([broj_fold,ModelUNNo])
Skor_com_AUC2 = np.zeros([broj_fold,ModelUNNo])
Skor_com_ACC = np.zeros([broj_fold,ModelUNNo])
Skor_com_ACC2 = np.zeros([broj_fold,ModelUNNo])
Skor_com_HL = np.zeros([broj_fold,ModelUNNo])

ACC_ST = np.zeros([broj_fold,ModelSTNo])
HL_ST = np.zeros([broj_fold,ModelSTNo])
time_ST = np.zeros([broj_fold,ModelSTNo])

skf = KFold(n_splits = broj_fold)
skf.get_n_splits(atribute, output)
i = 0

for train_index,test_index in skf.split(atribute, output):
    x_train_com, x_test = atribute.iloc[train_index,:], atribute.iloc[test_index,:]
    y_train_com, Y_test = output.iloc[train_index,:], output.iloc[test_index,:]
    provera = Y_test[Y_test==1].any().all()
    print(provera)

file = open("rezultatiEMOCIJE.txt","w")

for train_index,test_index in skf.split(atribute, output):
    
    x_train_com, x_test = atribute.iloc[train_index,:], atribute.iloc[test_index,:]
    y_train_com, Y_test = output.iloc[train_index,:], output.iloc[test_index,:] 
    x_train_un, x_train_st, y_train_un, Y_train = train_test_split(x_train_com, y_train_com, test_size=testsize2, random_state=31)

    """ STRUCTURED PREDICTORS """
    ACC_ST[i,:], HL_ST[i,:], time_ST[i,:] = Strukturni(x_train_com, y_train_com, x_test, Y_test)    
    """ UNSTRUCTURED PREDICTORS """
    Skor_com_AUC[i,:], Skor_com_AUC2[i,:], Skor_com_ACC[i,:], Skor_com_ACC2[i,:], Skor_com_HL[i,:], R_train, R_test, R2, Noinst_train, Noinst_test, timeUN[i,:] = Nestrukturni_fun(x_train_un, y_train_un, x_train_st, Y_train, x_test, Y_test, No_class)
    """ STructured matrix """
    Se_train, Se_test = Struktura_fun(No_class,NoGraph, R2 , y_train_com, Noinst_train, Noinst_test)
    
    
    """ Model GCRFC """
    Y_train = Y_train.values
    Y_test = Y_test.values 
    
    start_time = time.time()
    mod1 = GCRFCNB()
    mod1.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 6e-4, maxiter = iteracija)  
    probNB, YNB = mod1.predict(R_test,Se_test)
    timeNB[i] = time.time() - start_time
    
    start_time = time.time()
    mod2 = GCRFC()
    mod2.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija)  
    probB, YB, VarB = mod2.predict(R_test,Se_test)
    timeB[i] = time.time() - start_time 
    
    HLNB[i] = hamming_loss(Y_test,YNB)
    HLB[i] = hamming_loss(Y_test,YB)
       
    Y_test = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
    
    YNB =  YNB.reshape([YNB.shape[0]*YNB.shape[1]])
    probNB = probNB.reshape([probNB.shape[0]*probNB.shape[1]])
    YB =  YB.reshape([YB.shape[0]*YB.shape[1]])
    probB = probB.reshape([probB.shape[0]*probB.shape[1]])

    
    AUCNB[i] = roc_auc_score(Y_test,probNB)
    AUCB[i] = roc_auc_score(Y_test,probB)
    
    ACCNB[i] = accuracy_score(Y_test,YNB)
    ACCB[i] = accuracy_score(Y_test,YB)
    
    probNB[Y_test==0] = 1 - probNB[Y_test==0]
    probB[Y_test==0] = 1 - probNB[Y_test==0]
    
    logProbNB[i] = np.sum(np.log(probNB))
    logProbB[i] = np.sum(np.log(probB))

    
    file.write('AUC GCRFCNB prediktora je {}'.format(AUCNB[i]) + "\n")
    file.write('AUC GCRFCB prediktora je {}'.format(AUCB[i]) + "\n")

    file.write('ACC GCRFCNB prediktora je {}'.format(ACCNB[i]) + "\n")
    file.write('ACC GCRFCB prediktora je {}'.format(ACCB[i]) + "\n")
    
    file.write('HL GCRFCNB prediktora je {}'.format(HLNB[i]) + "\n")
    file.write('HL GCRFCB prediktora je {}'.format(HLB[i]) + "\n")

    file.write('AUC nestruktuiranih prediktora je {}'.format(Skor_com_AUC[i,:]) + "\n")
    file.write('AUC2 nestruktuiranih prediktora je {}'.format(Skor_com_AUC2[i,:]) + "\n")
    
    file.write('ACC nestruktuiranih prediktora je {}'.format(Skor_com_ACC[i,:]) + "\n")
    file.write('ACC2 nestruktuiranih prediktora je {}'.format(Skor_com_ACC2[i,:]) + "\n")
    
    file.write('HL nestruktuiranih prediktora je {}'.format(Skor_com_HL[i,:]) + "\n")
    
    file.write('Logprob GCRFCNB je {}'.format(logProbNB[i]) + "\n")
    file.write('Logprob GCRFCB je {}'.format(logProbB[i]) + "\n")
    
    file.write("--- %s seconds --- GCRFCNB" % (timeNB[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB" % (timeB[i]) + "\n")
    
    i= i + 1

file.write('CROSS AUC GCRFCNB prediktora je {}'.format(np.mean(AUCNB)) + "\n")
file.write('CROSS AUC GCRFCB prediktora je {}'.format(np.mean(AUCB)) + "\n")

file.write('CROSS ACC GCRFCNB prediktora je {}'.format(np.mean(ACCNB)) + "\n")
file.write('CROSS ACC GCRFCB prediktora je {}'.format(np.mean(ACCB)) + "\n")


file.write('CROSS HL GCRFCNB prediktora je {}'.format(np.mean(HLNB)) + "\n")
file.write('CROSS HL GCRFCB prediktora je {}'.format(np.mean(HLB)) + "\n")

file.write('CROSS ACC nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_ACC,axis=0)) + "\n")
file.write('CROSS ACC2 nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_ACC2,axis=0)) + "\n")
file.write('CROSS ACC strukturnih prediktora je {}'.format(np.mean(ACC_ST,axis=0)) + "\n")


file.write('CROSS AUC nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_AUC,axis=0)) + "\n")
file.write('CROSS AUC2 nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_AUC2,axis=0)) + "\n")

file.write('CROSS HL nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_HL,axis=0)) + "\n")
file.write('CROSS HL strukturnih prediktora je {}'.format(np.mean(HL_ST,axis=0)) + "\n")


file.write('CROSS Logprob GCRFCNB je {}'.format(np.mean(logProbNB)) + "\n")
file.write('CROSS Logprob GCRFCB je {}'.format(np.mean(logProbB)) + "\n")

file.write("--- %s seconds mean --- GCRFCNB" % (np.sum(timeNB)) + "\n")
file.write("--- %s seconds mean --- GCRFCB" % (np.sum(timeB)) + "\n")
file.write("--- %s seconds mean --- UNSTRUCTURED" % (np.sum(timeUN,axis = 0)) + "\n")   
file.write("--- %s seconds mean --- Strukturni" % (np.sum(time_ST,axis = 0)) + "\n")

file.close()

    