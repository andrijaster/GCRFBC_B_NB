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
from GCRFC_fast import GCRFC_fast
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
AUCBF = np.zeros(broj_fold)
AUCBF3_1 = np.zeros(broj_fold)
AUCBF3_2 = np.zeros(broj_fold)
AUCBF3_3 = np.zeros(broj_fold)
AUCBF5_1 = np.zeros(broj_fold)
AUCBF5_2 = np.zeros(broj_fold)
AUCBF5_3 = np.zeros(broj_fold)
AUCBF7_1 = np.zeros(broj_fold)
AUCBF7_2 = np.zeros(broj_fold)
AUCBF7_3 = np.zeros(broj_fold)
AUCBF9_1 = np.zeros(broj_fold)
AUCBF9_2 = np.zeros(broj_fold)
AUCBF9_3 = np.zeros(broj_fold)
AUCBF2 = np.zeros(broj_fold)
AUCBF21 = np.zeros(broj_fold)
AUCBF22 = np.zeros(broj_fold)
AUCBF3 = np.zeros(broj_fold)
AUCBF4 = np.zeros(broj_fold)
AUCBF41 = np.zeros(broj_fold)
AUCBF42 = np.zeros(broj_fold)
AUCBF5 = np.zeros(broj_fold)
AUCBF6 = np.zeros(broj_fold)
AUCBF61 = np.zeros(broj_fold)
AUCBF62 = np.zeros(broj_fold)
AUCBF7 = np.zeros(broj_fold)
AUCBF8 = np.zeros(broj_fold)
AUCBF81 = np.zeros(broj_fold)
AUCBF82 = np.zeros(broj_fold)

ACCNB = np.zeros(broj_fold)
ACCB = np.zeros(broj_fold)
ACCBF = np.zeros(broj_fold)
ACCBF3_1 = np.zeros(broj_fold)
ACCBF3_2 = np.zeros(broj_fold)
ACCBF3_3 = np.zeros(broj_fold)
ACCBF5_1 = np.zeros(broj_fold)
ACCBF5_2 = np.zeros(broj_fold)
ACCBF5_3 = np.zeros(broj_fold)
ACCBF7_1 = np.zeros(broj_fold)
ACCBF7_2 = np.zeros(broj_fold)
ACCBF7_3 = np.zeros(broj_fold)
ACCBF9_1 = np.zeros(broj_fold)
ACCBF9_2 = np.zeros(broj_fold)
ACCBF9_3 = np.zeros(broj_fold)
ACCBF2 = np.zeros(broj_fold)
ACCBF21 = np.zeros(broj_fold)
ACCBF22 = np.zeros(broj_fold)
ACCBF3 = np.zeros(broj_fold)
ACCBF4 = np.zeros(broj_fold)
ACCBF41 = np.zeros(broj_fold)
ACCBF42 = np.zeros(broj_fold)
ACCBF5 = np.zeros(broj_fold)
ACCBF6 = np.zeros(broj_fold)
ACCBF61 = np.zeros(broj_fold)
ACCBF62 = np.zeros(broj_fold)
ACCBF7 = np.zeros(broj_fold)
ACCBF8 = np.zeros(broj_fold)
ACCBF81 = np.zeros(broj_fold)
ACCBF82 = np.zeros(broj_fold)

HLNB = np.zeros(broj_fold)
HLB = np.zeros(broj_fold)
HLBF = np.zeros(broj_fold)
HLBF2 = np.zeros(broj_fold)
HLBF21 = np.zeros(broj_fold)
HLBF22 = np.zeros(broj_fold)
HLBF3 = np.zeros(broj_fold)
HLBF4 = np.zeros(broj_fold)
HLBF41 = np.zeros(broj_fold)
HLBF42 = np.zeros(broj_fold)
HLBF5 = np.zeros(broj_fold)
HLBF6 = np.zeros(broj_fold)
HLBF61 = np.zeros(broj_fold)
HLBF62 = np.zeros(broj_fold)
HLBF7 = np.zeros(broj_fold)
HLBF8 = np.zeros(broj_fold)
HLBF81 = np.zeros(broj_fold)
HLBF82 = np.zeros(broj_fold)

logProbNB = np.zeros(broj_fold)
logProbB = np.zeros(broj_fold)
logProbBF = np.zeros(broj_fold)
logProbBF2 = np.zeros(broj_fold)
logProbBF21 = np.zeros(broj_fold)
logProbBF22 = np.zeros(broj_fold)
logProbBF3 = np.zeros(broj_fold)
logProbBF4 = np.zeros(broj_fold)
logProbBF41 = np.zeros(broj_fold)
logProbBF42 = np.zeros(broj_fold)
logProbBF5 = np.zeros(broj_fold)
logProbBF6 = np.zeros(broj_fold)
logProbBF61 = np.zeros(broj_fold)
logProbBF62 = np.zeros(broj_fold)
logProbBF7 = np.zeros(broj_fold)
logProbBF8 = np.zeros(broj_fold)
logProbBF81 = np.zeros(broj_fold)
logProbBF82 = np.zeros(broj_fold)

timeNB = np.zeros(broj_fold)
timeB = np.zeros(broj_fold)
timeBF = np.zeros(broj_fold)
timeBF3_1 = np.zeros(broj_fold)
timeBF3_2 = np.zeros(broj_fold)
timeBF3_3 = np.zeros(broj_fold)
timeBF5_1 = np.zeros(broj_fold)
timeBF5_2 = np.zeros(broj_fold)
timeBF5_3 = np.zeros(broj_fold)
timeBF7_1 = np.zeros(broj_fold)
timeBF7_2 = np.zeros(broj_fold)
timeBF7_3 = np.zeros(broj_fold)
timeBF9_1 = np.zeros(broj_fold)
timeBF9_2 = np.zeros(broj_fold)
timeBF9_3 = np.zeros(broj_fold)
timeBF2 = np.zeros(broj_fold)
timeBF21 = np.zeros(broj_fold)
timeBF22 = np.zeros(broj_fold)
timeBF3 = np.zeros(broj_fold)
timeBF4 = np.zeros(broj_fold)
timeBF41 = np.zeros(broj_fold)
timeBF42 = np.zeros(broj_fold)
timeBF5 = np.zeros(broj_fold)
timeBF6 = np.zeros(broj_fold)
timeBF61 = np.zeros(broj_fold)
timeBF62 = np.zeros(broj_fold)
timeBF7 = np.zeros(broj_fold)
timeBF8 = np.zeros(broj_fold)
timeBF81 = np.zeros(broj_fold)
timeBF82 = np.zeros(broj_fold)
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
    mod3_1 = GCRFC_fast()
    mod3_1.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'KMeans', clus_no = 2)    
    probBF3_1, YBF3_1, VarBF3_1 = mod3_1.predict(R_test,Se_test)
    timeBF3_1[i] = time.time() - start_time 
    
    start_time = time.time()
    mod3_2 = GCRFC_fast()
    mod3_2.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'KMeans', clus_no = 3)    
    probBF3_2, YBF3_2, VarBF3_2 = mod3_2.predict(R_test,Se_test)
    timeBF3_2[i] = time.time() - start_time 

    start_time = time.time()
    mod3_3 = GCRFC_fast()
    mod3_3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'KMeans', clus_no = 4)    
    probBF3_3, YBF3_3, VarBF3_3 = mod3_3.predict(R_test,Se_test)
    timeBF3_3[i] = time.time() - start_time 
    
    start_time = time.time()
    mod2 = GCRFC()
    mod2.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija)  
    probB, YB, VarB = mod2.predict(R_test,Se_test)
    timeB[i] = time.time() - start_time 
    
    
    start_time = time.time()
    mod3 = GCRFC_fast()
    mod3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'KMeans', clus_no = 5)  
    probBF, YBF, VarBF = mod3.predict(R_test,Se_test)  
    timeBF[i] = time.time() - start_time
    
    start_time = time.time()
    mod4 = GCRFC_fast()
    mod4.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'KMeans', clus_no = 50)  
    probBF2, YBF2, VarBF2 = mod4.predict(R_test,Se_test)  
    timeBF2[i] = time.time() - start_time
    
    start_time = time.time()
    mod41 = GCRFC_fast()
    mod41.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'KMeans', clus_no = 150)  
    probBF21, YBF21, VarBF21 = mod41.predict(R_test,Se_test)  
    timeBF21[i] = time.time() - start_time
    
    start_time = time.time()
    mod42 = GCRFC_fast()
    mod42.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'KMeans', clus_no = 250)  
    probBF22, YBF22, VarBF22 = mod42.predict(R_test,Se_test)  
    timeBF22[i] = time.time() - start_time
    
    start_time = time.time()
    mod5 = GCRFC_fast()
    mod5.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'MiniBatchKMeans', clus_no = 5)  
    probBF3, YBF3, VarBF3 = mod5.predict(R_test,Se_test)  
    timeBF3[i] = time.time() - start_time
    
    start_time = time.time()
    mod5_1 = GCRFC_fast()
    mod5_1.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'MiniBatchKMeans', clus_no = 2)    
    probBF5_1, YBF5_1, VarBF5_1 = mod5_1.predict(R_test,Se_test)
    timeBF5_1[i] = time.time() - start_time 
    
    start_time = time.time()
    mod5_2 = GCRFC_fast()
    mod5_2.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'MiniBatchKMeans', clus_no = 3)    
    probBF5_2, YBF5_2, VarBF5_2 = mod5_2.predict(R_test,Se_test)
    timeBF5_2[i] = time.time() - start_time 

    start_time = time.time()
    mod5_3 = GCRFC_fast()
    mod5_3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'MiniBatchKMeans', clus_no = 4)    
    probBF5_3, YBF5_3, VarBF5_3 = mod5_3.predict(R_test,Se_test)
    timeBF5_3[i] = time.time() - start_time 
    
    start_time = time.time()
    mod6 = GCRFC_fast()
    mod6.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'MiniBatchKMeans', clus_no = 50)  
    probBF4, YBF4, VarBF4 = mod6.predict(R_test,Se_test)  
    timeBF4[i] = time.time() - start_time
    
    start_time = time.time()
    mod61 = GCRFC_fast()
    mod61.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'MiniBatchKMeans', clus_no = 150)  
    probBF41, YBF41, VarBF41 = mod61.predict(R_test,Se_test)  
    timeBF41[i] = time.time() - start_time
    
    start_time = time.time()
    mod62 = GCRFC_fast()
    mod62.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'MiniBatchKMeans', clus_no = 250)  
    probBF42, YBF42, VarBF42 = mod62.predict(R_test,Se_test)  
    timeBF42[i] = time.time() - start_time
    
    start_time = time.time()
    mod7 = GCRFC_fast()
    mod7.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixture', clus_no = 5)  
    probBF5, YBF5, VarBF5 = mod7.predict(R_test,Se_test)  
    timeBF5[i] = time.time() - start_time
    
    start_time = time.time()
    mod7_1 = GCRFC_fast()
    mod7_1.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'GaussianMixture', clus_no = 2)    
    probBF7_1, YBF7_1, VarBF7_1 = mod7_1.predict(R_test,Se_test)
    timeBF7_1[i] = time.time() - start_time 
    
    start_time = time.time()
    mod7_2 = GCRFC_fast()
    mod7_2.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'GaussianMixture', clus_no = 3)    
    probBF7_2, YBF7_2, VarBF7_2 = mod7_2.predict(R_test,Se_test)
    timeBF7_2[i] = time.time() - start_time 

    start_time = time.time()
    mod7_3 = GCRFC_fast()
    mod7_3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'GaussianMixture', clus_no = 4)    
    probBF7_3, YBF7_3, VarBF7_3 = mod7_3.predict(R_test,Se_test)
    timeBF7_3[i] = time.time() - start_time 
    
    start_time = time.time()
    mod8 = GCRFC_fast()
    mod8.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixture', clus_no = 50)  
    probBF6, YBF6, VarBF6 = mod8.predict(R_test,Se_test)  
    timeBF6[i] = time.time() - start_time
    
    start_time = time.time()
    mod81 = GCRFC_fast()
    mod81.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixture', clus_no = 150)  
    probBF61, YBF61, VarBF61 = mod81.predict(R_test,Se_test)  
    timeBF61[i] = time.time() - start_time
    
    start_time = time.time()
    mod82 = GCRFC_fast()
    mod82.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixture', clus_no = 250)  
    probBF62, YBF62, VarBF62 = mod82.predict(R_test,Se_test)  
    timeBF62[i] = time.time() - start_time
    
    start_time = time.time()
    mod9 = GCRFC_fast()
    mod9.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixtureProb', clus_no = 5)  
    probBF7, YBF7, VarBF7 = mod9.predict(R_test,Se_test)  
    timeBF7[i] = time.time() - start_time
    
    start_time = time.time()
    mod9_1 = GCRFC_fast()
    mod9_1.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'GaussianMixtureProb', clus_no = 2)    
    probBF9_1, YBF9_1, VarBF9_1 = mod9_1.predict(R_test,Se_test)
    timeBF9_1[i] = time.time() - start_time 
    
    start_time = time.time()
    mod9_2 = GCRFC_fast()
    mod9_2.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'GaussianMixtureProb', clus_no = 3)    
    probBF9_2, YBF9_2, VarBF9_2 = mod9_2.predict(R_test,Se_test)
    timeBF9_2[i] = time.time() - start_time 

    start_time = time.time()
    mod9_3 = GCRFC_fast()
    mod9_3.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija, method_clus = 'GaussianMixtureProb', clus_no = 4)    
    probBF9_3, YBF9_3, VarBF9_3 = mod9_3.predict(R_test,Se_test)
    timeBF9_3[i] = time.time() - start_time 
    
    start_time = time.time()
    mod10 = GCRFC_fast()
    mod10.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixtureProb', clus_no = 50)  
    probBF8, YBF8, VarBF8 = mod10.predict(R_test,Se_test)  
    timeBF8[i] = time.time() - start_time
    
    start_time = time.time()
    mod101 = GCRFC_fast()
    mod101.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixtureProb', clus_no = 150)  
    probBF81, YBF81, VarBF81 = mod101.predict(R_test,Se_test)  
    timeBF81[i] = time.time() - start_time
    
    start_time = time.time()
    mod102 = GCRFC_fast()
    mod102.fit(R_train, Se_train, Y_train, learn = 'TNC', learnrate = 3e-4, learnratec = 0.5, maxiter = iteracija,method_clus = 'GaussianMixtureProb', clus_no = 250)  
    probBF82, YBF82, VarBF82 = mod102.predict(R_test,Se_test)  
    timeBF82[i] = time.time() - start_time
    
    HLNB[i] = hamming_loss(Y_test,YNB)
    HLB[i] = hamming_loss(Y_test,YB)
    HLBF[i] = hamming_loss(Y_test,YBF)
    HLBF2[i] = hamming_loss(Y_test,YBF2)
    HLBF21[i] = hamming_loss(Y_test,YBF21)
    HLBF22[i] = hamming_loss(Y_test,YBF22)
    HLBF3[i] = hamming_loss(Y_test,YBF3)
    HLBF4[i] = hamming_loss(Y_test,YBF4)
    HLBF41[i] = hamming_loss(Y_test,YBF41)
    HLBF42[i] = hamming_loss(Y_test,YBF42)
    HLBF5[i] = hamming_loss(Y_test,YBF5)
    HLBF6[i] = hamming_loss(Y_test,YBF6)
    HLBF61[i] = hamming_loss(Y_test,YBF61)
    HLBF62[i] = hamming_loss(Y_test,YBF62)
    HLBF7[i] = hamming_loss(Y_test,YBF7)
    HLBF8[i] = hamming_loss(Y_test,YBF8)
    HLBF81[i] = hamming_loss(Y_test,YBF81)
    HLBF82[i] = hamming_loss(Y_test,YBF82) 
       
    Y_test = Y_test.reshape([Y_test.shape[0]*Y_test.shape[1]])
    
    YNB =  YNB.reshape([YNB.shape[0]*YNB.shape[1]])
    probNB = probNB.reshape([probNB.shape[0]*probNB.shape[1]])
    YB =  YB.reshape([YB.shape[0]*YB.shape[1]])
    probB = probB.reshape([probB.shape[0]*probB.shape[1]])
    YBF =  YBF.reshape([YBF.shape[0]*YBF.shape[1]])
    probBF = probBF.reshape([probBF.shape[0]*probBF.shape[1]])
    YBF2 =  YBF2.reshape([YBF2.shape[0]*YBF2.shape[1]])
    probBF2 = probBF2.reshape([probBF2.shape[0]*probBF2.shape[1]])
    YBF21 =  YBF21.reshape([YBF21.shape[0]*YBF21.shape[1]])
    probBF21 = probBF21.reshape([probBF21.shape[0]*probBF21.shape[1]])
    YBF22 =  YBF22.reshape([YBF22.shape[0]*YBF22.shape[1]])
    probBF22 = probBF22.reshape([probBF22.shape[0]*probBF22.shape[1]])
    YBF3 =  YBF3.reshape([YBF3.shape[0]*YBF3.shape[1]])
    probBF3 = probBF3.reshape([probBF3.shape[0]*probBF3.shape[1]])    
    YBF3_1 =  YBF3_1.reshape([YBF3_1.shape[0]*YBF3_1.shape[1]])
    probBF3_1 = probBF3_1.reshape([probBF3_1.shape[0]*probBF3_1.shape[1]])
    YBF3_2 =  YBF3_2.reshape([YBF3_2.shape[0]*YBF3_2.shape[1]])
    probBF3_2 = probBF3_2.reshape([probBF3_2.shape[0]*probBF3_2.shape[1]])
    YBF3_3 =  YBF3_3.reshape([YBF3_3.shape[0]*YBF3_3.shape[1]])
    probBF3_3 = probBF3_3.reshape([probBF3_3.shape[0]*probBF3_3.shape[1]])
    YBF4 =  YBF4.reshape([YBF4.shape[0]*YBF4.shape[1]])
    probBF4 = probBF4.reshape([probBF4.shape[0]*probBF4.shape[1]])
    YBF41 =  YBF41.reshape([YBF41.shape[0]*YBF41.shape[1]])
    probBF41 = probBF41.reshape([probBF41.shape[0]*probBF41.shape[1]])
    YBF42 =  YBF42.reshape([YBF42.shape[0]*YBF42.shape[1]])
    probBF42 = probBF42.reshape([probBF42.shape[0]*probBF42.shape[1]])
    YBF5 =  YBF5.reshape([YBF5.shape[0]*YBF5.shape[1]])
    YBF5_1 =  YBF5_1.reshape([YBF5_1.shape[0]*YBF5_1.shape[1]])
    probBF5_1 = probBF5_1.reshape([probBF5_1.shape[0]*probBF5_1.shape[1]])
    YBF5_2 =  YBF5_2.reshape([YBF5_2.shape[0]*YBF5_2.shape[1]])
    probBF5_2 = probBF5_2.reshape([probBF5_2.shape[0]*probBF5_2.shape[1]])
    YBF5_3 =  YBF5_3.reshape([YBF5_3.shape[0]*YBF5_3.shape[1]])
    probBF5_3 = probBF5_3.reshape([probBF5_3.shape[0]*probBF5_3.shape[1]])
    probBF5 = probBF5.reshape([probBF5.shape[0]*probBF5.shape[1]])
    YBF6 =  YBF6.reshape([YBF6.shape[0]*YBF6.shape[1]])
    probBF6 = probBF6.reshape([probBF6.shape[0]*probBF6.shape[1]])
    YBF61 =  YBF61.reshape([YBF61.shape[0]*YBF61.shape[1]])
    probBF61 = probBF61.reshape([probBF61.shape[0]*probBF61.shape[1]])
    YBF62 =  YBF62.reshape([YBF62.shape[0]*YBF62.shape[1]])
    probBF62 = probBF62.reshape([probBF62.shape[0]*probBF62.shape[1]])
    YBF7 =  YBF7.reshape([YBF7.shape[0]*YBF7.shape[1]])
    probBF7 = probBF7.reshape([probBF7.shape[0]*probBF7.shape[1]])
    YBF7_1 =  YBF7_1.reshape([YBF7_1.shape[0]*YBF7_1.shape[1]])
    probBF7_1 = probBF7_1.reshape([probBF7_1.shape[0]*probBF7_1.shape[1]])
    YBF7_2 =  YBF7_2.reshape([YBF7_2.shape[0]*YBF7_2.shape[1]])
    probBF7_2 = probBF7_2.reshape([probBF7_2.shape[0]*probBF7_2.shape[1]])
    YBF7_3 =  YBF7_3.reshape([YBF7_3.shape[0]*YBF7_3.shape[1]])
    probBF7_3 = probBF7_3.reshape([probBF7_3.shape[0]*probBF7_3.shape[1]])
    YBF8 =  YBF8.reshape([YBF8.shape[0]*YBF8.shape[1]])
    probBF8 = probBF8.reshape([probBF8.shape[0]*probBF8.shape[1]])
    YBF81 =  YBF81.reshape([YBF81.shape[0]*YBF81.shape[1]])
    probBF81 = probBF81.reshape([probBF81.shape[0]*probBF81.shape[1]])
    YBF82 =  YBF82.reshape([YBF82.shape[0]*YBF82.shape[1]])
    probBF82 = probBF82.reshape([probBF82.shape[0]*probBF82.shape[1]])
    YBF9_1 =  YBF9_1.reshape([YBF9_1.shape[0]*YBF9_1.shape[1]])
    probBF9_1 = probBF9_1.reshape([probBF9_1.shape[0]*probBF9_1.shape[1]])
    YBF9_2 =  YBF9_2.reshape([YBF9_2.shape[0]*YBF9_2.shape[1]])
    probBF9_2 = probBF9_2.reshape([probBF9_2.shape[0]*probBF9_2.shape[1]])
    YBF9_3 =  YBF9_3.reshape([YBF9_3.shape[0]*YBF9_3.shape[1]])
    probBF9_3 = probBF9_3.reshape([probBF9_3.shape[0]*probBF9_3.shape[1]])
    
    AUCNB[i] = roc_auc_score(Y_test,probNB)
    AUCB[i] = roc_auc_score(Y_test,probB)
    AUCBF[i] = roc_auc_score(Y_test,probBF)
    AUCBF2[i] = roc_auc_score(Y_test,probBF2)
    AUCBF21[i] = roc_auc_score(Y_test,probBF21)
    AUCBF22[i] = roc_auc_score(Y_test,probBF22)
    AUCBF3[i] = roc_auc_score(Y_test,probBF3)
    AUCBF3_1[i] = roc_auc_score(Y_test,probBF3_1)
    AUCBF3_2[i] = roc_auc_score(Y_test,probBF3_2)
    AUCBF3_3[i] = roc_auc_score(Y_test,probBF3_3)
    AUCBF4[i] = roc_auc_score(Y_test,probBF4)
    AUCBF41[i] = roc_auc_score(Y_test,probBF41)
    AUCBF42[i] = roc_auc_score(Y_test,probBF42)
    AUCBF5[i] = roc_auc_score(Y_test,probBF5)
    AUCBF5_1[i] = roc_auc_score(Y_test,probBF5_1)
    AUCBF5_2[i] = roc_auc_score(Y_test,probBF5_2)
    AUCBF5_3[i] = roc_auc_score(Y_test,probBF5_3)
    AUCBF6[i] = roc_auc_score(Y_test,probBF6)
    AUCBF61[i] = roc_auc_score(Y_test,probBF61)
    AUCBF62[i] = roc_auc_score(Y_test,probBF62)
    AUCBF7[i] = roc_auc_score(Y_test,probBF7)
    AUCBF7_1[i] = roc_auc_score(Y_test,probBF7_1)
    AUCBF7_2[i] = roc_auc_score(Y_test,probBF7_2)
    AUCBF7_3[i] = roc_auc_score(Y_test,probBF7_3)
    AUCBF8[i] = roc_auc_score(Y_test,probBF8)
    AUCBF81[i] = roc_auc_score(Y_test,probBF81)
    AUCBF82[i] = roc_auc_score(Y_test,probBF82)
    AUCBF7_1[i] = roc_auc_score(Y_test,probBF9_1)
    AUCBF7_2[i] = roc_auc_score(Y_test,probBF9_2)
    AUCBF7_3[i] = roc_auc_score(Y_test,probBF9_3)
    
    ACCNB[i] = accuracy_score(Y_test,YNB)
    ACCB[i] = accuracy_score(Y_test,YB)
    ACCBF[i] = accuracy_score(Y_test,YBF)
    ACCBF2[i] = accuracy_score(Y_test,YBF2)
    ACCBF21[i] = accuracy_score(Y_test,YBF21)
    ACCBF22[i] = accuracy_score(Y_test,YBF22)
    ACCBF3[i] = accuracy_score(Y_test,YBF3)
    ACCBF3_1[i] = accuracy_score(Y_test,YBF3_1)
    ACCBF3_2[i] = accuracy_score(Y_test,YBF3_2)
    ACCBF3_3[i] = accuracy_score(Y_test,YBF3_3)
    ACCBF4[i] = accuracy_score(Y_test,YBF4)
    ACCBF41[i] = accuracy_score(Y_test,YBF41)
    ACCBF42[i] = accuracy_score(Y_test,YBF42)
    ACCBF5[i] = accuracy_score(Y_test,YBF5)
    ACCBF5_1[i] = accuracy_score(Y_test,YBF5_1)
    ACCBF5_2[i] = accuracy_score(Y_test,YBF5_2)
    ACCBF5_3[i] = accuracy_score(Y_test,YBF5_3)
    ACCBF6[i] = accuracy_score(Y_test,YBF6)
    ACCBF61[i] = accuracy_score(Y_test,YBF61)
    ACCBF62[i] = accuracy_score(Y_test,YBF62)
    ACCBF7[i] = accuracy_score(Y_test,YBF7)
    ACCBF7_1[i] = accuracy_score(Y_test,YBF7_1)
    ACCBF7_2[i] = accuracy_score(Y_test,YBF7_2)
    ACCBF7_3[i] = accuracy_score(Y_test,YBF7_3)
    ACCBF8[i] = accuracy_score(Y_test,YBF8)
    ACCBF81[i] = accuracy_score(Y_test,YBF81)
    ACCBF82[i] = accuracy_score(Y_test,YBF82)
    ACCBF9_1[i] = accuracy_score(Y_test,YBF9_1)
    ACCBF9_2[i] = accuracy_score(Y_test,YBF9_2)
    ACCBF9_3[i] = accuracy_score(Y_test,YBF9_3)
    
    probNB[Y_test==0] = 1 - probNB[Y_test==0]
    probB[Y_test==0] = 1 - probNB[Y_test==0]
    probBF[Y_test==0] = 1 - probBF[Y_test==0]
    probBF2[Y_test==0] = 1 - probBF2[Y_test==0]
    probBF21[Y_test==0] = 1 - probBF21[Y_test==0]
    probBF22[Y_test==0] = 1 - probBF22[Y_test==0]
    probBF3[Y_test==0] = 1 - probBF3[Y_test==0]
    probBF3_1[Y_test==0] = 1 - probBF3_1[Y_test==0]
    probBF3_2[Y_test==0] = 1 - probBF3_2[Y_test==0]
    probBF3_3[Y_test==0] = 1 - probBF3_3[Y_test==0]
    probBF4[Y_test==0] = 1 - probBF4[Y_test==0]
    probBF41[Y_test==0] = 1 - probBF41[Y_test==0]
    probBF42[Y_test==0] = 1 - probBF42[Y_test==0]
    probBF5[Y_test==0] = 1 - probBF5[Y_test==0]
    probBF5_1[Y_test==0] = 1 - probBF5_1[Y_test==0]
    probBF5_2[Y_test==0] = 1 - probBF5_2[Y_test==0]
    probBF5_3[Y_test==0] = 1 - probBF5_3[Y_test==0]
    probBF6[Y_test==0] = 1 - probBF6[Y_test==0]
    probBF61[Y_test==0] = 1 - probBF61[Y_test==0]
    probBF62[Y_test==0] = 1 - probBF62[Y_test==0]
    probBF7[Y_test==0] = 1 - probBF7[Y_test==0]
    probBF8[Y_test==0] = 1 - probBF8[Y_test==0]
    probBF81[Y_test==0] = 1 - probBF81[Y_test==0]
    probBF82[Y_test==0] = 1 - probBF82[Y_test==0]
    probBF9_1[Y_test==0] = 1 - probBF9_1[Y_test==0]
    probBF9_2[Y_test==0] = 1 - probBF9_2[Y_test==0]
    probBF9_3[Y_test==0] = 1 - probBF9_3[Y_test==0]
    
    logProbNB[i] = np.sum(np.log(probNB))
    logProbB[i] = np.sum(np.log(probB))
    logProbBF[i] = np.sum(np.log(probBF))
    logProbBF2[i] = np.sum(np.log(probBF2))
    logProbBF21[i] = np.sum(np.log(probBF21))
    logProbBF22[i] = np.sum(np.log(probBF22))
    logProbBF3[i] = np.sum(np.log(probBF3))
    logProbBF4[i] = np.sum(np.log(probBF4))
    logProbBF41[i] = np.sum(np.log(probBF41))
    logProbBF42[i] = np.sum(np.log(probBF42))
    logProbBF5[i] = np.sum(np.log(probBF5))
    logProbBF6[i] = np.sum(np.log(probBF6))
    logProbBF61[i] = np.sum(np.log(probBF61))
    logProbBF62[i] = np.sum(np.log(probBF62))
    logProbBF7[i] = np.sum(np.log(probBF7))
    logProbBF8[i] = np.sum(np.log(probBF8))
    logProbBF81[i] = np.sum(np.log(probBF81))
    logProbBF82[i] = np.sum(np.log(probBF82))
    
    file.write('AUC GCRFCNB prediktora je {}'.format(AUCNB[i]) + "\n")
    file.write('AUC GCRFCB prediktora je {}'.format(AUCB[i]) + "\n")
    file.write('AUC GCRFCB_fast prediktora je {}'.format(AUCBF[i]) + "\n")
    file.write('AUC GCRFCB2_fast prediktora je {}'.format(AUCBF2[i]) + "\n")
    file.write('AUC GCRFCB21_fast prediktora je {}'.format(AUCBF21[i]) + "\n")
    file.write('AUC GCRFCB22_fast prediktora je {}'.format(AUCBF22[i]) + "\n")
    file.write('AUC GCRFCB3_fast prediktora je {}'.format(AUCBF3[i]) + "\n")
    file.write('AUC GCRFCB4_fast prediktora je {}'.format(AUCBF4[i]) + "\n")
    file.write('AUC GCRFCB41_fast prediktora je {}'.format(AUCBF41[i]) + "\n")
    file.write('AUC GCRFCB42_fast prediktora je {}'.format(AUCBF42[i]) + "\n")
    file.write('AUC GCRFCB5_fast prediktora je {}'.format(AUCBF5[i]) + "\n")
    file.write('AUC GCRFCB6_fast prediktora je {}'.format(AUCBF6[i]) + "\n")
    file.write('AUC GCRFCB61_fast prediktora je {}'.format(AUCBF61[i]) + "\n")
    file.write('AUC GCRFCB62_fast prediktora je {}'.format(AUCBF62[i]) + "\n")
    file.write('AUC GCRFCB7_fast prediktora je {}'.format(AUCBF7[i]) + "\n")
    file.write('AUC GCRFCB8_fast prediktora je {}'.format(AUCBF8[i]) + "\n")
    file.write('AUC GCRFCB41_fast prediktora je {}'.format(AUCBF81[i]))
    file.write('AUC GCRFCB42_fast prediktora je {}'.format(AUCBF82[i]) + "\n")

    file.write('ACC GCRFCNB prediktora je {}'.format(ACCNB[i]) + "\n")
    file.write('ACC GCRFCB prediktora je {}'.format(ACCB[i]) + "\n")
    file.write('ACC GCRFCB_fast prediktora je {}'.format(ACCBF[i]) + "\n")
    file.write('ACC GCRFCB2_fast prediktora je {}'.format(ACCBF2[i]) + "\n")
    file.write('ACC GCRFCB21_fast prediktora je {}'.format(ACCBF21[i]) + "\n")
    file.write('ACC GCRFCB22_fast prediktora je {}'.format(ACCBF22[i]) + "\n")
    file.write('ACC GCRFCB3_fast prediktora je {}'.format(ACCBF3[i]) + "\n")
    file.write('ACC GCRFCB4_fast prediktora je {}'.format(ACCBF4[i]) + "\n")
    file.write('ACC GCRFCB41_fast prediktora je {}'.format(ACCBF41[i]) + "\n")
    file.write('ACC GCRFCB42_fast prediktora je {}'.format(ACCBF42[i]) + "\n")
    file.write('ACC GCRFCB5_fast prediktora je {}'.format(ACCBF5[i]) + "\n")
    file.write('ACC GCRFCB6_fast prediktora je {}'.format(ACCBF6[i]) + "\n")
    file.write('ACC GCRFCB61_fast prediktora je {}'.format(ACCBF61[i]) + "\n")
    file.write('ACC GCRFCB62_fast prediktora je {}'.format(ACCBF62[i]) + "\n")
    file.write('ACC GCRFCB7_fast prediktora je {}'.format(ACCBF7[i]) + "\n")
    file.write('ACC GCRFCB8_fast prediktora je {}'.format(ACCBF8[i]) + "\n")
    file.write('ACC GCRFCB41_fast prediktora je {}'.format(ACCBF81[i]))
    file.write('ACC GCRFCB42_fast prediktora je {}'.format(ACCBF82[i]) + "\n")
    
    file.write('HL GCRFCNB prediktora je {}'.format(HLNB[i]) + "\n")
    file.write('HL GCRFCB prediktora je {}'.format(HLB[i]) + "\n")
    file.write('HL GCRFCB_fast prediktora je {}'.format(HLBF[i]) + "\n")
    file.write('HL GCRFCB2_fast prediktora je {}'.format(HLBF2[i]) + "\n")
    file.write('HL GCRFCB21_fast prediktora je {}'.format(HLBF21[i]) + "\n")
    file.write('HL GCRFCB22_fast prediktora je {}'.format(HLBF22[i]) + "\n")
    file.write('HL GCRFCB3_fast prediktora je {}'.format(HLBF3[i]) + "\n")
    file.write('HL GCRFCB4_fast prediktora je {}'.format(HLBF4[i]) + "\n")
    file.write('HL GCRFCB41_fast prediktora je {}'.format(HLBF41[i]) + "\n")
    file.write('HL GCRFCB42_fast prediktora je {}'.format(HLBF42[i]) + "\n")
    file.write('HL GCRFCB5_fast prediktora je {}'.format(HLBF5[i]) + "\n")
    file.write('HL GCRFCB6_fast prediktora je {}'.format(HLBF6[i]) + "\n")
    file.write('HL GCRFCB61_fast prediktora je {}'.format(HLBF61[i]) + "\n")
    file.write('HL GCRFCB62_fast prediktora je {}'.format(HLBF62[i]) + "\n")
    file.write('HL GCRFCB7_fast prediktora je {}'.format(HLBF7[i]) + "\n")
    file.write('HL GCRFCB8_fast prediktora je {}'.format(HLBF8[i]) + "\n")
    file.write('HL GCRFCB41_fast prediktora je {}'.format(HLBF81[i]))
    file.write('HL GCRFCB42_fast prediktora je {}'.format(HLBF82[i]) + "\n")
    
    file.write('AUC nestruktuiranih prediktora je {}'.format(Skor_com_AUC[i,:]) + "\n")
    file.write('AUC2 nestruktuiranih prediktora je {}'.format(Skor_com_AUC2[i,:]) + "\n")
    
    file.write('ACC nestruktuiranih prediktora je {}'.format(Skor_com_ACC[i,:]) + "\n")
    file.write('ACC2 nestruktuiranih prediktora je {}'.format(Skor_com_ACC2[i,:]) + "\n")
    
    file.write('HL nestruktuiranih prediktora je {}'.format(Skor_com_HL[i,:]) + "\n")
    
    file.write('Logprob GCRFCNB je {}'.format(logProbNB[i]) + "\n")
    file.write('Logprob GCRFCB je {}'.format(logProbB[i]) + "\n")
    file.write('Logprob GCRFCB_fast je {}'.format(logProbBF[i]) + "\n")
    file.write('Logprob GCRFCB2_fast je {}'.format(logProbBF2[i]) + "\n")
    file.write('Logprob GCRFCB21_fast je {}'.format(logProbBF21[i]) + "\n")
    file.write('Logprob GCRFCB22_fast je {}'.format(logProbBF22[i]) + "\n")
    file.write('Logprob GCRFCB3_fast je {}'.format(logProbBF3[i]) + "\n")
    file.write('Logprob GCRFCB4_fast je {}'.format(logProbBF4[i]) + "\n")
    file.write('Logprob GCRFCB41_fast je {}'.format(logProbBF41[i]) + "\n")
    file.write('Logprob GCRFCB42_fast je {}'.format(logProbBF42[i]) + "\n")
    file.write('Logprob GCRFCB5_fast je {}'.format(logProbBF5[i]) + "\n")
    file.write('Logprob GCRFCB6_fast je {}'.format(logProbBF6[i]) + "\n")
    file.write('Logprob GCRFCB61_fast je {}'.format(logProbBF61[i]) + "\n")
    file.write('Logprob GCRFCB62_fast je {}'.format(logProbBF62[i]) + "\n")
    file.write('Logprob GCRFCB7_fast je {}'.format(logProbBF7[i]) + "\n")
    file.write('Logprob GCRFCB8_fast je {}'.format(logProbBF8[i]) + "\n")
    file.write('Logprob GCRFCB81_fast je {}'.format(logProbBF81[i]) + "\n")
    file.write('Logprob GCRFCB82_fast je {}'.format(logProbBF82[i]) + "\n")
    
    file.write("--- %s seconds --- GCRFCNB" % (timeNB[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB" % (timeB[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB_fast" % (timeBF[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB2_fast" % (timeBF2[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB21_fast" % (timeBF21[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB22_fast" % (timeBF22[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB3_fast" % (timeBF3[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB4_fast" % (timeBF4[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB41_fast" % (timeBF41[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB42_fast" % (timeBF42[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB5_fast" % (timeBF5[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB6_fast" % (timeBF6[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB61_fast" % (timeBF61[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB62_fast" % (timeBF62[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB7_fast" % (timeBF7[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB8_fast" % (timeBF8[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB61_fast" % (timeBF81[i]) + "\n")
    file.write("--- %s seconds --- GCRFCB62_fast" % (timeBF82[i]) + "\n")
    
    i= i + 1

file.write('CROSS AUC GCRFCNB prediktora je {}'.format(np.mean(AUCNB)) + "\n")
file.write('CROSS AUC GCRFCB prediktora je {}'.format(np.mean(AUCB)) + "\n")
file.write('CROSS AUC GCRFCB_fast prediktora je {}'.format(np.mean(AUCBF)) + "\n")
file.write('CROSS AUC GCRFCB2_fast prediktora je {}'.format(np.mean(AUCBF2)) + "\n")
file.write('CROSS AUC GCRFCB21_fast prediktora je {}'.format(np.mean(AUCBF21)) + "\n")
file.write('CROSS AUC GCRFCB22_fast prediktora je {}'.format(np.mean(AUCBF22)) + "\n")
file.write('CROSS AUC GCRFCB3fast prediktora je {}'.format(np.mean(AUCBF3)) + "\n")
file.write('CROSS AUC GCRFCB3_1fast prediktora je {}'.format(np.mean(AUCBF3_1)) + "\n")
file.write('CROSS AUC GCRFCB3_2fast prediktora je {}'.format(np.mean(AUCBF3_2)) + "\n")
file.write('CROSS AUC GCRFCB3_3fast prediktora je {}'.format(np.mean(AUCBF3_3)) + "\n")
file.write('CROSS AUC GCRFCB5_1fast prediktora je {}'.format(np.mean(AUCBF5_1)) + "\n")
file.write('CROSS AUC GCRFCB5_2fast prediktora je {}'.format(np.mean(AUCBF5_2)) + "\n")
file.write('CROSS AUC GCRFCB5_3fast prediktora je {}'.format(np.mean(AUCBF5_3)) + "\n")
file.write('CROSS AUC GCRFCB7_1fast prediktora je {}'.format(np.mean(AUCBF7_1)) + "\n")
file.write('CROSS AUC GCRFCB7_2fast prediktora je {}'.format(np.mean(AUCBF7_2)) + "\n")
file.write('CROSS AUC GCRFCB7_3fast prediktora je {}'.format(np.mean(AUCBF7_3)) + "\n")
file.write('CROSS AUC GCRFCB9_1fast prediktora je {}'.format(np.mean(AUCBF9_1)) + "\n")
file.write('CROSS AUC GCRFCB9_2fast prediktora je {}'.format(np.mean(AUCBF9_2)) + "\n")
file.write('CROSS AUC GCRFCB9_3fast prediktora je {}'.format(np.mean(AUCBF9_3)) + "\n")
file.write('CROSS AUC GCRFCB4_fast prediktora je {}'.format(np.mean(AUCBF4)) + "\n")
file.write('CROSS AUC GCRFCB41_fast prediktora je {}'.format(np.mean(AUCBF41)) + "\n")
file.write('CROSS AUC GCRFCB42_fast prediktora je {}'.format(np.mean(AUCBF42)) + "\n")
file.write('CROSS AUC GCRFCB5_fast prediktora je {}'.format(np.mean(AUCBF5)) + "\n")
file.write('CROSS AUC GCRFCB6_fast prediktora je {}'.format(np.mean(AUCBF6)) + "\n")
file.write('CROSS AUC GCRFCB61_fast prediktora je {}'.format(np.mean(AUCBF61)) + "\n")
file.write('CROSS AUC GCRFCB62_fast prediktora je {}'.format(np.mean(AUCBF62)) + "\n")
file.write('CROSS AUC GCRFCB7_fast prediktora je {}'.format(np.mean(AUCBF7)) + "\n")
file.write('CROSS AUC GCRFCB8_fast prediktora je {}'.format(np.mean(AUCBF8)) + "\n")
file.write('CROSS AUC GCRFCB81_fast prediktora je {}'.format(np.mean(AUCBF81)) + "\n")
file.write('CROSS AUC GCRFCB82_fast prediktora je {}'.format(np.mean(AUCBF82)) + "\n")

file.write('CROSS ACC GCRFCNB prediktora je {}'.format(np.mean(ACCNB)) + "\n")
file.write('CROSS ACC GCRFCB prediktora je {}'.format(np.mean(ACCB)) + "\n")
file.write('CROSS ACC GCRFCB_fast prediktora je {}'.format(np.mean(ACCBF)) + "\n")
file.write('CROSS ACC GCRFCB2_fast prediktora je {}'.format(np.mean(ACCBF2)) + "\n")
file.write('CROSS ACC GCRFCB21_fast prediktora je {}'.format(np.mean(ACCBF21)) + "\n")
file.write('CROSS ACC GCRFCB22_fast prediktora je {}'.format(np.mean(ACCBF22)) + "\n")
file.write('CROSS ACC GCRFCB3_fast prediktora je {}'.format(np.mean(ACCBF3)) + "\n")
file.write('CROSS ACC GCRFCB3_1fast prediktora je {}'.format(np.mean(ACCBF3_1)) + "\n")
file.write('CROSS ACC GCRFCB3_2fast prediktora je {}'.format(np.mean(ACCBF3_2)) + "\n")
file.write('CROSS ACC GCRFCB3_3fast prediktora je {}'.format(np.mean(ACCBF3_3)) + "\n")
file.write('CROSS ACC GCRFCB5_1fast prediktora je {}'.format(np.mean(ACCBF5_1)) + "\n")
file.write('CROSS ACC GCRFCB5_2fast prediktora je {}'.format(np.mean(ACCBF5_2)) + "\n")
file.write('CROSS ACC GCRFCB5_3fast prediktora je {}'.format(np.mean(ACCBF5_3)) + "\n")
file.write('CROSS ACC GCRFCB7_1fast prediktora je {}'.format(np.mean(ACCBF7_1)) + "\n")
file.write('CROSS ACC GCRFCB7_2fast prediktora je {}'.format(np.mean(ACCBF7_2)) + "\n")
file.write('CROSS ACC GCRFCB7_3fast prediktora je {}'.format(np.mean(ACCBF7_3)) + "\n")
file.write('CROSS ACC GCRFCB9_1fast prediktora je {}'.format(np.mean(ACCBF9_1)) + "\n")
file.write('CROSS ACC GCRFCB9_2fast prediktora je {}'.format(np.mean(ACCBF9_2)) + "\n")
file.write('CROSS ACC GCRFCB9_3fast prediktora je {}'.format(np.mean(ACCBF9_3)) + "\n")
file.write('CROSS ACC GCRFCB4_fast prediktora je {}'.format(np.mean(ACCBF4)) + "\n")
file.write('CROSS ACC GCRFCB41_fast prediktora je {}'.format(np.mean(ACCBF41)) + "\n")
file.write('CROSS ACC GCRFCB42_fast prediktora je {}'.format(np.mean(ACCBF42)) + "\n")
file.write('CROSS ACC GCRFCB5_fast prediktora je {}'.format(np.mean(ACCBF5)) + "\n")
file.write('CROSS ACC GCRFCB6_fast prediktora je {}'.format(np.mean(ACCBF6)) + "\n")
file.write('CROSS ACC GCRFCB61_fast prediktora je {}'.format(np.mean(ACCBF61)) + "\n")
file.write('CROSS ACC GCRFCB62_fast prediktora je {}'.format(np.mean(ACCBF62)) + "\n")
file.write('CROSS ACC GCRFCB7_fast prediktora je {}'.format(np.mean(ACCBF7)) + "\n")
file.write('CROSS ACC GCRFCB8_fast prediktora je {}'.format(np.mean(ACCBF8)) + "\n")
file.write('CROSS ACC GCRFCB81_fast prediktora je {}'.format(np.mean(ACCBF81)) + "\n")
file.write('CROSS ACC GCRFCB82_fast prediktora je {}'.format(np.mean(ACCBF82)) + "\n")

file.write('CROSS HL GCRFCNB prediktora je {}'.format(np.mean(HLNB)) + "\n")
file.write('CROSS HL GCRFCB prediktora je {}'.format(np.mean(HLB)) + "\n")
file.write('CROSS HL GCRFCB_fast prediktora je {}'.format(np.mean(HLBF)) + "\n")
file.write('CROSS HL GCRFCB2_fast prediktora je {}'.format(np.mean(HLBF2)) + "\n")
file.write('CROSS HL GCRFCB21_fast prediktora je {}'.format(np.mean(HLBF21)) + "\n")
file.write('CROSS HL GCRFCB22_fast prediktora je {}'.format(np.mean(HLBF22)) + "\n")
file.write('CROSS HL GCRFCB3_fast prediktora je {}'.format(np.mean(HLBF3)) + "\n")
file.write('CROSS HL GCRFCB4_fast prediktora je {}'.format(np.mean(HLBF4)) + "\n")
file.write('CROSS HL GCRFCB41_fast prediktora je {}'.format(np.mean(HLBF41)) + "\n")
file.write('CROSS HL GCRFCB42_fast prediktora je {}'.format(np.mean(HLBF42)) + "\n")
file.write('CROSS HL GCRFCB5_fast prediktora je {}'.format(np.mean(HLBF5)) + "\n")
file.write('CROSS HL GCRFCB6_fast prediktora je {}'.format(np.mean(HLBF6)) + "\n")
file.write('CROSS HL GCRFCB61_fast prediktora je {}'.format(np.mean(HLBF61)) + "\n")
file.write('CROSS HL GCRFCB62_fast prediktora je {}'.format(np.mean(HLBF62)) + "\n")
file.write('CROSS HL GCRFCB7_fast prediktora je {}'.format(np.mean(HLBF7)) + "\n")
file.write('CROSS HL GCRFCB8_fast prediktora je {}'.format(np.mean(HLBF8)) + "\n")
file.write('CROSS HL GCRFCB81_fast prediktora je {}'.format(np.mean(HLBF81)) + "\n")
file.write('CROSS HL GCRFCB82_fast prediktora je {}'.format(np.mean(HLBF82)) + "\n")

file.write('CROSS ACC nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_ACC,axis=0)) + "\n")
file.write('CROSS ACC2 nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_ACC2,axis=0)) + "\n")
file.write('CROSS ACC strukturnih prediktora je {}'.format(np.mean(ACC_ST,axis=0)) + "\n")


file.write('CROSS AUC nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_AUC,axis=0)) + "\n")
file.write('CROSS AUC2 nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_AUC2,axis=0)) + "\n")

file.write('CROSS HL nestruktuiranih prediktora je {}'.format(np.mean(Skor_com_HL,axis=0)) + "\n")
file.write('CROSS HL strukturnih prediktora je {}'.format(np.mean(HL_ST,axis=0)) + "\n")


file.write('CROSS Logprob GCRFCNB je {}'.format(np.mean(logProbNB)) + "\n")
file.write('CROSS Logprob GCRFCB je {}'.format(np.mean(logProbB)) + "\n")
file.write('CROSS Logprob GCRFCB_fast je {}'.format(np.mean(logProbBF)) + "\n")
file.write('CROSS Logprob GCRFCB2_fast je {}'.format(np.mean(logProbBF2)) + "\n")
file.write('CROSS Logprob GCRFCB21_fast je {}'.format(np.mean(logProbBF21)) + "\n")
file.write('CROSS Logprob GCRFCB22_fast je {}'.format(np.mean(logProbBF22)) + "\n")
file.write('CROSS Logprob GCRFCB3_fast je {}'.format(np.mean(logProbBF3)) + "\n")
file.write('CROSS Logprob GCRFCB4_fast je {}'.format(np.mean(logProbBF4)) + "\n")
file.write('CROSS Logprob GCRFCB41_fast je {}'.format(np.mean(logProbBF41)) + "\n")
file.write('CROSS Logprob GCRFCB42_fast je {}'.format(np.mean(logProbBF42)) + "\n")
file.write('CROSS Logprob GCRFCB5_fast je {}'.format(np.mean(logProbBF5)) + "\n")
file.write('CROSS Logprob GCRFCB6_fast je {}'.format(np.mean(logProbBF6)) + "\n")
file.write('CROSS Logprob GCRFCB61_fast je {}'.format(np.mean(logProbBF61)) + "\n")
file.write('CROSS Logprob GCRFCB62_fast je {}'.format(np.mean(logProbBF62)) + "\n")
file.write('CROSS Logprob GCRFCB7_fast je {}'.format(np.mean(logProbBF7)) + "\n")
file.write('CROSS Logprob GCRFCB8_fast je {}'.format(np.mean(logProbBF8)) + "\n")
file.write('CROSS Logprob GCRFCB81_fast je {}'.format(np.mean(logProbBF81)) + "\n")
file.write('CROSS Logprob GCRFCB82_fast je {}'.format(np.mean(logProbBF82)) + "\n")

file.write("--- %s seconds mean --- GCRFCNB" % (np.sum(timeNB)) + "\n")
file.write("--- %s seconds mean --- GCRFCB" % (np.sum(timeB)) + "\n")
file.write("--- %s seconds mean --- GCRFCB_fast" % (np.sum(timeBF)) + "\n")
file.write("--- %s seconds mean --- GCRFCB2_fast" % (np.sum(timeBF2)) + "\n")
file.write("--- %s seconds mean --- GCRFCB21_fast" % (np.sum(timeBF21)) + "\n")
file.write("--- %s seconds mean --- GCRFCB22_fast" % (np.sum(timeBF22)) + "\n")
file.write("--- %s seconds mean --- GCRFCB3_fast" % (np.sum(timeBF3)) + "\n")
file.write("--- %s seconds mean --- GCRFCB3_1fast" % (np.sum(timeBF3_1)) + "\n")
file.write("--- %s seconds mean --- GCRFCB3_2fast" % (np.sum(timeBF3_2)) + "\n")
file.write("--- %s seconds mean --- GCRFCB3_3fast" % (np.sum(timeBF3_3)) + "\n")
file.write("--- %s seconds mean --- GCRFCB5_1fast" % (np.sum(timeBF5_1)) + "\n")
file.write("--- %s seconds mean --- GCRFCB5_2fast" % (np.sum(timeBF5_2)) + "\n")
file.write("--- %s seconds mean --- GCRFCB5_3fast" % (np.sum(timeBF5_3)) + "\n")
file.write("--- %s seconds mean --- GCRFCB7_1fast" % (np.sum(timeBF7_1)) + "\n")
file.write("--- %s seconds mean --- GCRFCB7_2fast" % (np.sum(timeBF7_2)) + "\n")
file.write("--- %s seconds mean --- GCRFCB7_3fast" % (np.sum(timeBF7_3)) + "\n")
file.write("--- %s seconds mean --- GCRFCB9_1fast" % (np.sum(timeBF9_1)) + "\n")
file.write("--- %s seconds mean --- GCRFCB9_2fast" % (np.sum(timeBF9_2)) + "\n")
file.write("--- %s seconds mean --- GCRFCB9_3fast" % (np.sum(timeBF9_3)) + "\n")
file.write("--- %s seconds mean --- GCRFCB4_fast" % (np.sum(timeBF4)) + "\n")
file.write("--- %s seconds mean --- GCRFCB41_fast" % (np.sum(timeBF41)) + "\n")
file.write("--- %s seconds mean --- GCRFCB42_fast" % (np.sum(timeBF42)) + "\n")
file.write("--- %s seconds mean --- GCRFCB5_fast" % (np.sum(timeBF5)) + "\n")
file.write("--- %s seconds mean --- GCRFCB6_fast" % (np.sum(timeBF6)) + "\n")
file.write("--- %s seconds mean --- GCRFCB61_fast" % (np.sum(timeBF61)) + "\n")
file.write("--- %s seconds mean --- GCRFCB62_fast" % (np.sum(timeBF62)) + "\n")
file.write("--- %s seconds mean --- GCRFCB7_fast" % (np.sum(timeBF7)) + "\n")
file.write("--- %s seconds mean --- GCRFCB8_fast" % (np.sum(timeBF8)) + "\n")
file.write("--- %s seconds mean --- GCRFCB81_fast" % (np.sum(timeBF81)) + "\n")
file.write("--- %s seconds mean --- GCRFCB82_fast" % (np.sum(timeBF82)) + "\n")
file.write("--- %s seconds mean --- UNSTRUCTURED" % (np.sum(timeUN,axis = 0)) + "\n")   
file.write("--- %s seconds mean --- Strukturni" % (np.sum(time_ST,axis = 0)) + "\n")

file.close()

    