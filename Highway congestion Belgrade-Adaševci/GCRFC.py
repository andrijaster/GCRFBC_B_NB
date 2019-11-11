# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 08:19:07 2018

@author: Andrija Master
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
import scipy as sp


""" CLASS GCRFC """

class GCRFC:
    
    def __init__(self):
        pass
    
    def muKov(alfa, R, Precison, Noinst, NodeNo):
        mu = np.zeros([Noinst,NodeNo])
        bv = 2*np.matmul(R,alfa)
        bv = bv.reshape([Noinst,NodeNo])
        Kov = np.linalg.inv(Precison)
        for m in range(Noinst):
            mu[m,:] = Kov[m,:,:].dot(bv[m,:])
        return mu,Kov
        
    def Prec(alfa,beta,NodeNo,Se,Noinst):
        alfasum = np.sum(alfa)
        Q1 = np.identity(NodeNo)*alfasum
        Q2 = np.zeros([Noinst,NodeNo,NodeNo])
        Prec = np.zeros([Noinst,NodeNo,NodeNo])
        pomocna = np.zeros(Se.shape)
        for j in range(Se.shape[1]):
            pomocna[:,j,:,:] = Se[:,j,:,:] * beta[j]
        Q2 = -np.sum(pomocna,axis = 1)
        for m in range(Noinst):
            Prec[m,:,:] = 2*(Q2[m,:,:]+np.diag(-Q2[m,:,:].sum(axis=0))+Q1)
        return Prec
    
    def sigmaCal(ceta): # Provereno
        Sigma = 1/(1 + np.exp(-ceta))
        Dsigmadceta = Sigma*(1 - Sigma)
        return Sigma,Dsigmadceta
    
    def Lambda(Noinst,NodeNo,ceta,sigma): # Provereno
        Lambda = np.zeros([Noinst,NodeNo,NodeNo])
        diagonal = np.tanh(ceta/2)/(4*ceta)
        for m in range(Noinst):
            Lambda[m,:,:] = np.diag(diagonal[m,:])
        return Lambda
    
    def Sinv_S(Prec,Lambda): # Provereno
        Sinv = Prec + 2*Lambda     
        S = np.linalg.inv(Sinv)
        return Sinv,S
    
    def Tmat(Y): # Provereno
        T = Y - 1/2
        return T
    
    def mivec(S,T,Prec,mu,Noinst,NodeNo): # Provereno
        mi = np.zeros([Noinst,NodeNo])
        for m in range(Noinst):
            mi[m,:] = S[m,:,:].dot(T[m,:]+Prec[m,:,:].dot(mu[m,:]))
        return mi
    
    def sigmaCalPRED(ceta): # Provereno
        Sigma = 1/(1 + np.exp(-ceta))
        Sigma[Sigma>0.99999999] = 0.99999999
        Sigma[Sigma<1e-10] = 1e-10
        return Sigma
    

    """ PREDICT """
    
    def predict(self,R,Se): # PROVERENO

        def function(var, mu):
            def integral(z):
                sigmoid = 1/(1+np.exp(-z))
                marnormal = 1/np.sqrt(2*np.pi*var)*np.exp(-(z-mu)**2/(2*var))
                return sigmoid*marnormal
            return integral
        
        NodeNo = Se.shape[3]
        Noinst = Se.shape[0]
        Precison = GCRFC.Prec(self.alfa, self.beta, NodeNo, Se, Noinst)
        mu,Kovmat = GCRFC.muKov(self.alfa, R, Precison, Noinst, NodeNo)
         
        Var=np.zeros([Noinst,NodeNo])
        Prob=np.zeros([Noinst,NodeNo])
        Clas=np.zeros([Noinst,NodeNo])
        
        for p in range(Noinst):
            Var[p,:]=Kovmat[p,:,:].diagonal()
            for i in range(NodeNo):
#                if Var[p,i] > 0.01:
                Prob[p,i] = (sp.integrate.quad(function(Var[p,i], mu[p,i]),  (mu[p,i] - 10*np.sqrt(Var[p,i])),  (mu[p,i] + 10*np.sqrt(Var[p,i]))))[0]
#                else:
#                Prob[p,i] = GCRFC.sigmaCalPRED(mu[p,i])
                Clas[p,i]=np.round(Prob[p,i], 0)
        print(np.linalg.norm(Var))
        self.Prob = Prob
        self.Class = Clas
        return self.Prob, self.Class, Var
    
    """ FIT """
    
    def fit(self,R,Se,Y,x0 = None, learn = 'SLSQP', maxiter = 1000, learnrate = 0.1, learnratec = 0.3):
        
        def dLdX(x, ModelUNNo, ModelSTNo, TotalNo, Se, R, Y, Noinst, NodeNo):
                
            def Trace(x, y): # Provereno
                i1,j1 = x.shape
                trMat = 0
                for k in range(i1):
                    trMat = trMat+x[k,:].dot(y[:,k])
                return trMat
            
            def dPrecdalfa(NodeNo,ModelUNNo): # Provereno
                dPrecdalfa = np.zeros([ModelUNNo,NodeNo,NodeNo])
                dQ1dalfa = np.identity(NodeNo)
                for p in range(ModelUNNo):
                    dPrecdalfa[p,:,:] = dQ1dalfa*2
                return dPrecdalfa
        
            def dbdalfa(ModelUNNo,Noinst,R,NodeNo): # Provereno  1 
                dbdalfa = np.zeros([Noinst,ModelUNNo,NodeNo])
                for m in range(ModelUNNo):
                    dbdalfa[:,m,:] = 2*R[:,m].reshape([Noinst, NodeNo])
                return dbdalfa
            
            def dmutdalfa(dbdalfa,dPrecdalfa,Kov,ModelUNNo,Noinst,mu,NodeNo): # Provereno
                dmutdalfa = np.zeros([Noinst,ModelUNNo,NodeNo])
                for m in range(Noinst):
                    for p in range(ModelUNNo):
                        dmutdalfa[m,p,:] = (dbdalfa[m,p,:]-dPrecdalfa[p,:,:].dot(mu[m,:])).T.dot(Kov[m,:,:])
                return dmutdalfa
            
            def dPrecdbeta(Noinst,ModelSTNo,NodeNo,Se): # PROVERENO
                dPrecdbeta = np.zeros([Noinst,ModelSTNo,NodeNo,NodeNo])
                dPrecdbeta = -Se
                for m in range(Noinst):
                    for L in range(ModelSTNo):
                        dPrecdbeta[m,L,:,:]=2*(dPrecdbeta[m,L,:,:] + np.diag(-dPrecdbeta[m,L,:,:].sum(axis=1))) 
                return dPrecdbeta
            
            def dmutdbeta(dPrecdbeta,mu,Kov,Noinst,ModelSTNo,NodeNo): # Provereno
                dmutdbeta = np.zeros([Noinst,ModelSTNo,NodeNo])
                for m in range(0,Noinst):
                    for p in range(0,ModelSTNo):
                        dmutdbeta[m,p,:] = (-dPrecdbeta[m,p,:,:].dot(mu[m,:])).T.dot(Kov[m,:,:])
                return dmutdbeta
            
            def SigFun_dlamdcet(Noinst,NodeNo,ceta,sigma,dsigmadceta): # Provereno *2
                dlambdadceta = np.zeros([Noinst,NodeNo,NodeNo,NodeNo])
                sigmafun = (1/sigma + 1/2*ceta)*dsigmadceta + (1/2*sigma - 3/4)
                diagonal = 1/(4*ceta**2)*(0.5*ceta*(1-np.tanh(ceta/2)**2)-np.tanh(ceta/2))
                for m in range(Noinst):
                    for p in range(NodeNo):
                        dlambdadceta[m,p,p,p] = diagonal[m,p]
                return sigmafun,dlambdadceta
            
    
            def dLdceta(S,Sinv,dlambdadceta,mu,mi,T,Prec,sigmafun,Noinst,Nonode): # Provereno *2
                DLdceta = np.zeros([Noinst,Nonode])
                for i in range(Noinst):
                    for j in range(Nonode):
                        DLdceta[i,j] = -Trace(S[i,:,:],dlambdadceta[i,j,:,:])\
                        - 2*((T[i,:].T + mu[i,:].T.dot(Prec[i,:,:])).dot(S[i,:,:]).dot(dlambdadceta[i,j,:,:]).dot(S[i,:,:])).dot(Sinv[i,:,:]).dot(mi[i,:]) +\
                        mi[i,:].T.dot(dlambdadceta[i,j,:,:]).dot(mi[i,:]) + sigmafun[i,j]
                return -1*DLdceta
        
            def dLdbeta(T,ModelSTNo,Noinst,S,Sinv,mu,mi,Prec,dPrecdalfa,KovMat,dmutdbeta,dPrecdbeta): # Provereno 
                DLdbeta=np.zeros(ModelSTNo)
                for k in range(ModelSTNo):
                    for i in range(Noinst):
                        DLdbeta[k] = -1/2*Trace(S[i,:,:],dPrecdbeta[i,k,:,:]) + (-(T[i,:].T + mu[i,:].T.dot(Prec[i,:,:])).dot(S[i,:,:]).dot(dPrecdbeta[i,k,:,:]).dot(S[i,:,:]) +\
                        dmutdbeta[i,k,:].dot(Prec[i,:,:]).dot(S[i,:,:]) + mu[i,:].T.dot(dPrecdbeta[i,k,:,:]).dot(S[i,:,:])).dot(Sinv[i,:,:]).dot(mi[i,:]) +\
                        1/2*mi[i,:].T.dot(dPrecdbeta[i,k,:,:]).dot(mi[i,:]) - dmutdbeta[i,k,:].dot(Prec[i,:,:]).dot(mu[i,:]) - 1/2*mu[i,:].T.dot(dPrecdbeta[i,k,:,:]).dot(mu[i,:]) +\
                        1/2*Trace(KovMat[i,:,:],dPrecdbeta[i,k,:,:]) + DLdbeta[k]
                return -1*DLdbeta
            
            def dLdalfa(T,ModelUNNo,Noinst,S,Sinv,mu,mi,Prec,dPrecdalfa,KovMat,dmutdalfa): # Provereno 
                DLdalfa = np.zeros(ModelUNNo)
                for k in range(ModelUNNo):
                    for i in range(Noinst):
                        DLdalfa[k] = - 1/2*Trace(S[i,:,:],dPrecdalfa[k,:,:]) +\
                        (-(T[i,:].T + mu[i,:].T.dot(Prec[i,:,:])).dot(S[i,:,:]).dot(dPrecdalfa[k,:,:]).dot(S[i,:,:]) + dmutdalfa[i,k,:].dot(Prec[i,:,:]).dot(S[i,:,:]) +\
                         mu[i,:].T.dot(dPrecdalfa[k,:,:]).dot(S[i,:,:])).dot(Sinv[i,:,:]).dot(mi[i,:])+1/2*mi[i,:].T.dot(dPrecdalfa[k,:,:]).dot(mi[i,:])\
                         - dmutdalfa[i,k,:].dot(Prec[i,:,:]).dot(mu[i,:]) - 1/2*mu[i,:].T.dot(dPrecdalfa[k,:,:]).dot(mu[i,:]) +\
                         1/2*Trace(KovMat[i,:,:],dPrecdalfa[k,:,:]) + DLdalfa[k]
                return -1*DLdalfa
            
            if learn == 'GRAD':
                alfa = np.exp(x[:ModelUNNo])
                alfa[alfa<1e-8] = 1e-8
                print('alfa je {}'.format(alfa))
                beta = np.exp(x[ModelUNNo:ModelSTNo+ModelUNNo])
                beta[beta<1e-8] = 1e-8
                print('beta je {}'.format(beta))
                ceta = x[-TotalNo:].reshape(Noinst,NodeNo)
            else:
                alfa = x[:ModelUNNo]
                beta = x[ModelUNNo:ModelSTNo+ModelUNNo]
                ceta = x[-TotalNo:].reshape(Noinst,NodeNo)
            sigma,dsigmadceta = GCRFC.sigmaCal(ceta)
            DPrecdalfa = dPrecdalfa(NodeNo,ModelUNNo)
            Precison = GCRFC.Prec(alfa, beta, NodeNo, Se, Noinst)
            DPrecdbeta = dPrecdbeta(Noinst,ModelSTNo,NodeNo,Se)
            lambdaMat = GCRFC.Lambda(Noinst,NodeNo,ceta,sigma)
            Sinv,S = GCRFC.Sinv_S(Precison,lambdaMat)
            mu,kovMat = GCRFC.muKov(alfa, R, Precison, Noinst, NodeNo)
            mu[np.isnan(mu)] = 0
            T = GCRFC.Tmat(Y)
            mi = GCRFC.mivec(S,T,Precison,mu,Noinst,NodeNo)
            Dbdalfa = dbdalfa(ModelUNNo,Noinst,R,NodeNo)
            Dmutdalfa = dmutdalfa(Dbdalfa,DPrecdalfa,kovMat,ModelUNNo,Noinst,mu,NodeNo)
            Dmutdbeta = dmutdbeta(DPrecdbeta,mu,kovMat,Noinst,ModelSTNo,NodeNo)
            sigmafun,Dlambdadceta = SigFun_dlamdcet(Noinst,NodeNo,ceta,sigma,dsigmadceta)    
            DLdceta = dLdceta(S,Sinv,Dlambdadceta,mu,mi,T,Precison,sigmafun,Noinst,NodeNo)
            DLdceta = np.reshape(DLdceta,(1,TotalNo))
            DLdceta.shape=-1
            DLdbeta = dLdbeta(T,ModelSTNo,Noinst,S,Sinv,mu,mi,Precison,DPrecdalfa,kovMat,Dmutdbeta,DPrecdbeta)
            DLdalfa = dLdalfa(T,ModelUNNo,Noinst,S,Sinv,mu,mi,Precison,DPrecdalfa,kovMat,Dmutdalfa)
            DLdx = np.concatenate((DLdalfa,DLdbeta,DLdceta))
            if learn == 'GRAD':
                DLdx = np.concatenate((alfa*DLdalfa,beta*DLdbeta,DLdceta))
            else:
                DLdx = np.concatenate((DLdalfa,DLdbeta,DLdceta))                    
            return DLdx
        
        def L(x, ModelUNNo, ModelSTNo, TotalNo, Se, R, Y, Noinst, NodeNo): 
            if learn == 'GRAD':
                alfa = np.exp(x[:ModelUNNo])
                beta = np.exp(x[ModelUNNo:ModelSTNo+ModelUNNo])
                ceta = x[-TotalNo:].reshape(Noinst,NodeNo)
            else:
                alfa = x[:ModelUNNo]
                beta = x[ModelUNNo:ModelSTNo+ModelUNNo]
                ceta = x[-TotalNo:].reshape(Noinst,NodeNo)
            sigma,dsigmadceta = GCRFC.sigmaCal(ceta)
            Precison = GCRFC.Prec(alfa, beta, NodeNo, Se, Noinst)
            lambdaMat = GCRFC.Lambda(Noinst,NodeNo,ceta,sigma)
            Sinv,S = GCRFC.Sinv_S(Precison,lambdaMat)
            mu,kovMat = GCRFC.muKov(alfa, R, Precison, Noinst, NodeNo)
            T = GCRFC.Tmat(Y)
            mi = GCRFC.mivec(S,T,Precison,mu,Noinst,NodeNo)
            L=0
            for i in range(Noinst):
                    sigmafun = np.sum(np.log(sigma[i,:]) - ceta[i,:]/2 + np.diag(lambdaMat[i,:,:])*ceta[i,:]**2)
                    L= 1/2*np.log(np.linalg.det(S[i,:,:])) - 1/2*np.log(np.linalg.det(kovMat[i,:,:])) \
                    + 1/2*mi[i,:].T.dot(Sinv[i,:,:]).dot(mi[i,:]) - 1/2*mu[i,:].T.dot(Precison[i,:,:]).dot(mu[i,:]) + sigmafun + L
            print('skor je {}'.format(L))
            return -1*L   
            
        
        ModelUNNo = R.shape[1]
        NodeNo = Se.shape[2]
        Noinst = Se.shape[0]
        ModelSTNo = Se.shape[1]
        TotalNo = Noinst * NodeNo
        if x0 is None:
            if learn == 'GRAD':
                x01 = np.random.randn(ModelUNNo + ModelSTNo)*1
                x02 = np.random.randn(TotalNo)*1
                x0 = np.concatenate((x01,x02))
                np.save('x0',x0)
            else:
                x01 = np.abs(np.random.randn(ModelUNNo + ModelSTNo))*1
                x02 = np.random.randn(TotalNo)*1
                x0 = np.concatenate((x01,x02))
                np.save('x0',x0)
        if learn == 'SLSQP':
            cons = ({'type':'ineq', 'fun': lambda x: x[:ModelUNNo+ModelSTNo]})
            res = minimize(L, x0, method='TNC', jac=dLdX, args=(ModelUNNo, ModelSTNo, TotalNo, Se, R, Y, Noinst, NodeNo) \
                       ,options={'disp': True,'maxiter': 1000},constraints = cons)
            self.alfa = res.x[:ModelUNNo]
            self.beta = res.x[ModelUNNo:ModelSTNo+ModelUNNo]
            
        elif learn == 'TNC':
            bnd = ((1e-6,None),)*(ModelSTNo+ModelUNNo) + ((None,None),)*TotalNo
            res = sp.optimize.fmin_tnc(L, x0, fprime = dLdX, \
                                       args=(ModelUNNo, ModelSTNo, TotalNo, Se, R, Y, Noinst, NodeNo),\
                                       bounds = bnd, maxfun=maxiter)
            self.alfa = res[0][:ModelUNNo]
            self.beta = res[0][ModelUNNo:ModelSTNo+ModelUNNo]
            self.x = res[0]
        elif learn == 'EXP':
            x = x0
            u1 = np.log(x0[:ModelUNNo+ModelSTNo])            
            for i in range(maxiter):
                DLdx = -dLdX(x, ModelUNNo, ModelSTNo, TotalNo, Se, R, Y, Noinst, NodeNo)
                u1 = u1 + learnrate*x[:ModelUNNo+ModelSTNo]*DLdx[:ModelUNNo+ModelSTNo]
                ceta = x[ModelUNNo+ModelSTNo:] + learnratec*DLdx[ModelUNNo+ModelSTNo:]
                L1 = -L(x, ModelUNNo, ModelSTNo, TotalNo, Se, R, Y, Noinst, NodeNo)
                alfbet = np.exp(u1)
                x = np.concatenate([alfbet,ceta])
                print('U iteciji {} DLDX je {}'.format(i,DLdx[:ModelSTNo+ModelUNNo]))
                print('U iteciji {} L je {}'.format(i,L1))
            self.alfa = x[:ModelUNNo]
            self.beta = x[ModelUNNo:ModelSTNo+ModelUNNo]
            self.x = x
        elif learn == 'GRAD':
            res = minimize(L, x0, method = 'CG', jac = dLdX, \
                                       args=(ModelUNNo, ModelSTNo, TotalNo, Se, R, Y, Noinst, NodeNo),\
                                       options={'disp': True,'maxiter': maxiter})
            self.alfa = np.exp(res.x[:ModelUNNo])
            self.beta = np.exp(res.x[ModelUNNo:ModelSTNo+ModelUNNo])
            self.x = res.x
                

