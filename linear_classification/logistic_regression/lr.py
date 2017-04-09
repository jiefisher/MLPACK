'''
Created on 2017/4/9/

@author: jiefisher
'''
import numpy as np
import pandas as pd
class LogisticRegression(object):
    '''
    classdocs
    '''


    def __init__(self, learning_rate=.0001,l2=.0005,iters=1000):
        '''
        Constructor
        '''
        self.learning_rate=learning_rate
        self.l2=l2
        self.iters=iters
        
    def train(self,x_train,x_target):
        '''
        Train
        '''
        x_feature=np.mat(x_train)
        x_label=np.mat(x_target).T
        l1=np.ones(x_feature.shape[0])
        x_feature=np.column_stack((x_feature,l1))
        weights=np.mat(np.zeros(x_feature.shape[1]))
        
        for i in range(0,self.iters):
            A=x_feature*weights.T
        
            E=1/(1+np.exp(-A))-x_label
            
            weights=weights-self.learning_rate*((x_feature.T*E).T+\
                                                self.l2*weights*weights.T*weights)
            self.weights=weights
            
    def predict(self,y_train):
        '''
        predict
        '''
        result=[]
        y_feature=np.mat(y_train)
        l1=np.ones(y_feature.shape[0])
        y_feature=np.column_stack((y_feature,l1))
        A=y_feature*self.weights.T

        E=1/(1+np.exp(-A))
        for i in range(0,E.shape[0]):
            for j in range(0,E.shape[1]):
                if E[i][j]>=0.5:
                    #print(1)
                    result.append(1)
                else:
                    #print(0)
                    result.append(0)
        return result

