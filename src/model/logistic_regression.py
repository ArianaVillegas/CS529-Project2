import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler


class LogisticRegression:
    """
    Logistic Regression.
    
    Parameters
    ----------
    lr : learning rate for gradient descent.
    reg : Regularization parameter
    """

    def __init__(self, lr = 0.01, reg=0.01):
        self.lr = lr
        self.reg = reg
        return
    

    '''
    Evaluate the Gradient descent in each time step, calculating first the gradient
    and performing the update formula.

    Parameters
    ----------
    x : dataframe of data.
    y : dataframe of labels.
    ypred: predicted probability matrix for every class
    '''
    def gradient_descent(self, x, y, ypred):
        rest = y - ypred
        grad = rest.T @ x
        self.w = self.w +  self.lr*grad - self.lr*self.reg*self.w
        return

    '''
    Make the prediction for every sample in the dataset.

    Parameter
    ----------
    x : dataframe of data.
    '''
    def prediction(self, x):
        first_factor = self.w @ x.transpose()
        first_factor -= np.max(first_factor, axis=0)
        log_den = np.log(np.sum(np.exp(first_factor), axis=0))
        probs = first_factor - log_den
        probs = np.exp(probs)
        return probs.T

    
    '''
    Return the prediction for every sample in the dataset, returning the
    label with maximum probabily for each class.

    Parameter
    ----------
    x : dataframe of data.
    '''
    def eval(self,x):
        x = sp.hstack([x, sp.csr_matrix(np.ones((x.shape[0], 1)))])
        probs = self.prediction(x)
        indexs = np.argmax(probs,axis=1) + 1
        return indexs

    '''
    Funtion to train the model based on the dataframe and number of iterations.

    Parameter
    ----------
    x : dataframe of data.
    y : dataframe of labels in one hot encoding.
    iterations: number of iterations for training.
    '''
    def train(self, x, y, iterations=10):     
        y = OneHotEncoder().fit_transform(y)
        # Initialize weights matrix
        n_rows = y.shape[1]
        n_columns = x.shape[1]
        self.w = sp.random(n_rows, n_columns+1, density=0.25).toarray()
        x = sp.hstack([x,sp.csr_matrix(np.ones((y.shape[0],1)))])
        start_time=time.time()
        interval = iterations//10
        for i in range(iterations):
            if i%interval == 0:
                print(i)
            # Predictions 
            y_pred = self.prediction(x)
            # Gradient Descent 
            self.gradient_descent(x,y,y_pred)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time taken for {iterations} iterations: {total_time:.6f} seconds")
        return  