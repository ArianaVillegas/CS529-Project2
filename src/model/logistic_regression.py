import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

def normalize_array(arr, epsilon=1):
    """
    Normalizes a 2D array by dividing each element by the sum of its corresponding column,
    replacing any zero sums with a small value epsilon to avoid division by zero.
    """
    col_sums = np.sum(arr, axis=0)
    zero_sums = np.where(col_sums == 0)[0]
    col_sums[zero_sums] = epsilon
    return arr / col_sums

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
    
    def gradient_descent(self, x, y, ypred):
        rest = y - ypred
        grad = rest.T @ x
        self.w = self.w +  self.lr*grad - self.lr*self.reg*self.w
        return


    def prediction(self, x):
        first_factor = self.w @ x.transpose()
        first_factor = np.exp(first_factor)
        first_factor[-1] = np.ones(x.shape[0])
        probs = first_factor/np.sum(first_factor, axis=0)
        return probs.T

    def eval(self,x):
        x = sp.hstack([x, sp.csr_matrix(np.ones((x.shape[0], 1)))])
        probs = self.prediction(x)
        indexs = np.argmax(probs,axis=1) + 1
        return indexs


    def train(self, x, y, iterations=100):     
        y = OneHotEncoder().fit_transform(y)
        # Initialize weights matrix
        n_rows = y.shape[1]
        n_columns = x.shape[1]
        self.w = sp.random(n_rows, n_columns+1, density=0.3).toarray()
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