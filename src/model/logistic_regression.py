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

    def __init__(self, lr = 0.01, reg=0.1):
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
        print(np.max(first_factor))
        print(np.min(first_factor))
        #first_factor = np.clip(a_min=None,a_max=700,a=first_factor)

        first_factor = np.exp(first_factor)
        #print(np.max(first_factor.toarray()))
        #print(np.max(first_factor.toarray()))
        #first_factor = np.exp(first_factor)
        #second_factor = (np.sum(first_factor,axis=1)).reshape(-1,1)
        #probs = first_factor/(1+second_factor)
        #last_prob = 1/(1+second_factor)
        first_factor[-1] = np.ones(first_factor.shape[1])
        probs = first_factor/np.sum(first_factor,axis=0)
        #probs = np.concatenate((probs[:,:-1], last_prob), axis=1)
        return probs.T

    def eval(self,x):
        x = x.toarray()
        x = normalize_array(x)
        #x = MaxAbsScaler().fit_transform(x)
        x = np.insert(x,0,np.ones(len(x)), axis=1)
        x = sp.csr_matrix(x)
        probs = self.prediction(x)
        probs = np.array(probs)
        print(f"probs shape: {probs.shape}")
        indexs = np.argmax(probs,axis=1) + 1
        print(f"index shape: {indexs.shape}")
        #one_hot = np.zeros_like(probs)
        #one_hot[np.arange(probs.shape[0]),max_index] = 1
        return indexs


    def train(self, x, y, iterations=100):
        
        #x = MaxAbsScaler().fit_transform(x)        
        x = normalize_array(x)
        y = OneHotEncoder().fit_transform(y)
        # Initialize weights matrix
        n_rows = y.shape[1]  
        self.n_clasess= n_rows
        n_columns = (x[0].T.shape)[0]
        #self.w = np.random.rand(n_rows,n_columns+1)
        self.w = sp.random(n_rows,n_columns+1, density=0.3).toarray()
        print(f"max: {np.max(self.w)}")
        print(f"min: {np.min(self.w)}")
        x = sp.hstack([x,sp.csr_matrix(np.ones((y.shape[0],1)))])
        # Convert to sparse matrix
        #y = sp.csr_matrix(y)
        #print(self.w[:,1])
        start_time=time.time()
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        
        
        for i in range(iterations):
            # Predictions 
            y_pred = self.prediction(x)
            #y_pred = sp.csr_matrix(y_pred)
            # Gradient Descent 
            self.gradient_descent(x,y,y_pred)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time taken for {iterations} iterations: {total_time:.6f} seconds")
        return  