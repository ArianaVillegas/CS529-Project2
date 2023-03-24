import pandas as pd
import numpy as np
import scipy.sparse as sp

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

    def __init__(self, lr = 0.01, reg=0.5):
        self.lr = lr
        self.reg = reg
        return
    
    def gradient_descent(self, x, y, ypred):
        rest = y - ypred
        grad = rest.transpose() @ x
        self.w = self.w +  self.lr*grad - self.lr*self.reg*self.w
        return
    

    def prediction(self, x):
        
        first_factor = x @ self.w.T
        first_factor = sp.csr_matrix(np.exp(first_factor.toarray()))
        second_factor = (np.sum(first_factor,axis=1)).reshape(-1,1)
        probs = first_factor/(1+second_factor)
        last_prob = 1/(1+second_factor)
        probs = np.concatenate((probs[:,:-1], last_prob), axis=1)
        return probs

    def eval(self,x):
        x = x.toarray()
        x = normalize_array(x)
        x = np.insert(x,0,np.ones(len(x)), axis=1)
        x = sp.csr_matrix(x)
        probs = self.prediction(x)
        probs = np.array(probs)
        max_index = np.argmax(probs,axis=1)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(probs.shape[0]),max_index] = 1
        return one_hot


    def train(self, x, y, iterations=100):
        x = x.toarray()
        x = normalize_array(x)
        # Initialize weights matrix
        n_rows = len(y[0])  
        self.n_clasess= n_rows
        n_columns = (x[0].T.shape)[0]
        self.w = np.random.rand(n_rows,n_columns+1)
        self.w = sp.csr_matrix(self.w)
        x = np.insert(x,0,np.ones(len(y)), axis=1)

        # Convert to sparse matrix
        x = sp.csr_matrix(x)
        y = sp.csr_matrix(y)
        print(self.w[:,1])
        
        for i in range(iterations):
            # Predictions 
            y_pred = self.prediction(x)
            y_pred = sp.csr_matrix(y_pred)
            # Gradient Descent 
            self.gradient_descent(x,y,y_pred)
        print(self.w[:,1])
        return  