import pandas as pd
import numpy as np


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
        grad = (np.matmul(np.transpose(y-ypred),x)) 
        self.w = self.w +  self.lr*grad - self.lr*self.reg*self.w
        return
    

    def prediction(self, x):
        # previous calculations
        first_factor = np.exp(x @ self.w.T)
        second_factor = (np.sum(first_factor,axis=1)).reshape(-1,1)
        probs = first_factor/(1+second_factor)
        last_prob = 1/(1+second_factor)
        probs = np.concatenate((probs[:,:-1], last_prob), axis=1) # concatenate all but the last class
        return probs

    def eval(self,x):
        x = x.toarray()
        x = normalize_array(x)
        x = np.insert(x,0,np.ones(len(x)), axis=1)
        probs = self.prediction(x)
        max_index = np.argmax(probs,axis=1)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(probs.shape[0]),max_index] = 1
        return one_hot


    def train(self, x, y, iterations=30):
        x = x.toarray()
        x = normalize_array(x)
        # Initialize weights matrix
        n_rows = len(y[0])
        self.n_clasess= n_rows
        n_columns = (x[0].T.shape)[0]
        self.w = np.random.rand(n_rows,n_columns+1)
        x = np.insert(x,0,np.ones(len(y)), axis=1)
        print(self.w[:,0])
        for i in range(iterations):
            # Predictions 
            y_pred = self.prediction(x)
            # Gradient Descent 
            self.gradient_descent(x,y,y_pred)
        print(self.w[:,0])
        return