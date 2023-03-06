import pandas as pd
import numpy as np


class LogisticRegression:
    """
    Logistic Regression.
    
    Parameters
    ----------
    lr : learning rate for gradient descent.
    reg : Regularization
    """

    def __init__(self, lr = 0.01, reg=0.5):
        self.lr = lr
        self.reg = reg
        return
    
    def gradient_descent(self, x, y, ypred):
        self.w = self.w + self.lr * (np.matmul(x, np.transpose(y - ypred))) - self.lr*self.reg*self.w
        return
    

    def prediction(self, x):
        # previous calculations
        #calculation =  np.exp(self.w0 + np.dot(self.w, np.transpose(x)))
        first_factor = np.exp(self.w0 + np.matmul(x, np.transpose(self.w)))
        second_factor = (np.sum(first_factor,axis=1)).reshape(-1,1)        
        probs = first_factor/(1+second_factor)
        probs = np.concatenate(probs[:,:-1], 1/(1+second_factor)) # concatenate all but the last class.
        # Calculate probabilities for all classes
        #probs = calculation/(1+np.sum(calculation))
        #probs = np.append(probs, 1/(1+np.sum(calculation)))
        return probs

    def fit(self, x, y, iterations=1000):
        # Initialize weights matrix
        n_rows = np.unique(y).T.shape
        self.w0 = np.random.rand(*n_rows) 
        self.w = np.random.rand(*(n_rows), *(x[0].T.shape))
        for i in range(iterations):
            # Predictions 
            y_pred = self.prediction(x)
            # Gradient Descent 
            self.gradient_descent(x,y,y_pred)
        return