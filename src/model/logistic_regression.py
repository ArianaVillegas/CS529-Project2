import pandas as pd
import numpy as np


class LogisticRegression:
    """
    Logistic Regression.
    
    Parameters
    ----------
    lr : learning rate for gradient descent.
    """

    def __init__(self, lr = 0.01):
        self.lr = lr
        return
    
    def gradient_descent(self, y, ypred):
        self.w = self.w + self.lr * (y - ypred)  - self.lr*self.w
        return
    

    def prediction(self, x, y):
        # previous calculations
        calculation =  np.exp(self.w0 + np.diagonal(np.dot(x, np.transpose(self.w0))))

        # Calculate probability of first class
        probs = calculation / 1 + np.sum(calculation)
        last_prob = 1 / 1 + np.sum(calculation)

        return np.append(probs, np.array(last_prob))

    def train(self, x, y):
        # Initialize weights matrix
        self.w0 = np.random.rand(len(np.unique(y)))
        self.w = np.random.rand(len(np.unique(y))-1,len(x[0]))
        # Predictions
        
        # Gradient Descent 
        self.gradient_descent()

        # 
    

        return