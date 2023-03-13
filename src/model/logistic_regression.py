import pandas as pd
import numpy as np


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
      #  x = x.toarray()
        first_factor = np.exp(self.w0 + np.matmul(x,self.w.T))
        second_factor = (np.sum(first_factor,axis=1)).reshape(-1,1)
        probs = first_factor/(1+second_factor)
        last_prob = 1/(1+second_factor)
        probs = np.concatenate((probs[:,:-1], last_prob), axis=1) # concatenate all but the last class
        return probs

    def eval(self,x):
        x = x.toarray()
        probs = self.prediction(x)
        max_index = np.argmax(probs,axis=1)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(probs.shape[0]),max_index] = 1
        return one_hot


    def train(self, x, y, iterations=100):
        # Initialize weights matrix
       # print(x.shape)
        #print(y.shape)
        n_rows = len(y[0])
        n_columns = (x[0].T.shape)[0]
        self.w0 = np.random.rand(n_rows)
        self.w = np.random.rand(n_rows,n_columns)
        #print(self.w.shape)
        x = x.toarray()
        for i in range(iterations):
            # Predictions 
            y_pred = self.prediction(x)
            # Gradient Descent 
            self.gradient_descent(x,y,y_pred)
        return