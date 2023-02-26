import pandas as pd
import numpy as np

from src.utils import MLE, MAP


class NaiveBayes:
    """
    NaiveBayes Classifier.
    
    Parameters
    ----------
    beta : beta prior that add extra observation for each feature.
    """
    
    def __init__(self, beta=None):
        self.beta = beta
    
    def train(self, x, y):
        """
        Train NaiveBayes Classifier with a training dataframe.
        
        Parameters
        ----------
        x : dataframe of data
        y : dataframe of labels
        """
        self.df_mle = np.log2(MLE(y))
        self.df_map = np.log2(MAP(x, y, self.beta))
        
    def eval(self, x):
        """
        Train NaiveBayes Classifier with a training dataframe.
        
        Parameters
        ----------
        x : dataframe of data
        
        Returns
        -------
        y: dataframe with labels column.
        """
        product = x.dot(self.df_map.T)
        add = np.add(product, self.df_mle)
        y = np.reshape(np.argmax(add, axis=1), newshape=(x.shape[0], 1))
        return y + 1
        