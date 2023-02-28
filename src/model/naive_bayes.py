import numpy as np


def MLE(labels):
    """
    Utility that estiamte MLE for all the classes in labels.
    
    Parameters
    ----------
    *labels : serie with the labels of the classes in a dataset.
    
    Returns
    -------
    mle: serie with different class values and their MLE.
    """
    size = labels.shape[0]
    _, counts = np.unique(labels, return_counts=True)
    return counts/size


def MAP(x, y, beta=None):
    """
    Utility that estiamte MAP for all word given a class.
    
    Parameters
    ----------
    x : data with the number of repetitions by word in the vocabulary.
    y : labels for each element in the data.
    beta: beta prior that add extra observation for each feature.
    
    Returns
    -------
    map: data of all the vocabulary given an specific class.
    """
    y = np.append(y, np.reshape(np.arange(y.shape[0]), newshape=y.shape), axis=1)
    y = y[y[:, 0].argsort()]
    y_idx = np.split(y[:,1], np.unique(y[:,0], return_index=True)[1][1:])
    
    size = x.shape[1]
    beta = 1/size if beta == None else beta
    alpha = 1 + beta
    doc_count = np.concatenate([np.array(x[idx,:].sum(axis=0)) for idx in y_idx])
    
    nom = doc_count + (alpha-1)
    den = np.reshape(np.sum(doc_count, axis=1) + ((alpha-1) * size), newshape=(nom.shape[0], 1))
    map_ = np.divide(nom, den)
    return map_


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
        