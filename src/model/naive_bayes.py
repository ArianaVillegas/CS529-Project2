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
        self.mle = None
        self.map = None
        self.weight = None
    
    def train(self, x, y):
        #print(x[100])
        #self.w = np.random.rand(np.unique(y).size -1, x[0].size + 1)
        #print(self.w.shape)
        """
        Train NaiveBayes Classifier with a training dataframe, and store
        the log values to avoid recalculating them in evaluation phase.
        
        Parameters
        ----------
        x : dataframe of data.
        y : dataframe of labels.
        """
        self.df_mle = MLE(y)
        self.df_map = MAP(x, y, self.beta)
        
    def eval(self, x):
        """
        Train NaiveBayes Classifier with a training dataframe.
        
        Parameters
        ----------
        x : dataframe of data.
        
        Returns
        -------
        y: dataframe with labels column.
        """
        log_mle = np.log2(self.df_mle)
        log_map = np.log2(self.df_map)
        product = x.dot(log_map.T)
        add = np.add(product, log_mle)
        y = np.reshape(np.argmax(add, axis=1), newshape=(x.shape[0], 1))
        return y + 1
    
    def rank_words(self, topk=100, imp_type='gini'):
        """
        Rank words based in the impurity of each word. To measure impurtiy we
        use the Gini index after normalizing probabilities per word to sum up
        1. 
        
        Parameters
        ----------
        topk : the number or words index that should be returned.
        imp_type : impurity metric. Default: gini.
        
        Returns
        -------
        sort_i  dx: index of the words ranked by influence of the optimizer.
        """
        print(self.df_map.sum(axis=1))
        print(self.df_map.sum(axis=0).shape)
        self.weight = self.df_map / self.df_map.sum(axis=0)
        print(self.weight.shape)
        if imp_type == 'gini':
            self.weight = np.power(self.weight, 2).sum(axis=0)
        elif imp_type == 'entropy':
            self.weight = np.multiply(self.weight, np.log2(self.weight)).sum(axis=0)
        else:
            return Exception(f'Impurity type {imp_type} not defined')
        sort_idx = np.argsort(self.weight)[:topk]
        return sort_idx