import numpy as np


def MLE(labels):
    """
    Quick utility that estiamte MLE for all the classes in labels.
    
    Parameters
    ----------
    *labels : serie with the labels of the classes in a dataset.
    
    Returns
    -------
    mle: serie with different class values and their MLE.
    """
    return labels.value_counts(normalize=True)


def MAP(df):
    """
    Quick utility that estiamte MAP for all word given a class.
    
    Parameters
    ----------
    df : Dataframe with words repetitions and the last column has
    the label of the class.
    
    Returns
    -------
    map: dataframe of all the vocabulary given an specific class.
    """
    size = len(df.columns)
    alpha = 1 + 1/size
    doc_count = df.groupby(by=[df.columns[-1]]).sum()
    df_nom = doc_count + (alpha-1)
    df_den = df_nom + ((alpha-1) * size)
    df_map = df_nom.div(df_den, axis=0)
    return df_map
    