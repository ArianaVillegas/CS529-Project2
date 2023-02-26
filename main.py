import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.model import NaiveBayes
from src.load import read_large_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Document Classifier')
    parser.add_argument('--train-file', default='data/training.csv', type=str)
    parser.add_argument('--test-file', default='data/testing.csv', type=str)
    parser.add_argument('--output-file', default='data/output.csv', type=str)
    parser.add_argument('--vocabulary', default='data/vocabulary.txt', type=str)
    parser.add_argument('--labels', default='data/labels.txt', type=str)
    
    args = parser.parse_args()
    
    labels = np.array(pd.read_csv(args.labels, sep=' ', names=['name'])['name'])
    
    train_array = read_large_df(args.train_file)
    train_x, val_x, train_y, val_y = train_test_split(train_array[:,:-1], train_array[:,-1], test_size=0.3, random_state=42)
    train_idx, train_x = train_x[:,0], train_x[:,1:]
    val_idx, val_x = val_x[:,0], val_x[:,1:]
    
    train_y = train_y.toarray()
    val_y = val_y.toarray()
    
    model = NaiveBayes()
    model.train(train_x, train_y)
    train_pred = model.eval(train_x)
    val_pred = model.eval(val_x)
    print(f'Train accuracy: {np.round((train_y==train_pred).mean()*100, 2)}%')
    print(f'Val accuracy: {np.round((val_y==val_pred).mean()*100, 2)}%')
    
    cm = confusion_matrix(val_y, val_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.matshow(cm, cmap=plt.cm.Spectral_r)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=round(cm[i, j], 2), va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.xticks(range(len(labels)), [''] + labels, rotation='vertical', fontsize=18)
    plt.yticks(range(len(labels)), [''] + labels, fontsize=18)
    plt.tight_layout()
    plt.savefig('cm.png')
    
    
    test_array = read_large_df(args.test_file)
    test_idx, test_x = test_array[:,0].toarray(), test_array[:,1:]
    test_pred = model.eval(test_x)
    output = pd.DataFrame(data=np.concatenate([test_idx, test_pred], axis=1))
    output.to_csv(args.output_file, header=None)