import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.model import NaiveBayes
from src.load import read_large_df
from src.utils import plot_confussion_matrix, cross_validation_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Document Classifier with Naive Bayes')
    parser.add_argument('--train-file', default='data/training.csv', type=str)
    parser.add_argument('--test-file', default='data/testing.csv', type=str)
    parser.add_argument('--output-file', default='data/output.csv', type=str)
    parser.add_argument('--vocabulary', default='data/vocabulary.txt', type=str)
    parser.add_argument('--labels', default='data/labels.txt', type=str)
    
    args = parser.parse_args()
    
    labels = np.array(pd.read_csv(args.labels, sep=' ', names=['name'])['name'])
    
    train_array = read_large_df(args.train_file)
    train_x, train_y = train_array[:,1:-1], train_array[:,-1]
    scores = cross_validation_split(train_x, train_y, NaiveBayes())
    print(f'Accuracy: {np.round(np.mean(scores)*100, 2)}%')
    
    b_acc = []
    b_std = []
    b_val = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    for b in b_val:
        scores = cross_validation_split(train_x, train_y, NaiveBayes(beta=b))
        acc = np.mean(scores * 100)
        std = np.std(scores * 100)
        b_acc.append(acc)
        b_std.append(std)
        plt.annotate(str(np.round(acc, 2)), xy=(b*1.25,acc-0.4), fontsize=14)
    plt.plot(np.array(b_val), np.array(b_acc))
    plt.scatter(np.array(b_val), np.array(b_acc))
    plt.fill_between(np.array(b_val), np.array(b_acc) - np.array(b_std), np.array(b_acc) + np.array(b_std), alpha=0.2)
    plt.xscale('log')
    plt.xlim(right=8)
    # plt.ylim(bottom=83.5)
    plt.xlabel('Beta', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('acc_naive_bayes.png')
    
    
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)
    train_y = train_y.toarray()
    val_y = val_y.toarray()
    
    model = NaiveBayes(beta=0.01)
    model.train(train_x, train_y)
    train_pred = model.eval(train_x)
    val_pred = model.eval(val_x)
    print(f'Train accuracy: {np.round((train_y==train_pred).mean()*100, 2)}%')
    print(f'Val accuracy: {np.round((val_y==val_pred).mean()*100, 2)}%')
    
    plot_confussion_matrix(val_y, val_pred, labels, 'cm_naive_bayes.png')
        
    test_array = read_large_df(args.test_file)
    test_idx, test_x = test_array[:,0].toarray(), test_array[:,1:]
    test_pred = model.eval(test_x)
    output = pd.DataFrame(data=np.concatenate([test_idx, test_pred], axis=1), columns=['id', 'class'], dtype=np.int32)
    output.to_csv(args.output_file, index=False)