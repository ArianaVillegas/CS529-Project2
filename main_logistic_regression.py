import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz, load_npz
import torch


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from src.model import LogisticRegression
from src.load import read_large_df
from src.utils import plot_confussion_matrix, cross_validation_split_logistic,cross_validation_split ,accuracy


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Document Classifier with Naive Bayes')
    parser.add_argument('--train-file', default='data/training.csv', type=str)
    parser.add_argument('--test-file', default='data/testing.csv', type=str)
    parser.add_argument('--output-file', default='data/output.csv', type=str)
    parser.add_argument('--vocabulary', default='data/vocabulary.txt', type=str)
    parser.add_argument('--labels', default='data/labels.txt', type=str)
    
    args = parser.parse_args()
    
    labels = np.array(pd.read_csv(args.labels, sep=' ', names=['name'])['name'])
    vocabulary = np.array(pd.read_csv(args.vocabulary, sep=' ', names=['name'])['name'])
    
    train_array = read_large_df(args.train_file)
    train_x, train_y = train_array[:,1:-1], train_array[:,-1]
    train_x = MaxAbsScaler().fit_transform(train_x)
    train_y = train_y.toarray()


    np.random.seed(3) #set numpy random seed    

    # Converting to one hot ecoding
    
    
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

    
    scores = cross_validation_split(train_x, train_y, LogisticRegression())
    print(scores)
    print(f'Accuracy: {np.round(np.mean(scores)*100, 2)}%')
    

    '''
    b_acc = []
    b_std = []
    lr_val = [0.001, 0.01]
    reg_val = [0.01, 0.001]
    iterations=1000

    for lr in lr_val:
        scores = cross_validation_split(train_x, train_y, LogisticRegression(lr=lr), n_splits=4)
        acc = np.mean(scores * 100)
        std = np.std(scores * 100)
        b_acc.append(acc)
        b_std.append(std)
        plt.annotate(str(np.round(acc, 2)), xy=(lr*1.25,acc-0.4), fontsize=14)
    plt.plot(np.array(lr_val), np.array(b_acc))
    plt.scatter(np.array(lr_val), np.array(b_acc))
    plt.fill_between(np.array(lr_val), np.array(b_acc) - np.array(b_std), np.array(b_acc) + np.array(b_std), alpha=0.2)
    plt.xscale('log')
    plt.xlim(right=8)
    # plt.ylim(bottom=83.5)
    plt.xlabel('lr', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('acc_logistic_regression.png')
    '''
    
    ''' Training the model '''
  
    model = LogisticRegression()
    num_iterations = 100
    model.train(train_x, train_y, iterations=num_iterations)
    
    train_pred = model.eval(train_x)
    val_pred = model.eval(val_x)
    
    # Printing the accuracy 
    print(f'Train accuracy: {np.round((train_y==train_pred).mean()*100, 2)}%')
    print(f'Val accuracy: {np.round((val_y==val_pred).mean()*100, 2)}%')
    
   # print("Train accuracy accuracy: " + str(accuracy(train_pred,train_y)*100))
   # print("Val accuracy: " + str(accuracy(val_pred, val_y)*100))

    #plot_confussion_matrix(val_y, val_pred, labels, 'cm_logistric_regression.png')
    
    '''
    sort_idx = model.rank_words(topk=20)
    words = vocabulary[sort_idx]
    print(words)

    test_array = read_large_df(args.test_file)
    test_idx, test_x = test_array[:,0].toarray(), test_array[:,1:]
    test_pred = model.eval(test_x)
    output = pd.DataFrame(data=np.concatenate([test_idx, test_pred], axis=1), columns=['id', 'class'], dtype=np.int32)
    output.to_csv(args.output_file, index=False)
    '''