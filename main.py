import argparse
# import pandas as pd
import dask.dataframe as pd


from src.utils import MLE, MAP


def read_large_df(filename):
    df = pd.read_csv(filename, header=None)
    # for (columnName, columnData) in df.items():
    #     df[columnName] = pd.array.SparseArray(columnData.values, dtype='uint8')
    return df 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Document Classifier')
    parser.add_argument('--train-file', default='data/training.csv', type=str)
    parser.add_argument('--test-file', default='data/testing.csv', type=str)
    parser.add_argument('--vocabulary', default='data/vocabulary.txt', type=str)
    parser.add_argument('--labels', default='data/labels.txt', type=str)
    
    args = parser.parse_args()
    
    df = read_large_df(args.train_file)
    df_train, df_val = df.random_split([0.7, 0.3], random_state=42)
    print(len(df_train.columns))
    
    df_mle = MLE(df_train[df_train.columns[-1]])
    df_map = MAP(df_train)
    # print(len(df_mle))
    # print(len(df_map))
    
    df_test = read_large_df(args.test_file)
    print(len(df_test.columns))