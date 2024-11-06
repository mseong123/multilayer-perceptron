'''training of training and test data using MLP and saving weights and network topology to file '''

import sys
import pandas as pd
import numpy as np
import layers
import mlp

def one_hot_encode(df)->np.ndarray:
    '''return one hot encode classes of M and B into numpy array. ie [[1,0],[0,1]]'''
    return pd.get_dummies(df[1], dtype=int).to_numpy()

def preprocess(df_train:pd.DataFrame, df_valid:pd.DataFrame) -> tuple:
    '''preprocess dataframe and convert to numpy array for training'''
    y_train = one_hot_encode(df_train)
    y_valid = one_hot_encode(df_valid)
    df_train = df_train.drop([0,1], axis=1)
    df_valid = df_valid.drop([0,1], axis=1)
    return (y_train, y_valid, df_train, df_valid)


def main()->None:
    '''train MLP model using training and test data'''
    try:
        df_train = pd.read_csv(sys.argv[1], header=None)
        df_valid = pd.read_csv(sys.argv[2], header=None)
    except Exception as e:
        print(f"error: {e}")
    y_train, y_valid, df_train, df_valid = preprocess(df_train, df_valid)
    # initialise model, first layer based on features in dataset
    model:mlp.MLPClassifier = mlp.MLPClassifier([layers.DenseLayer(size=len(df_train.columns)),layers.DenseLayer(size=20, activation="sigmoid"),layers.DenseLayer(size=10, activation="sigmoid"),layers.DenseLayer(size=2, activation="softmax")])
    model.fit(df_train.values, df_valid.values, y_train, y_valid)

if __name__ == "__main__":
    main()