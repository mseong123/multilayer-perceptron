'''training of training and test data using MLP and saving weights and network topology to file '''

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def plot_metrics(loss_train:np.ndarray, loss_valid:np.ndarray, accuracy_train:np.ndarray, accuracy_valid:np.ndarray) -> None:
    '''plotting function matplotlib'''
    _, ax = plt.subplots(1,2, layout='constrained')
    ax[0].plot(loss_train, color="blue", label="training data loss")
    ax[0].plot(loss_valid, color="red", label="validation data loss")
    ax[0].set_title("Loss Metric")
    ax[0].legend()
    ax[1].plot(accuracy_train, color="green", label="training data accuracy")
    ax[1].plot(accuracy_valid, color="yellow", label="validation data accuracy")
    ax[1].set_title("Accuracy Metric")
    ax[1].legend()
    plt.show()

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
    loss_train, loss_valid, accuracy_train, accuracy_valid = model.fit(df_train.values, df_valid.values, y_train, y_valid, batch_size=16, epoch=100, _lambda=3, L2regularization=False, early_stopping=True)
    plot_metrics(loss_train, loss_valid, accuracy_train, accuracy_valid)
    print("saving model './layer.npz' to disk...")
    np.savez("layer.npz", layers=model.layers)


if __name__ == "__main__":
    main()