'''program to load topology/weights/biases and predict on a given dataset'''

import sys
import numpy as np
import pandas as pd
import mlp

def one_hot_encode(df)->np.ndarray:
    '''return one hot encode classes of M and B into numpy array. ie [[1,0],[0,1]]'''
    return pd.get_dummies(df[1], dtype=int).to_numpy()

def preprocess(df_valid:pd.DataFrame) -> tuple:
    '''preprocess dataframe and convert to numpy array for training'''
    y_valid = one_hot_encode(df_valid)
    df_valid = df_valid.drop([0,1], axis=1)
    return (y_valid, df_valid)

def main()->None:
    '''prediction program'''
    try:
        df_valid:pd.DataFrame = pd.read_csv(sys.argv[1], header=None)
        data = np.load(sys.argv[2], allow_pickle=True)
    except Exception as e:
        print(f"error: {e}")
    y_valid, df_valid = preprocess(df_valid)
    # load layers
    loaded_layers:list = data["layers"]
    model:mlp.MLPClassifier = mlp.MLPClassifier()
    print("Loading model topology, weights and biases...")
    model.layers = loaded_layers
    loss, accuracy, error = model.predict(y_valid, df_valid.to_numpy())
    print(f"Validation data shape: {df_valid.shape}")
    print("Prediction metrics:")
    print(f"Loss value as per BCE = {loss:.4f}")
    print(f"Accuracy of prediction = {accuracy:.2f}")
    print(f"Error of prediction = {error:.2f}")

     

if __name__ == "__main__":
    main()