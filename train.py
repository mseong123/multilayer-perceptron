'''training of training and test data using MLP and saving weights and network topology to file '''

import sys
import pandas as pd
import numpy as np
import layers
import mlp

def main()->None:
    '''train MLP model using training and test data'''
    model:mlp.MLPClassifier = mlp.MLPClassifier([layers.DenseLayer(size=30),layers.DenseLayer(size=30),layers.DenseLayer(size=2)])
    try:
        df_train = pd.read_csv(sys.argv[1])
        df_valid = pd.read_csv(sys.argv[2])
    except Exception as e:
        print(f"error: {e}")
    print(df_train)
    print(df_valid)

if __name__ == "__main__":
    main()