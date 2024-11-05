'''Multiayer Perceptron Classifier class definition'''
import numpy as np

class MLPClassifier():
    '''Based on Tensorflow Sequential model'''
    def __init__ (self, layers=None):
        if layers is None:
            layers = []
        else:
            self._layers=layers

    def _init_weight(self, seed:int) -> None:
        rng = np.random.default_rng(seed)
        for idx, layer in enumerate(self._layers):
            if idx == 0:
                continue
            else:
                # Gaussian normal distribution of mean 0 and std deviation 1
                layer.weight = rng.normal(0, 1, tuple(self._layers[idx-1].size, layer.size))
    
    def _init_bias(self, seed:int) -> None:
        rng = np.random.default_rng(seed)
        for idx, layer in enumerate(self._layers):
            if idx == 0:
                continue
            else:
                # Gaussian normal distribution of mean 0 and std deviation 1
                layer.bias = rng.normal()

    def _feedforward(self):

    def fit(self, x_train:np.ndarray, x_valid:np.ndarray, y:np.ndarray, seed:int=42, learning_rate:float=0.01, iteration:int=100, batch_size:int=8, epoch:int=84)->tuple[list,list]:
        '''train model based on hyperparams'''
        print(f"x_train shape : {x_train.shape}")
        print(f"x_valid shape : {x_valid.shape}")
        self._init_weight(seed)
        self._init_bias(seed)



