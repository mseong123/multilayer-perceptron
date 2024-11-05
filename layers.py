'''Layers class'''
import numpy as np

class DenseLayer():
    '''DenseLayer class with weight and bias attributes'''
    def __init__(self, size:tuple|int, activate:str=None):
        self._weight = np.array([])
        self._bias = np.array([])
        if isinstance(size, tuple):
            self._size = size[0]
        elif isinstance(size, int):
            self._size = size
        self._activate=activate
    
    def init_weight_bias(self, input:np.ndarray | None):
        '''init weight size and value'''
        if input is not None:
            self._weight:np.ndarray = np.zeros(input.size[1], self._size)
            self._bias:np.ndarray = np.zeros(self._size)

    @property
    def weight(self)->np.ndarray:
        return self._weight

    @property
    def bias(self)->np.ndarray:
        return self._bias
    
    @property
    def size(self)->int:
        return self._size

    @weight.setter
    def weight(self, weight:np.ndarray)->None:
        self._weight = weight
    
    @bias.setter
    def bias(self, bias:np.ndarray) -> None:
        self._bias = bias

    def sigmoid(self, z) -> np.float64:
        return 1 / (1 + np.exp(-z))

    def output(self, )
    

    
