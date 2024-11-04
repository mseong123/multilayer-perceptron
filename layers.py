'''Layers class'''
import numpy as np

class DenseLayer():
    '''DenseLayer class with weight and bias attributes'''
    def __init__(self, shape:tuple|int, activate:str=None):
        self._weight = np.array([])
        self._bias = np.array([])
        if isinstance(shape, tuple):
            self._size = shape[0]
        elif isinstance(shape, int):
            self._size = shape
        self._activate=activate
    
    def init_weight_bias(self, input:np.ndarray | None):
        '''init weight size and value'''
        if input is not None:
            self._weight:np.ndarray = np.zeros(input.shape[1], self._size)
            self._bias:np.ndarray = np.zeros(self._size)

    @property
    def weight(self)->np.ndarray:
        return self._weight

    @property
    def bias(self)->np.ndarray:
        return self._bias

    @weight.setter
    def weight(self, weight:np.ndarray)->None:
        self._weight = weight
    
    @bias.setter
    def bias(self, bias:np.ndarray) -> None:
        self._bias = bias
    

    
