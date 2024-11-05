'''Layers class'''
import numpy as np

class DenseLayer():
    '''DenseLayer class with weight and bias attributes'''
    def __init__(self, size:tuple|int, activation:str|None = None):
        self._weight:np.ndarray = np.array([])
        self._bias:np.ndarray = np.array([])
        if isinstance(size, tuple):
            self._size:int = size[0]
        elif isinstance(size, int):
            self._size:int = size
        self._activation:str = activation
        self._input_matrix:np.ndarray = np.array([])
    
    @property
    def weight(self)->np.ndarray:
        '''getter for layer weight'''
        return self._weight

    @property
    def bias(self)->np.ndarray:
        '''getter for layer bias'''
        return self._bias
    
    @property
    def size(self)->int:
        '''getter for layer size'''
        return self._size
    
    @property
    def input_matrix(self) -> np.ndarray:
        '''getter for input matrix'''
        return self._input_matrix

    @property
    def activation(self) -> str:
        '''getter for activation'''
        return self._activation

    @weight.setter
    def weight(self, weight:np.ndarray)->None:
        self._weight = weight
    
    @bias.setter
    def bias(self, bias:np.ndarray) -> None:
        self._bias = bias

    @input_matrix.setter
    def input_matrix(self, _input_matrix:np.ndarray) -> None:
        self._input_matrix = _input_matrix

    def _sigmoid(self, z) -> np.float64:
        '''sigmoid function. returns shape (sample_size,)'''
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, y_predict:np.ndarray) -> np.ndarray:
        return y_predict * (1 - y_predict)

    def _softmax(self, z) -> np.ndarray:
        '''sigmoid function. Returns shape (sample_size, size)'''
        # to prevent overflow by shifting max value to 0 and every else to negative
        z_exp = np.exp(z - np.max(z))
        return z_exp / z_exp.sum(axis=-1, keepdims=True)

    def y_predict(self) -> np.ndarray:
        '''return z after activation function'''
        z:np.ndarray = np.dot(self._input_matrix, self.weight) + self.bias
        if len(self._weight) == 0:
            return self.input_matrix
        elif self._activation == "sigmoid":
            return self._sigmoid(z)
        elif self._activation == "softmax":
            return self._softmax(z) 

    def error_delta(self, y:np.ndarray) -> np.ndarray:
        '''return error of layer'''
        y_predict:np.ndarray = self.y_predict()
        if self.activation == "softmax":
            return y_predict[:,0] - y
        if self.activation == "sigmoid":
            return (y_predict - y) * self._sigmoid_derivative(self.y_predict())







    

    
