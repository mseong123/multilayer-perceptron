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

    def _normalize(self, x:np.ndarray) -> np.ndarray:
        '''convert each value to normalised Z score with mean of 0 and std deviation of 1'''
        # normalise input
        return ((x - np.mean(x,axis=0).reshape(1,-1)) / np.std(x,axis=0).reshape(1,-1))
    
    def _binary_cross_entropy_loss(self, y:np.ndarray, y_predict:np.ndarray)->np.ndarray:
        '''calculate and return loss value'''
        epsilon = 1e-10
        # shape is (size of sample,)
        # add epsilon to prevent log(0) which is undefined
        return np.mean(-((y * np.log(y_predict + epsilon)) + \
                        ((1 - y) * (np.log(1 - y_predict + epsilon)))))
    
    def shuffle(self, y:np.ndarray, X:np.ndarray, seed:int) -> None:
        '''shuffle label and data in place according to seed'''
        rng = np.random.default_rng(seed)
        indices = np.arange(len(y))
        rng.shuffle(indices)
        return (y[indices], X[indices])

    def gradient(self, y:np.ndarray, y_predict:np.ndarray,X:np.ndarray,\
                  sgd:bool) ->tuple[np.ndarray]:
        '''calculate gradient of weight and bias'''
        #dL/dy_predict = - ((y/y_predict) - (1-y/1-y_predict))
        # dy_predict/dz = y_predict * (1 - y_predict)
        # dz/dw
        # gradient of loss w.r.t. dL/dw = (dL/dy_predict) x (dy_predict / dz) x (dz / dw)
        # simplify = ((y_predict - y) * X) / sample_size.
        # bias is the same except for dz/db = 1 hence (y_predict - y) / sample size
        if sgd is False:
            gradient_weight:np.ndarray = np.dot(X.T, y_predict - y) / len(y_predict)
        else:
            gradient_weight:np.ndarray = X * (y_predict - y)
        gradient_bias:np.ndarray = np.mean(y_predict - y)
        return (gradient_weight, gradient_bias)
    
    def _feedforward(self, x_shuffled:np.ndarray) -> None:
        y_predict:np.ndarray = np.array([])
        for _, layer in enumerate(self._layers):
            layer.input_matrix = y_predict
            y_predict = layer.y_predict(x_shuffled)
    
    def _sigmoid_derivative(self, y_predict:np.ndarray) -> np.ndarray:
        return y_predict * (1 - y_predict)

    def _backpropagate(self, y_batch:np.ndarray) -> None:
        layer_error_delta:np.ndarray = np.array([])
        for layer in reversed(list(enumerate(self._layers))):
            if layer.activation == "softmax":
                layer.weight -= layer.weight.T.dot(layer.error_delta(y_batch))
                layer_error_delta = layer.error_delta(y_batch) 
            elif layer.activation == "sigmoid":
                layer.weight -= layer.weight.T.dot(layer_error_delta)
                layer_error_delta = layer.error_delta()

        

    def fit(self, x_train:np.ndarray, x_valid:np.ndarray, y_train:np.ndarray, y_valid:np.ndarray, seed:int=42, early_stopping=False, learning_rate:float=0.01, batch_size:int=8, epoch:int=84)->tuple[list,list]:
        '''train model based on hyperparams'''
        print(f"x_train shape : {x_train.shape}")
        print(f"x_valid shape : {x_valid.shape}")
        self._init_weight(seed)
        self._init_bias(seed)
        x_train = self._normalize(x_train)
        x_valid = self._normalize(x_valid)
        for i in range(epoch):
            # For each Epoch shuffle training set(validation set don't require shuffling)
            (y_shuffled, x_shuffled)=self.shuffle(y_train, x_train, seed)
            start_index:int = 0
            end_index:int = start_index + batch_size
            while end_index < len(x_shuffled):
                # mini batch SGD update every iteration (up to entire dataset for one epoch)
                x_batch:np.ndarray = x_shuffled[start_index:end_index]
                y_batch:np.ndarray = y_shuffled[start_index:end_index]
                self._feedforward(x_batch)
                self._backpropagate(y_batch)
                # mini batch SGD backpropagate
                gradient_weight, gradient_bias = self.gradient(y_batch, y_predict, X_batch, False)
                self._weight -= self._learning_rate * gradient_weight
                self._bias -= self._learning_rate * gradient_bias
                start_index = end_index
                end_index = start_index + self._batch_size




