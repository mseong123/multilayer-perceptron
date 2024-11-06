'''Multiayer Perceptron Classifier class definition'''
import numpy as np

class MLPClassifier():
    '''Based on Tensorflow Sequential model'''
    def __init__ (self, layers=None, seed:int=42):
        if layers is None:
            layers = []
        else:
            self._layers=layers
            self._init_weight(seed)
            self._init_bias(seed)

    def _init_weight(self, seed:int) -> None:
        rng = np.random.default_rng(seed)
        for idx, layer in enumerate(self._layers):
            if idx == 0:
                continue
            else:
                # Gaussian normal distribution of mean 0 and std deviation 1
                layer.weight = rng.normal(0, 1, tuple([self._layers[idx-1].size, layer.size]))
    
    def _init_bias(self, seed:int) -> None:
        rng = np.random.default_rng(seed)
        for idx, layer in enumerate(self._layers):
            if idx == 0:
                continue
            else:
                # Gaussian normal distribution of mean 0 and std deviation 1
                layer.bias = rng.normal(0, 1, layer.size)

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
    
    def _accuracy_score(self, y:np.ndarray, y_predict:np.ndarray):
        pass

    def _shuffle(self, y:np.ndarray, x:np.ndarray, seed:int) -> None:
        '''shuffle label and data in place according to seed'''
        rng = np.random.default_rng(seed)
        indices = np.arange(len(y))
        rng.shuffle(indices)
        return (y[indices], x[indices])
    
    def _sigmoid_derivative(self, y_predict:np.ndarray) -> np.ndarray:
        return y_predict * (1 - y_predict)

    def _feedforward(self, x:np.ndarray) -> np.ndarray:
        y_predict:np.ndarray = x
        for _, layer in enumerate(self._layers):
            layer.input_matrix = y_predict
            y_predict = layer.y_predict()
        return y_predict
        
    def _backpropagate(self, y_batch:np.ndarray, y_predict:np.ndarray, learning_rate:float) -> None:
        layer_error_delta:np.ndarray = np.array([])
        for i in range(len(self._layers) - 1, -1 , -1):
            layer = self._layers[i]
            if i == len(self._layers) - 1:
                layer_error:np.ndarray = y_predict - y_batch
                layer.weight -= layer.input_matrix.T.dot(layer_error) * learning_rate
                layer.bias -= np.mean(layer_error, axis=0) * learning_rate
                layer_error_delta = layer_error.dot(layer.weight.T)
            elif 0 <  i < len(self._layers) - 1:
                layer.weight -= layer.input_matrix.T.dot(layer_error_delta) * learning_rate
                layer.bias -= np.mean(layer_error_delta, axis=0) * learning_rate
                layer_error_delta = layer_error_delta.dot(layer.weight.T)
            else:
                continue
    
    @property
    def layers(self) -> np.ndarray:
        '''getter for layers'''
        return self._layers
    

    
    def fit(self, x_train:np.ndarray, x_valid:np.ndarray, y_train:np.ndarray, y_valid:np.ndarray, seed:int=42, early_stopping=False, learning_rate:float=0.01, batch_size:int=50, epoch:int=1)->tuple[list,list]:
        '''train model based on hyperparams'''
        print(f"x_train shape : {x_train.shape}")
        print(f"x_valid shape : {x_valid.shape}")
        x_train = self._normalize(x_train)
        x_valid = self._normalize(x_valid)
        for i in range(epoch):
            # For each Epoch shuffle training set(validation set don't require shuffling)
            (y_shuffled, x_shuffled)=self._shuffle(y_train, x_train, seed)
            start_index:int = 0
            end_index:int = start_index + batch_size
            while end_index < len(x_shuffled):
                # mini batch SGD update every iteration (up to entire dataset for one epoch)
                x_batch:np.ndarray = x_shuffled[start_index:end_index]
                y_batch:np.ndarray = y_shuffled[start_index:end_index]
                y_predict:np.ndarray = self._feedforward(x_batch)
                self._backpropagate(y_batch, y_predict, learning_rate)
                start_index = end_index
                end_index = start_index + batch_size
            y_predict_train = self._feedforward(x_train)
            y_predict_valid = self._feedforward(x_valid)
            self._accuracy_score(y_train, y_predict_train)
            print(f"epoch {i+1}/{epoch} - loss: {self._binary_cross_entropy_loss(y_train, y_predict_train)} \
                  - val_loss: {self._binary_cross_entropy_loss(y_valid, y_predict_valid)}")
            




