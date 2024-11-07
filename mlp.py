'''Multiayer Perceptron Classifier class definition'''
import numpy as np

class MLPClassifier():
    '''Based on Tensorflow Sequential model'''
    def __init__ (self, layers=None, seed:int=42):
        if layers is None:
            self._layers:list = []
        else:
            self._layers:list = layers
            self._init_weight(seed)
            self._init_bias(seed)
        # early stop parameters
        self._tol:float = 1e-4
        self._n_epoch_no_change:int = 5
        self._early_stop_count:int = 0
        # adam optimiser parameters
        self._beta1:float = 0.9
        self._beta2:float = 0.9999
        self._m_weight_array:list = [0] * len(self._layers)
        self._v_weight_array:list = [0] * len(self._layers)
        self._m_bias_array:list = [0] * len(self._layers)
        self._v_bias_array:list = [0] * len(self._layers)

    def _init_weight(self, seed:int) -> None:
        ''' init weight based on seed, shape is (input_size, size)'''
        rng = np.random.default_rng(seed)
        for idx, layer in enumerate(self._layers):
            if idx == 0:
                continue
            else:
                # Gaussian normal distribution of mean 0 and std deviation 1
                layer.weight = rng.normal(0, 1, tuple([self._layers[idx-1].size, layer.size]))
    
    def _init_bias(self, seed:int) -> None:
        '''init bias based on seed, 1 per neuron. Shape is (size,)'''
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
        # total weight in network
        # shape is (size of sample,). Add epsilon to prevent log(0) which is undefined
        return np.mean(-((y * np.log(y_predict + epsilon)) + ((1 - y) * (np.log(1 - y_predict + epsilon)))))
    
    def _accuracy_score(self, y:np.ndarray, y_predict:np.ndarray):
        '''calculate number of classification that's correct divide by length of sample'''
        for _, value in enumerate(y_predict):
            if value[0] < 0.5:
                value[0] = 0
            else:
                value[0] = 1
            if value[1] < 0.5:
                value[1] = 0
            else:
                value[1] = 1
        return np.sum(y[:,0] == y_predict[:,0]) / len(y)

    def _shuffle(self, y:np.ndarray, x:np.ndarray, seed:int) -> None:
        '''shuffle label and data in place according to seed'''
        rng = np.random.default_rng(seed)
        indices = np.arange(len(y))
        rng.shuffle(indices)
        return (y[indices], x[indices])
 
    def _sigmoid_derivative(self, y_predict:np.ndarray) -> np.ndarray:
        '''sigmoid derivative'''
        return y_predict * (1 - y_predict)

    def _feedforward(self, x:np.ndarray) -> np.ndarray:
        '''feedforward - used for prediction'''
        y_predict:np.ndarray = x
        for _, layer in enumerate(self._layers):
            layer.input_matrix = y_predict
            y_predict = layer.y_predict()
        return y_predict
 
    def _backpropagate(self, y_batch:np.ndarray, y_predict:np.ndarray, learning_rate:float, _lambda:float, L2regularization:bool) -> None:
        '''backpropagation'''
        layer_error:np.ndarray = np.array([])
        for i in range(len(self._layers) - 1, -1 , -1):
            layer = self._layers[i]
            if i == len(self._layers) - 1:
                # output layer gradient is (y_predict - y) * input
                layer_error:np.ndarray = y_predict - y_batch
                if L2regularization is True:
                # including term of weight decay (w * lambda/n) for L2 regularization to prevent overfitting
                    layer.weight -= (layer.input_matrix.T.dot(layer_error) + (layer.weight * _lambda / len(y_batch)))  * learning_rate
                else:
                    layer.weight -= layer.input_matrix.T.dot(layer_error) * learning_rate
                layer.bias -= np.mean(layer_error, axis=0) * learning_rate
                layer_error = layer_error.dot(layer.weight.T)
            elif 0 <  i < len(self._layers) - 1:
                # each hidden layer gradient is (error * sigmoid derivative) * input 
                layer_error_delta:np.ndarray = (layer_error * self._sigmoid_derivative(layer.y_predict()))
                if L2regularization is True:
                # including term of weight decay (w * lambda/n) for L2 regularization to prevent overfitting
                    layer.weight -= (layer.input_matrix.T.dot(layer_error_delta) + (layer.weight * _lambda / len(y_batch))) * learning_rate
                else:
                    layer.weight -= layer.input_matrix.T.dot(layer_error_delta) * learning_rate
                layer.bias -= np.mean(layer_error_delta, axis=0) * learning_rate
                layer_error = layer_error_delta.dot(layer.weight.T)
            else:
                # no action for input layer
                continue

    def _adam_adjustment(self, gradient:np.ndarray, bias:np.ndarray, iteration:int, i:int)->tuple:
            '''adam adjustments'''
            if iteration == 1:
                # at first iteration make all numpy same shape
                self._m_weight_array[i] = np.zeros(gradient.shape)
                self._v_weight_array[i] = np.zeros(gradient.shape)
                self._m_bias_array[i] = np.zeros(bias.shape)
                self._v_bias_array[i] = np.zeros(bias.shape)
            # Weights
            self._m_weight_array[i] = (self._beta1 * self._m_weight_array[i]) + ((1 - self._beta1) * gradient)
            self._v_weight_array[i] = (self._beta2 * self._v_weight_array[i]) + ((1 - self._beta2) * (gradient **2))
            adjusted_m_weight = self._m_weight_array[i] / (1 - (self._beta1 ** iteration))
            adjusted_v_weight = self._v_weight_array[i] / (1 - (self._beta2 ** iteration))
            # Bias
            self._m_bias_array[i] = (self._beta1 * self._m_bias_array[i]) + ((1 - self._beta1) * bias)
            self._v_bias_array[i] = (self._beta2 * self._v_bias_array[i]) + ((1 - self._beta2) * (bias **2))
            adjusted_m_bias = self._m_bias_array[i] / (1 - (self._beta1 ** iteration))
            adjusted_v_bias = self._v_bias_array[i] / (1 - (self._beta2 ** iteration))
            return (adjusted_m_weight, adjusted_v_weight, adjusted_m_bias, adjusted_v_bias)
                


    def _backpropagate_adam(self, y_batch:np.ndarray, y_predict:np.ndarray, learning_rate:float, _lambda:float, L2regularization:bool, iteration:int) -> None:
        '''backpropagation for adam'''
        layer_error:np.ndarray = np.array([])
        epsilon = 1e-10
        for i in range(len(self._layers) - 1, -1 , -1):
            layer = self._layers[i]
            if i == len(self._layers) - 1:
                # output layer gradient is (y_predict - y) * input
                layer_error:np.ndarray = y_predict - y_batch
                # Calculate gradients
                if L2regularization is True:
                    gradient:np.ndarray = layer.input_matrix.T.dot(layer_error) + (layer.weight * _lambda / len(y_batch))
                else:
                    gradient:np.ndarray = layer.input_matrix.T.dot(layer_error)
                bias:np.ndarray = np.mean(layer_error, axis=0)
                adjusted_m_weight, adjusted_v_weight, adjusted_m_bias, adjusted_v_bias = self._adam_adjustment(gradient, bias, iteration, i)
                layer.weight -= (adjusted_m_weight / (np.sqrt(adjusted_v_weight) + epsilon)) * learning_rate
                layer.bias -=  (adjusted_m_bias / (np.sqrt(adjusted_v_bias) + epsilon)) * learning_rate
                layer_error = layer_error.dot(layer.weight.T)
            elif 0 <  i < len(self._layers) - 1:
                # each hidden layer gradient is (error * sigmoid derivative) * input 
                layer_error_delta:np.ndarray = (layer_error * self._sigmoid_derivative(layer.y_predict()))
                if L2regularization is True:
                    gradient:np.ndarray = layer.input_matrix.T.dot(layer_error_delta) + (layer.weight * _lambda / len(y_batch))
                else:
                    gradient:np.ndarray = layer.input_matrix.T.dot(layer_error_delta)
                bias:np.ndarray = np.mean(layer_error_delta, axis=0)
                adjusted_m_weight, adjusted_v_weight, adjusted_m_bias, adjusted_v_bias = self._adam_adjustment(gradient, bias, iteration, i)
                layer.weight -= (adjusted_m_weight / (np.sqrt(adjusted_v_weight) + epsilon)) * learning_rate
                layer.bias -=  (adjusted_m_bias / (np.sqrt(adjusted_v_bias) + epsilon)) * learning_rate
                layer_error = layer_error_delta.dot(layer.weight.T)
            else:
                # no action for input layer
                continue
    
    def _check_early_stopping(self, loss_valid:list, i:int) -> bool:
        '''If validation loss is not improving by tol for no. of epochs consecutively, return false'''
        stop:bool = False
        if i > 0 and (loss_valid[i - 1] - loss_valid[i]) < self._tol:
            self._early_stop_count += 1
        if i > 0 and (loss_valid[i - 1] - loss_valid[i]) > self._tol:
            self._early_stop_count = 0
        if self._early_stop_count >= self._n_epoch_no_change:
            stop = True
        return stop
 
    @property
    def layers(self) -> np.ndarray:
        '''getter for layers'''
        return self._layers
    
    @layers.setter
    def layers(self, layers) -> None:
        '''setter for layers'''
        self._layers = layers

    def predict(self, y_valid:np.ndarray, x_valid:np.ndarray) -> tuple:
        '''return model metrics'''
        x_valid = self._normalize(x_valid)
        y_predict_valid = self._feedforward(x_valid)
        loss_valid:np.float64 = self._binary_cross_entropy_loss(y_valid, y_predict_valid)
        accuracy_valid:np.float64 = self._accuracy_score(y_valid, y_predict_valid)
        error_valid:np.float64 = 1 - accuracy_valid
        return (loss_valid, accuracy_valid, error_valid)

 
    def fit(self, x_train:np.ndarray, x_valid:np.ndarray, y_train:np.ndarray, y_valid:np.ndarray, seed:int=42, optimizer=None, _lambda:float=3, L2regularization=False, early_stopping=False, learning_rate:float=0.01, batch_size:int=30, epoch:int=10)->tuple[list,list]:
        '''train model based on hyperparams'''
        print(f"x_train shape : {x_train.shape}")
        print(f"x_valid shape : {x_valid.shape}")
        x_train = self._normalize(x_train)
        x_valid = self._normalize(x_valid)
        count:int = 0

        # metrics to return
        loss_train:list = []
        loss_valid:list = []
        accuracy_train:list = []
        accuracy_valid:list = []

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
                if optimizer == "adam":
                    self._backpropagate_adam(y_batch, y_predict, learning_rate, _lambda, L2regularization, count + 1)
                else:
                    self._backpropagate(y_batch, y_predict, learning_rate, _lambda, L2regularization)
                start_index = end_index
                end_index = start_index + batch_size
                count += 1
            y_predict_train = self._feedforward(x_train)
            y_predict_valid = self._feedforward(x_valid)
            loss_train.append(self._binary_cross_entropy_loss(y_train, y_predict_train))
            loss_valid_epoch = self._binary_cross_entropy_loss(y_valid, y_predict_valid)
            loss_valid.append(loss_valid_epoch)
            accuracy_train.append(self._accuracy_score(y_train, y_predict_train))
            accuracy_valid.append(self._accuracy_score(y_valid, y_predict_valid))
            # Early stopping
            if early_stopping is True:
                result:bool = self._check_early_stopping(loss_valid, i)
                if result is True:
                    break 
            print(f"epoch {i+1}/{epoch} - loss: {loss_train[i]:.4f} \
 - val_loss: {loss_valid[i]:.4f}")
        return (loss_train, loss_valid, accuracy_train, accuracy_valid)
            




