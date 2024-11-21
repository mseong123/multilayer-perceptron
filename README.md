# multilayer-perceptron | 42KL

This project's objective is to obtain a deep understanding of how neural network and deep learning works in the modern Machine Learning domain. We have to design and train a neural network with a minimum of 2 hidden layers to classify if a tumor is benign or malignant using a well known [`dataset`](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) of breast cancer from Wisconsin. This project is coded from scratch using Python, Numpy, Matplotlib and Pandas without any external ML libraries. 

This project also aims to familiarize with the hyperparameters used, their nuances and impact on the model metrics from both theoretical and heuristic approaches. One of the main uses of Neural Network vs other ML models is the ability to model non-linearly separable data and very complex relationships in the dataset.

Included in mandatory part:
 - splitting of main dataset into training and validation segment with random seed shuffling and stratification to avoid class bias
 - forward feed and error backpropagation between the layers
 - code the class in a modular way to resemble Tensorflow Keras sequential NN model with Layer class and weight/bias encapsulation
 - sigmoid activation function in hidden layers and softmax in output layer
 - Binary Cross Entropy function as loss value
 - Using metrics with graph visualisation to gauge model performance during training and post training.

Included in bonus part:
 - L2 regularization to avoid overfitting
 - early stopping
 - Adam gradient descent optimizer

Highly recommended reading: [Neural Networks and Deep Learning from Michael Nielsen](http://neuralnetworksanddeeplearning.com/index.html) which provides a comprehensive view of subject matter and distill the 
various technical concepts down to an intuitive and illustrative manner.

The model achieved accuracy of 94% to 97% in the randomised generated validation sets. 
Final results **120/100**

See [`Subject PDF`](https://github.com/mseong123/multilayer-perceptron/blob/main/en.subject.pdf) link.

## To install dependencies
```
pip install -r requirements.txt
```

## To run MANDATORY

### To split dataset
```
python split.py data.csv 42
```
This will split dataset into train.csv and test.csv using a 80/20 ratio and the samples will be stratified using the ratio of the classes. Use a seed parameter for reproducible results - default is 42 if parameter is not included.

### To train model
```
python train.py data_train.csv data_test.csv
```
This will initialize the model with random weights and biases based on Gaussian normal distribution. The model will be trained using `data_train.csv` and validated at every epoch with `data_test.csv`.
History of loss and accuracy metrics will be shown in graphs at the end of training. Network topology along with weights and biases are saved in `layer.npz`.

### To load and predict model
```
python predict.py data_test.csv layer.npz
```
This will load the network topology along with weights and biases (result from previous training and show the accuracy score using `data_test.csv`.

### To further evaluate model (amend train.py file in the below lines)
Adding and removing layers (line 44), set the size of neuron of each layer along with their activation function.
```
model:mlp.MLPClassifier = mlp.MLPClassifier([layers.DenseLayer(size=len(df_train.columns)),layers.DenseLayer(size=20, activation="sigmoid"),layers.DenseLayer(size=10, activation="sigmoid"),layers.DenseLayer(size=2, activation="softmax")])
```
Configuring hyperparameters (line 45)
- batch sizes
- epochs
- _lambda value and L2regularization(set to True or False)
- early stopping
- optimizer(set to "adam" or None)
```
loss_train, loss_valid, accuracy_train, accuracy_valid = model.fit(df_train.values, df_valid.values, y_train, y_valid, batch_size=16, epoch=10, _lambda=3, L2regularization=False, early_stopping=True, optimizer="adam")
```

