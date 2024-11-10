# multilayer-perceptron | 42KL

This project's objective is to obtain a deep understanding of how neural network and deep learning works in the modern Machine Learning domain. We have to design and train a neural network with a minimum of 2 hidden layers to classify if a tumor is benign or malignant using a well known [`dataset`](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) of breast cancer from Wisconsin. This project is coded from scratch using Python, Numpy, Matplotlib and Pandas without any external ML libraries. 

This project also aims to familiarize with the hyperparameters used, their nuances and impact on the model from both theoretical and heuristic approaches. One of the main uses of Neural Network vs other ML models is the ability to model non-linearly separable data and very complex relationships in the dataset.

Included in mandatory part:
 - splitting of dataset into training and validation
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


Included in mandatory part:
 - Z score normalization of dataset
 - coding implementation of data science and statistical concepts
 - data cleaning (how to deal with missing data)
 - data visualisation (to check for collinearity between features using histogram, scatterplot and scatterplot matrix)
 - backpropagation of loss values to respective weights and biases using gradient descent

Included in BONUS part:
 - implementation of stochastic gradient descent
 - implementation of mini-batch gradient descent

The model achieved accuracy of 99% in all scenarios using test data as per Scikit-learn.metrics accuracy score. The classes and functions are coded using reference to Scikit-learn LogisticRegression class. 
Final results **125/100**

See [`Subject PDF`](https://github.com/mseong123/dslr/blob/main/en.subject.pdf) link.

## To install dependencies
```
pip install -r requirements.txt
```

## To run MANDATORY

### To train model
```
python logreg_train.py dataset_train.csv
```
This will train the regression model based on the `dataset_train.csv` and will show loss function value at every 100 iterations. Adjust hyperparameters learning rate and iteration in logreg_train.py to tune model's
performance. The above will output file `weight.npz` which will be used by model to predict classification output based on test dataset.

### To predict model
```
python logreg_predict.py dataset_test.csv weight.npz
```
This will output a file `houses.csv` which shows labelling of the classes in the test dataset. 

### To evaluate model
```
python evaluate.py
```
This will evaluate the labels generated in `houses.csv` against `dataset_truth.csv` using Scikit-learn accuracy_score method to evaluate precision of model (99%)

## To run BONUS

### To train model using SGD
```
python logreg_train.py dataset_train.csv SGD
```
To train model using stochastic gradient descent. Hyperparamater (epoch) can be adjusted.

### To train model using mini_batch
```
python logreg_train.py dataset_train.csv mini_batch
```
To train model using stochastic gradient descent. Hyperparamater (epoch and batch_size) can be adjusted.
Both models above can be evaluated using mandatory functions as shown above. Precision rate is 99% for both optimization methods.

