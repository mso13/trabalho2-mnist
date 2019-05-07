# Used libraries

import tensorflow as tf 
from MultiLayerPerceptron import MLP
from CustomFunctions import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def main():

	# Load the dataset
	(X_train, y_train), (X_test, y_test) = load_dataset()

	# Normalizes the dataset
	X_train = normalize_dataset(X_train)
	X_test 	= normalize_dataset(X_test)

	# Create the Neural Network Structure

	# 1. Define the number of Neurons and Layers
	neurons_per_layer = [128, 128] # 2 layers, 128 neurons each

	# 2. Define the number of output classes
	output_classes = 10

	# 3. Define the activation function on hidden layers
	# Options:
	#   - 'relu'
	#   - 'tanh'
	#   - 'sigmoid'
	activation_functions = ['relu', 'tanh']

	# 4. Define the activation function on output layer
	# Options:
	#   - 'softmax'
	out_layer_activation = 'softmax'

	# 5. Define the optimizer
	#  Options:
	#   - 'adam'
	#   - SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
	optimizer = 'sgd'

	# 6. Define the loss function
	# Options: 
	#   - 'sparse_categorical_crossentropy'
	#   - 'mean_squared_error'
	#   -  sqr_error
	loss = 'sparse_categorical_crossentropy'

	# 7. Define the learning rate
	#   - Variate the learning rates!
	learning_rate = 0.01

	# Learn and Predict Model

	# 1. Construct the MLP
	mlp = MLP(neurons_per_layer,
			  output_classes,
			  activation_functions,
			  out_layer_activation,
			  optimizer,
			  loss,
			  learning_rate)

	# 2. Fit the model
	mlp.learn(X_train, y_train, epochs=5)

	# 3. Evaluate the model
	test_classes = mlp.model.predict_classes(X_test)
	test_outputs = mlp.predict(X_test)

	print('Accuracy and CM on Training Set:')
	print(accuracy_score(y_train, mlp.model.predict_classes(X_train)))
	print(confusion_matrix(y_train, mlp.model.predict_classes(X_train)))

	print('Accuracy and CM on Test Set:')
	print(accuracy_score(y_test, mlp.model.predict_classes(X_test)))
	print(confusion_matrix(y_test, mlp.model.predict_classes(X_test)))

	# Previous attempt

	# mlp = MLP([60, 10], 0.01)
	# print('non categorical shape and first sample')
	# print(y_train.shape)
	# print(y_train[0])
	# print('categorical shape and first sample')
	# print(to_categorical(y_train).shape)
	# print(to_categorical(y_train)[0])

	# # Converting y_train to categorical will transform the outputs
	# # as a one hot variable (1 for the desired class and 0 for the others)
	# # allowing the network to train each output neuron.
	
	# mlp.learn(x_train, to_categorical(y_train), epochs=10)
	# test_classes  = mlp.model.predict_classes(x_test)
	# test_outputs = mlp.predict(x_test)


def load_dataset():
	mnist = tf.keras.datasets.mnist
	return mnist.load_data()

def normalize_dataset(X):
	return tf.keras.utils.normalize(X)


if __name__ == "__main__":
	main()