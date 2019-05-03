import tensorflow as tf 
from NeuralNetworks import MLP

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
	activation_functions = ['relu', 'relu']

	# 4. Define the activation function on output layer
	out_layer_activation = ['softmax']

	# 5. Define the optimizer
	optimizer = 'adam'

	# 6. Define the loss function
	loss = 'sparse_categorical_crossentropy'

	# Define the learning rate
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
	mlp.learn(X_train, y_train, epochs=10)

	# 3. Evaluate the model
	test_classes  	= mlp.model.predict_classes(X_test)
	test_outputs 	= mlp.predict(X_test)

	print('Accuracy and CM on Training Set:')
	print(accuracy_score(y_train, mlp.model.predict_classes(X_train)))
	print(confusion_matrix(y_train, mlp.model.predict_classes(X_train)))

	print('Accuracy and CM on Test Set:')
	print(accuracy_score(y_test, mlp.model.predict_classes(X_test)))
	print(confusion_matrix(y_test, mlp.model.predict_classes(X_test)))


def load_dataset():
	mnist = tf.keras.datasets.mnist
	return mnist.load_data()

def normalize_dataset(X):
	return tf.keras.utils.normalize(X)


if __name__ == "__main__":
	main()