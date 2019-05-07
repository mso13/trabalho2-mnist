# Used libraries

import tensorflow as tf
from tensorflow.keras.optimizers import SGD

# Multilayer Perceptron Model
class MLP:
	def __init__(self, 
				neurons_per_layer,
				output_classes,
				activation_functions,
				out_layer_activation,
				optimizer,
				loss,
				learning_rate):

		# Feed-Forward Neural Network Model
		self.model = tf.keras.models.Sequential()

		# Flatten the 28x28 images provided
		self.model.add(tf.keras.layers.Flatten())

		# Hidden Layers
		for neurons, act in zip(neurons_per_layer, activation_functions):
			self.model.add(tf.keras.layers.Dense(neurons, activation=act))

		# Output Layer
		self.model.add(tf.keras.layers.Dense(output_classes,
			activation=out_layer_activation))

        # Build Model
		self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
	def learn(self, X, y, epochs):
		self.model.fit(X, y, epochs=epochs)

	def predict(self, X_new, batch_size=100):
		predictions = self.model.predict(X_new, batch_size=batch_size)
		return predictions