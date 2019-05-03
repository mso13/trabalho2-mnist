import tensorflow as tf

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
			self.model.add(tf.keras.layers.Dense(neurons, activation=get_activation(act)))

		# Output Layers
		self.model.add(tf.keras.layers.Dense(output_classes,
			activation=get_activation(out_layer_activation)))

        # Build Model
		self.model.compile(optimizer=get_optmizer(optimizer), loss=get_loss(), metrics=['accuracy'])
    
	def learn(self, X, y, epochs):
		self.model.fit(X, y, epochs=epochs)

	def predict(self, X_new, batch_size=100):
		predictions = self.model.predict(X_new, batch_size=batch_size)
		return predictions

	def get_activation(self, activation_function):

		if activation_function == 'relu':
			act = tf.nn.relu

		if activation_function == 'softmax':
			act = tf.nn.softmax

		return act

	def get_optmizer(self, optimizer_function):

		if optimizer_function == 'adam':
			opt = 'adam'

		return opt

	def get_loss(self, loss_function):

		if loss_function == 'sparse_categorical_crossentropy':
			loss = 'sparse_categorical_crossentropy'

		return loss