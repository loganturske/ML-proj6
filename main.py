from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import sys

#
# This function will load in a CSV file
#
def load_csv(filename):
	# Create and empty list
	dataset = list()
	# Open the file to read
	with open(filename, 'r') as file:
		# Create a reader of the file
		csv_reader = reader(file)
		# For each row in the the file read in
		for row in csv_reader:
			# If there is an empty row
			if not row:
				# Skip
				continue
			# Add row the the dataset list
			dataset.append(row)
	# Return the dataset that you created
	return dataset
 

#
# This function will convert strings in a column to floats
#
def str_column_to_float(dataset, column):
	# For each row in the dataset
	for row in dataset:
		# Convert the string in that particular column of the row to a float
		row[column] = float(row[column].strip())
 

# 
# This function will convert strings in a column to integers
#
def str_column_to_int(dataset, column):
	# Create a column of all the classes
	class_values = [row[column] for row in dataset]
	# Get a list of the all the unique classes
	unique = set(class_values)
	# Create an empty dictonary
	lookup = dict()
	# For each of the unique classes
	for i, value in enumerate(unique):
		# Set the index of the class to the index in the dictonary
		lookup[value] = i
	# For each row in the data set
	for row in dataset:
		# Set the column of that row to be the value in the lookup dictonary
		row[column] = lookup[row[column]]
	# Return the dictonary
	return lookup
 

#
# This function will find the min and max values for each column
#
def dataset_minmax(dataset):
	# Create an empty list
	minmax = list()
	# Create a list of the minimum and maximum values for each column
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	# Return the miniums ans maximums of the columns
	return stats
 

#
# This function will escale dataset columns to the range 0-1 (i.e Normalize the data)
#
def normalize_dataset(dataset, minmax):
	# For each row in the dataset
	for row in dataset:
		# For each of the feature values in the row
		for i in range(len(row)-1):
			# Normalize the data based on the minmax list passed in
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

 

#
# This function will calculate the accuracy percentage
#
def accuracy_metric(actual, predicted):
	# Set a running total of the number of correct guesses
	correct = 0
	# For each of the data rows that were guessed
	for i in range(len(actual)):
		# If you got a correct guess
		if actual[i] == predicted[i]:
			# Increment the running correct counter by 1
			correct += 1
	# Calculate the accuracy and return it
	return correct / float(len(actual)) * 100.0


#
# This function will initialize a network
#
def initialize_network(n_inputs, n_hidden, n_outputs):
	# Create an empty list
	network = list()
	# Create the hidden layers by initializing random weights for each layer
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	# Append all the hidden layers to the network
	network.append(hidden_layer)
	# Create an output layer for each outputs you expect by added random weights for each output
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	# Add the output layer to the network
	network.append(output_layer)
	# Return the network that was created
	return network

 
#
# This function will calculate neuron activation for the input
#
def activate(weights, inputs):
	# Get the last column of the weights list
	activation = weights[-1]
	# For the index of the weights list
	for i in range(len(weights)-1):
		# Add the weight and the value of the input multiplied together
		activation += weights[i] * inputs[i]
	# Return the activation value
	return activation
 

#
# This function will transfer neuron activation
#
def transfer(activation):
	# Preform the sigmoid function
	return 1.0 / (1.0 + exp(-activation))


#
# This function will calculate the derivative of an neuron output
#
def transfer_derivative(output):
	# Return the derivative of the neuron output passed in
	return output * (1.0 - output)


#
# This function will forward propagate the input to a network output
#
def forward_propagate(network, row):
	# The inputs will be the row passed in, get a reference to it
	inputs = row
	# For each layer in the network
	for layer in network:
		# Create and empty list
		new_inputs = []
		# For each neuron in the layer
		for neuron in layer:
			# Preform the activation function by passing in the weights and inputs
			activation = activate(neuron['weights'], inputs)
			# Set the output of the neuron to be the activation value put through a sigmoid
			neuron['output'] = transfer(activation)
			# Add the neuron output to the new input list
			new_inputs.append(neuron['output'])
		# Set the inputs to be the new inputs
		inputs = new_inputs
	# Return the new inputs
	return inputs


#
# This functino will update thenetwork weights with error
#
def update_weights(network, row, l_rate):
	# For each layer in the network
	for i in range(len(network)):
		# Get the value of the last column in the row passed in
		inputs = row[:-1]
		# If you are not the first layer
		if i != 0:
			# The inputs will be a list for each neuron output in the layer before
			inputs = [neuron['output'] for neuron in network[i - 1]]
		# For each neuron in this layer
		for neuron in network[i]:
			# For each input
			for j in range(len(inputs)):
				# Calculate the weights of the neuron by
				# leanring rate * delta * the inputs
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			# The last weight will be the learning rat multiplied by the delta
			neuron['weights'][-1] += l_rate * neuron['delta']


#
# This function will train a network for a fixed number of epochs
#
def train_network(network, train, l_rate, n_epoch, n_outputs):
	# For each epoch
	for epoch in range(n_epoch):
		# Set a running error
		sum_error = 0
		# For each row in the training set
		for row in train:
			# Forward propagate the row in the newtork
			outputs = forward_propagate(network, row)
			# Set a list of 0s for each output
			expected = [0 for i in range(n_outputs)]
			# For the class of this row, set the expected output to be a 1
			expected[row[-1]] = 1
			# Calculate the running error
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			# Preform backwards propogation error on the network given the expected
			backward_propagate_error(network, expected)
			# Update the weights based on the row
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


#
# This function will make a prediction with the network
#
def predict(network, row):
	# Get the ouputs of the network for the row forwared propagated through the network
	outputs = forward_propagate(network, row)
	# Return the largest of the outputs
	return outputs.index(max(outputs))
 

#
# This function will use backpropagate error and store in neurons
#
def backward_propagate_error(network, expected):
	# For each layer in the network
	for i in reversed(range(len(network))):
		# Get a reference to the layer
		layer = network[i]
		# Create an error list that is empyt
		errors = list()
		# If you are not the last layer (i.e output layer)
		if i != len(network)-1:
			# For each neuron in the layer
			for j in range(len(layer)):
				# Set a error to be 0
				error = 0.0
				# For each neuron in the next layer
				for neuron in network[i + 1]:
					# Add the calculated error to the running error
					# weight * delta
					error += (neuron['weights'][j] * neuron['delta'])
				# Add the error calculated to the running errors list
				errors.append(error)
		# If you are on the last layer of the network
		else:
			# For each neuron in the layer
			for j in range(len(layer)):
				# Get a reference to the neuron
				neuron = layer[j]
				# Add to the running erros list
				# the expectied value subtracted by the output value
				errors.append(expected[j] - neuron['output'])
		# for each neuron in the layer
		for j in range(len(layer)):
			# Get a reference to the neuron
			neuron = layer[j]
			# Set the delta of the neuron to be the error value of that neuron multiplied
			# by the derivative of the transfer
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
			

#
# This is the backpropagation algorithm with stochastic gradient descent
#
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	# The number of inputs if the length of the training set minus 1
	n_inputs = len(train[0]) - 1
	# The number of outputs is the length of the set of all the classes in the training set
	n_outputs = len(set([row[-1] for row in train]))
	# Initialize the network
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	# Train the network on the training set
	train_network(network, train, l_rate, n_epoch, n_outputs)
	# Create an empty list to house all the predictions
	predictions = list()
	# For each row in the test set
	for row in test:
		# Get a prediction of that row
		prediction = predict(network, row)
		# Add the predition to the prediction list
		predictions.append(prediction)
	# Return the list of predictions
	return(predictions)
 

#
# This function will split a dataset into k folds for cross validation
#
def cross_validation_split(dataset, n_folds):
	# Create an empty list that will house the folds
	dataset_split = list()
	# Create a copy of the dataset as a list
	dataset_copy = list(dataset)
	# Get the number of entries for each fold
	fold_size = int(len(dataset) / n_folds)
	# For each fold
	for i in range(n_folds):
		# Create an empty list
		fold = list()
		# While you still need to populate the fold
		while len(fold) < fold_size:
			# Get a random value as an index of the dataset
			index = randrange(len(dataset_copy))
			# Add the random row to the fold
			fold.append(dataset_copy.pop(index))
		# Add the fold to the dataset that will house the folds
		dataset_split.append(fold)
	# Return the dataset list that houses the folds
	return dataset_split
 

#
# This function will evaluate the algorithm using cross validation
#
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	# Get all of the folds to use for cross validation
	folds = cross_validation_split(dataset, n_folds)
	# Create an empty list to house all of the scores
	scores = list()
	# For each fold
	for fold in folds:
		# Get a reference to the training set by making a list out of the folds
		train_set = list(folds)
		# Remove the fold that you are on
		train_set.remove(fold)
		# Make the training sets have a list on them
		train_set = sum(train_set, [])
		# Create an empty list 
		test_set = list()
		# For each row in the fold
		for row in fold:
			# Make a list of the row
			row_copy = list(row)
			# Add the row to the test set
			test_set.append(row_copy)
			# Set the last column to None
			row_copy[-1] = None
		# Use the algorithm that was passed in on the training set and the test set
		predicted = algorithm(train_set, test_set, *args)
		# Get a list of the actual classes for each row in the fold
		actual = [row[-1] for row in fold]
		# Get the accuracy of what you guessed vs the actual classes
		accuracy = accuracy_metric(actual, predicted)
		# Add the score for this fold to the accuracy list
		scores.append(accuracy)
	# Return the scores for each fold
	return scores


#
# This is the main function of the program
#
if __name__ == "__main__":
	# Get the filename of the dataset
	filename = sys.argv[1]
	# Load the file into the dataset variable
	dataset = load_csv(filename)
	# For each column in the dataset but the last
	for i in range(len(dataset[0])-1):
		# Turn the values into floats
		str_column_to_float(dataset, i)
	# Convert the last column of the dataset into an integer
	str_column_to_int(dataset, len(dataset[0])-1)
	# Get the minimums and maximums for each of the feature columns
	minmax = dataset_minmax(dataset)
	# Normalize the dataset
	normalize_dataset(dataset, minmax)
	# evaluate algorithm
	n_folds = 5
	l_rate = 0.3
	n_epoch = 500
	n_hidden = 5
	# Run the algo
	scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))