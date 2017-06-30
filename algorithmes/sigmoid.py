import numpy as np

def predict(inputs, weights):
	activation = weights[0]
	for i in range(len(inputs)-1):
		activation += weights[i + 1] * inputs[i]
	return sigmoid(activation)

def sigmoid (x):
	return 1/(1 + np.exp(-x))

def sigmoid_ (x):
	return x * (1 - x)