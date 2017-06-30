from algorithmes import simple as simple_algorithm, sigmoid as sigmoid_algorithm
from models import layer, neuron, network

# Calculate weights
dataset = [[2.7810836, 2.550537003, 3.7810836, 3.550537003, 0],
           [1.465489372, 2.362125076, 2.465489372, 3.362125076, 0],
           [3.396561688, 4.400293529, 4.396561688, 5.400293529, 0],
           [1.38807019, 1.850220317, 2.38807019, 2.850220317, 0],
           [3.06407232, 3.005305973, 4.06407232, 4.005305973, 0],
           [7.627531214, 2.759262235, 8.627531214, 3.759262235, 1],
           [5.332441248, 2.088626775, 6.332441248, 3.088626775, 1],
           [6.922596716, 1.77106367, 7.922596716, 2.77106367, 1],
           [8.675418651, -0.242068655, 9.675418651, 0.242068655, 1],
           [7.673756466, 3.508563011, 8.673756466, 4.508563011, 1],
           [8.675418651, -0.242068655, 9.675418651, 0.242068655, 1],
           [7.673756466, 3.508563011, 8.673756466, 4.508563011, 1]]

my_network = network.Network([
    layer.Layer('input', sigmoid_algorithm, [neuron.Neuron(), neuron.Neuron()]),
    layer.Layer('hidden', sigmoid_algorithm, [neuron.Neuron(), neuron.Neuron(), neuron.Neuron()]),
    layer.Layer('output', sigmoid_algorithm, [neuron.Neuron()])
])

data = list(data_item[:-1] for data_item in dataset)
sample = list(sample_item[-1:] for sample_item in dataset)

l_rate = 1
n_epoch = 4300
my_network.train(data, sample, l_rate, n_epoch)