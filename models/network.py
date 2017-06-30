from algorithmes import sigmoid as sigmoid_algorithm

class Network:
    input_data = []
    sample_data = []
    layers = []

    last_neuron_id = 0

    def __init__(self, layers):
        self.layers = layers
        for layer in self.layers:
            layer.network = self

    def getIdForNewNeuron(self):
        self.last_neuron_id += 1
        return self.last_neuron_id

    def train(self, input_data, sample_data, l_rate, n_epoch):
        self.input_data = input_data
        self.sample_data = sample_data

        for epoch in range(n_epoch):
            sum_delta = 0.0
            for row_index, row in enumerate(input_data):

                inputs = row

                for layer in self.layers:

                    layer.neurons_results = []
                    for neuron_index, neuron in enumerate(layer.neurons):
                        layer.neurons_results.append(neuron.predict(inputs))

                    inputs = layer.neurons_results

                delta = self.delta(sample_data[row_index][0])

                sum_delta += delta ** 2
                for layer in self.layers:

                    for neuron in layer.neurons:
                        neuron.weights[0] = neuron.weights[0] + l_rate * delta
                        for i in range(len(neuron.inputs)):
                            neuron.weights[i + 1] += l_rate * delta * neuron.inputs[i]

            print('>epoch=%d, lrate=%.3f, delta=%.3f' % (epoch, l_rate, sum_delta))

    def delta(self, sample):
        return sigmoid_algorithm.sigmoid_(sample - self.layers[-1].neurons_results[0])