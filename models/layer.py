class Layer:
    type = None
    algorithm = None
    network = None

    neurons = []
    neurons_results = []

    def __init__(self, type, algorithm, neurons):
        self.type = type
        self.algorithm = algorithm
        self.neurons = neurons

        for neuron in self.neurons:
            neuron.layer = self