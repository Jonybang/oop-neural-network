class Neuron:
    id = None
    inputs = []
    results = []
    weights = []
    layer = None

    def predict(self, inputs):
        if len(self.weights) == 0:
            self.initWeights(len(inputs))

        if self.id == None:
            self.id = self.layer.network.getIdForNewNeuron()

        self.inputs = inputs
        self.results = self.layer.algorithm.predict(inputs, self.weights)
        return self.results


    def initWeights(self, inputs_count):
        self.weights = [0.0 for i in range(inputs_count + 1)]
