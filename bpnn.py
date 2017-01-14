import numpy as np

class BPNeuralNetwork:
    def __init__(self,
            layers, bias = True, bias_value = 1.0,
            min_init_weight_value = -0.5, max_init_weight_value = 0.5,
            learning_rate = 1.0):
        if len(layers) < 3:
            raise ValueError('At least one input, one hidden, and and one output layer is required.')

        self.bias = bias
        self.learning_rate = learning_rate

        self.activations = []
        for i in xrange(len(layers)):
            try:
                units = int(layers[i])
                if units < 1:
                    raise ValueError

                if i + 1 < len(layers):
                    units = units + self.bias

                self.activations.append(np.zeros(units))
            except ValueError:
                raise ValueError('Invalid layer size.')

        if self.bias:
            for i in xrange(len(self.activations) - 1):
                self.activations[i][-1] = bias_value
    
        self.weights = []
        self.weight_corrections = []
        self.weighted_sums = []
        self.errors = []
        for i in xrange(len(self.activations) - 1):
            self.weights.append([])
            self.weight_corrections.append([])
            self.errors.append(np.zeros(len(self.activations[i + 1])))
            self.weighted_sums.append(np.zeros(len(self.activations[i + 1])))

            for j in xrange(len(self.activations[i + 1])):
                self.weights[i].append(np.random.uniform(min_init_weight_value, max_init_weight_value, len(self.activations[i])))
                self.weight_corrections[i].append(np.zeros(len(self.activations[i])))
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        sigmoid_input = self.sigmoid(x)
        return sigmoid_input * (1.0 - sigmoid_input)
    
    def predict(self, inputs):
        if len(inputs) != len(self.activations[0]) - self.bias:
            raise ValueError('Invalid number of inputs.')

        for i in xrange(len(self.activations[0]) - self.bias):
            self.activations[0][i] = inputs[i]

        for i in xrange(len(self.activations) - 1):
            for j in xrange(len(self.activations[i + 1])):
                self.weighted_sums[i][j] = 0.0

                for k in xrange(len(self.activations[i])):
                    self.weighted_sums[i][j] = self.weighted_sums[i][j] + self.activations[i][k] * self.weights[i][j][k]
                    
                if not self.bias or (self.bias and (j + 1 != len(self.activations[i + 1]) or i + 1 == len(self.activations) - 1)):
                    self.activations[i + 1][j] = self.sigmoid(self.weighted_sums[i][j])
                
    def train(self, inputs, outputs):
        self.predict(inputs)

        if len(outputs) != len(self.activations[-1]):
            raise ValueError('Invalid number of outputs.')

        for i in xrange(len(outputs)):
            self.errors[-1][i] = (outputs[i] - self.activations[-1][i]) * self.sigmoid_prime(self.weighted_sums[-1][i])
            for j in xrange(len(self.activations[-2])):
                self.weight_corrections[-1][i][j] = self.learning_rate * self.errors[-1][i] * self.activations[-2][j]

        for i in xrange(len(self.activations) - 2):
            for j in xrange(len(self.activations[-(2 + i)])):
                delta_input_sum = 0.0
                for k in xrange(len(self.activations[-(1 + i)])):
                    delta_input_sum = delta_input_sum + self.errors[-(1 + i)][k] * self.weights[-(1 + i)][k][j]

                self.errors[-(2 + i)][j] = delta_input_sum * self.sigmoid_prime(self.weighted_sums[-(2 + i)][j])

                for k in xrange(len(self.activations[-(3 + i)])):
                    self.weight_corrections[-(2 + i)][j][k] = self.learning_rate * self.errors[-(2 + i)][j] * self.activations[-(3 + i)][k]

        for i in xrange(len(self.activations) - 1):
            for j in xrange(len(self.activations[i + 1])):
                self.weights[i][j] = np.add(self.weights[i][j], self.weight_corrections[i][j])

        return ((outputs - self.activations[-1]) ** 2).mean(axis=0)
