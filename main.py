import numpy as np
import scipy.special


class NeuralNetwork:

    def __init__(self, inputnodes: int, hiddennodes: int, outputnodes: int, learningrate: float):
        print('initializing neural network ...')
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        # weight matrices
        # weights insides the arrays are w_i_j where link is from node i to node j in the next layer
        # wih <- input to hidden
        # who <- hidden to output
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        targets = np.array(targets_list, ndmin=2).T
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # BACKPROPAGATION:
        # calculate errors
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # update weights
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))
        pass

    def query(self, inputs_list):
        print('processing query ...')
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':
    print('starting programm ...')
    n = NeuralNetwork(3, 3, 3, 0.3)


