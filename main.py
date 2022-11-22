'''
Neural Network to train on minst data set
using 'Neuronale Netze selbst programmieren', O'reilly, 2017
'''

import numpy as np
import scipy.special
import matplotlib.pyplot as plt


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

    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load mnist data set
    training_data_file = open('mnist_train.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # go through all the records in training data set
    print('training ...')
    for record in training_data_list:
        all_values = training_data_list[0].split(',')
        image_array = np.asfarray(all_values[1:]).reshape((28, 28))
        # display current picture:
        '''plt.imshow(image_array, cmap='Greys', interpolation='None')
        plt.show()'''
        scaled_input = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01

        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(scaled_input, targets)
        pass
    print('finished training')
