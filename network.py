"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = (0., 0.) # accuracy, consistency
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            if key in ['dr0', 'dr1', 'dr2', 'dr3']:
                self.network[key] = random.uniform(self.nn_param_choices[key][0], self.nn_param_choices[key][1])
            else:
                self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, eras):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy[0] == 0.:
            self.accuracy = train_and_score(self.network, nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test,eras)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network loss criteria & consistency: %.2f , %2f" % (self.accuracy ))
