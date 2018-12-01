"""Entry point to evolving the neural network. Start here."""
import logging
import numpy as np
from optimizer import Optimizer
from tqdm import tqdm
from train import get_cifar10, get_mnist, load_data, compile_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)



def get_session(gpu_fraction=.3):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


KTF.set_session(get_session())

def train_networks(networks, nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, eras):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, eras)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    total_consistency = 0
    for network in networks:
        total_accuracy += network.accuracy[0]
        total_consistency += network.accuracy[1]
        logging.info("network loss: %.2f, %.2f" % (network.accuracy))

    return total_accuracy / len(networks), total_consistency / len(networks)

def generate(generations, population, nn_param_choices, nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, eras):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, eras)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average loss: %.2f , %2f" % (average_accuracy ))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)
            print_networks(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)

    # Print out the top 5 networks.
    print_networks(networks[:5])
    return networks[:5]

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    """Evolve a network."""
    generations = 20  # Number of times to evole the population.
    population = 30  # Number of networks in each generation.
    dataset = 'numerai'

    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist()
    elif dataset == 'numerai':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test, ids, eras = load_data()



    nn_param_choices = {
        'nb_neurons_l0': [16, 32, 64, 128], #, 256, 512, 768, 1024],
        'nb_neurons_l1': [2, 4, 8, 16, 32, 64], #, 128, 256, 512, 768, 1024],
        'nb_neurons_l2': [2, 4, 8, 16, 32, 64], #, 128, 256, 512, 768, 1024],
        'nb_neurons_l3': [2, 4, 8, 16, 32, 64], #, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'dr0':[0,1],
        'dr1': [0, 1],
        'dr2': [0, 1],
        'dr3': [0, 1],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    networks = generate(generations, population, nn_param_choices, nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, eras)

    logging.info(print_networks(networks))

    model_list = []
    y_prediction = []
    y_final = np.zeros(y_test.shape)
    i=0
    for net in networks:
        net.print_network()
        model_list[i] = compile_model(net.network, nb_classes, input_shape)
        model_list[i].fit(x_train.values, y_train.values, batch_size=batch_size, epochs=10,
                          validation_data=(x_test[:46362].drop('era', axis=1).values, y_test[:46362].drop('era', axis=1).values))
        y_prediction[i] = model_list[i].predict(x_test.drop('era', axis=1).values, batch_size=32)
        y_final +=y_prediction/len(networks)
        i += 1

    scaler = MinMaxScaler(feature_range=(0.30001, 0.699900))
    results = y_final[:, 0]
    results = scaler.fit_transform(results.reshape(-1, 1))
    print(min(results), max(results))
    results_df = pd.DataFrame(data=results, index=ids, columns=['probability_bernie'])
    results_df.to_csv("./rounds/round_125/round_125/predictions_sp bernie_genetic.csv", index=True)
    print('save predictions_sp1 bernie genetic.csv completed you can upload')



if __name__ == '__main__':
    main()
