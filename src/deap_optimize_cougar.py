
##############################################################################################################
#                                                                                                            #
#    File: deap_optimize_cougar.py                                                                           #
#    Date: May 23, 2020                                                                                      #
#    Purpose:                                                                                                #
#      This file contains functionality to evolve clustering parameters for 2D data projections.             #
#                                                                                                            #
#    Copyright (c) 2020 Zachary Wilkins                                                                      #
#    COUGAR is licensed under the MIT Licence.                                                               #
#                                                                                                            #
##############################################################################################################

from datetime import datetime
import logging as lg
from math import factorial
from os import mkdir, path, environ
import pickle
import random

from deap import creator, base, tools, algorithms
import numpy
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D  # Necessary for plotting 3D graphs
import pandas as pd
from scoop import futures

from cougar import Cougar, RANDOM_STATE

WORKING_DIR = '../Cougar_Output/Run_{}'.format(datetime.now().strftime('%b-%d-%Y-%H-%M'))
CHECKPOINT_FREQUENCY = 5
DEFAULT_CLUSTERING_PARAMS = {
    # eps (Scaled out of 100), min_samples (Scaled out of 1000)
    'dbscan': (0.005, 0.04),
    # min_samples (Scaled out of 1000), max_eps (Scaled out of 100)
    'optics': (0.04, 0.5),
    # n_clusters (Scaled out of 100), n_init (Scaled out of 100), max_iter (Scaled out of 1000)
    'kmeans': (0.08, 0.1, 0.3)
}
# Lower bounds for each parameter, then upper bounds, respectively
DEFAULT_CLUSTERING_BOUNDS = {
    # eps (Scaled out of 100), min_samples (Scaled out of 1000)
    'dbscan': ([0.0, 0.005], [1.0, 0.25]),
    # min_samples (Scaled out of 1000), max_eps (Scaled out of 100)
    'optics': ([0.005, 0.0], [0.25, 1.0]),
    # n_clusters (Scaled out of 100), n_init (Scaled out of 100), max_iter (Scaled out of 1000)
    'kmeans': ([0.02, 0.01, 0.01], [1.0, 1.0, 1.0])
}
CLUSTERING_ALGORITHM = environ['CLUSTERING_ALGORITHM']
CHECKPOINT_FILE = environ.get('CHECKPOINT_FILE')
DATASET_SIZE = int(environ.get('DATASET_SIZE'))
HOLDOUT_PRESENT = int(environ.get('HOLDOUT_PRESENT'))

"""
Objectives:
 - Create as many good clusters as possible (maximize number of clusters with quality >80%)
 - Try not to put wrong points in a cluster (minimize sum of the sum squared error for each cluster)
 - Create the biggest clusters possible (maximize median cluster sample count)
"""
NUMBER_OF_OBJECTIVES = 3
"""
P: Partitions between the hyperplane points
-------------------------------------------
Deb suggests:
 - P=12 when NUMBER_OF_OBJECTIVES=3 -> H=91
But that takes a reaaaaaaally long time because the population is huge.
So P=4 -> H=15 is much more reasonable!
"""
P = 4
""" From Deb: H = ( M + P - 1 ) choose ( P ) """
H = factorial(NUMBER_OF_OBJECTIVES + P - 1) / (factorial(P) * factorial(NUMBER_OF_OBJECTIVES - 1))
""" Previously known as NDIM, number of dimensions in an individual """
INDIVIDUAL_LENGTH = len(DEFAULT_CLUSTERING_PARAMS[CLUSTERING_ALGORITHM])
""" From Deb: smallest multiple of 4 greater than H """
POPULATION_SIZE = int(H + (4 - H % 4))
NUMBER_OF_GENERATIONS = 100
CROSSOVER_PROBABILITY = 1.0
MUTATION_PROBABILITY = 1.0
IND_MUTATION_PROBABILITY = 0.25


def build_cluster_params(individual: list) -> dict:
    if CLUSTERING_ALGORITHM == 'dbscan':
        return {
            'eps': individual[0] * 100,
            'min_samples': int(individual[1] * 1000)
        }
    elif CLUSTERING_ALGORITHM == 'optics':
        return {
            'min_samples': int(individual[0] * 1000),
            'max_eps': individual[1] * 100
        }
    elif CLUSTERING_ALGORITHM == 'kmeans':
        return {
            # Mitigate n_*=0 errors
            'n_clusters': max(int(individual[0] * 100), 1),
            'n_init': max(int(individual[1] * 100), 1),
            'max_iter': int(individual[2] * 1000)
        }
    else:
        lg.error(f'Unknown clustering algorithm: {CLUSTERING_ALGORITHM}')
        raise ValueError('Unknown clustering algorithm')


def eval_cougar(individual: list, index: int = None) -> list:
    """
    Evaluate a COUGAR individual, returning a variety of metrics.

    :param individual: a list of floats, representing clustering algorithm parameters
    :param index: an int index which, if present, prompts COUGAR to save output. This is typically
    employed on the last generation, to save the resulting individuals.
    :return: a list containing: the number of clusters, the total sum of the sum-squared errors of each cluster,
    and the median cluster count obtained using the passed individual
    """
    cougar_instance = Cougar(
        dataset=2,
        location='../EMBER/Parquet',
        working=WORKING_DIR,
        reduction='umap',
        params1={'n_neighbors': int(DATASET_SIZE * 0.01), 'min_dist': 0.1},
        cluster=CLUSTERING_ALGORITHM,
        params2=build_cluster_params(individual),
        save=index is not None
    )
    if HOLDOUT_PRESENT:
        # Need to preserve the random state to ensure embedding is consistent when holdout data is evaluated
        cougar_instance.reduction_params['random_state'] = RANDOM_STATE
    cougar_instance.compute_md5_params()
    lg.debug('Loading vectorization...')
    cougar_instance.vectorize(load_from_disk=True)
    lg.debug('Loading dimension reduction...')
    cougar_instance.load_prebuilt_model(is_reduce=True)

    cougar_instance.cluster()

    return [*cougar_instance.evaluate(index)]


def cougar_individual() -> list:
    default = DEFAULT_CLUSTERING_PARAMS[CLUSTERING_ALGORITHM]
    ind = list()
    for param in default:
        low = param - param * random.random()
        high = param + param * random.random()
        ind.append(random.uniform(low, high))
    return ind


# Must define these values, and functions above, at the root level of the script to
# ensure that SCOOP workers can access them.
creator.create('FitnessMaxMinMax', base.Fitness, weights=(1.0, -1.0, 1.0))
creator.create('Individual', list, fitness=creator.FitnessMaxMinMax)
ref_points = tools.uniform_reference_points(NUMBER_OF_OBJECTIVES, P)

toolbox = base.Toolbox()
toolbox.register('attr_cougar', cougar_individual)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_cougar)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('evaluate', eval_cougar)
BOUND_LOW = DEFAULT_CLUSTERING_BOUNDS[CLUSTERING_ALGORITHM][0]
BOUND_UP = DEFAULT_CLUSTERING_BOUNDS[CLUSTERING_ALGORITHM][1]
toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register('mutate', tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=IND_MUTATION_PROBABILITY)
toolbox.register('select', tools.selNSGA3, ref_points=ref_points)
# This registration enables parallelization via SCOOP. Comment out to run in sequence.
toolbox.register('map', futures.map)

random.seed(None)


def plot_2d_fitness(gen, y_axis_data, y_axis_label, y_line_label, x_axis_label='Generation', filename='deap_fitness'):
    plt.grid(True)
    plt.plot(gen, y_axis_data, '-b', label=y_line_label)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(loc='upper left')
    plt.savefig(f'{WORKING_DIR}/{filename}.png')
    plt.close()


def plot_3d_fitness(x_data, y_data, z_data, x_label, y_label, z_label):
    fig = plt.figure()
    # One row, one column, starting at index 1
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_data, y_data, z_data, label='Avg. Fitness')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    # Plot beginning and end points
    ax.scatter(x_data[:1], y_data[:1], z_data[:1], s=40, c='g', marker='o')
    ax.scatter(x_data[-1:], y_data[-1:], z_data[-1:], s=40, c='r', marker='*')
    ax.legend()
    plt.savefig(f'{WORKING_DIR}/deap_avg_all_fitness.png')
    plt.close()


def main(checkpoint=None):
    if checkpoint:
        with open(checkpoint, 'rb') as f:
            checkpoint_data = pickle.load(f)
        logbook = checkpoint_data['logbook']
        pop = checkpoint_data['population']
        start_generation = checkpoint_data['generation']
        random.setstate(checkpoint_data['random_state'])
    else:
        logbook = tools.Logbook()
        logbook.header = 'gen', 'evals', 'std', 'min', 'avg', 'max'
        pop = toolbox.population(n=POPULATION_SIZE)
        start_generation = 1

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', numpy.mean, axis=0)
    stats.register('std', numpy.std, axis=0)
    stats.register('min', numpy.min, axis=0)
    stats.register('max', numpy.max, axis=0)

    if not checkpoint:
        # Compile statistics about the population
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)
        lg.info('Beginning evolutionary process!')
    else:
        lg.info(f'Resuming evolutionary process from Generation {start_generation}!')

    # Begin the generational process
    for gen in range(start_generation, NUMBER_OF_GENERATIONS + 1):
        offspring = algorithms.varAnd(pop, toolbox, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # Save the last generation
        if gen == NUMBER_OF_GENERATIONS:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, range(len(invalid_ind)))
        else:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, POPULATION_SIZE)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        # print(pop)
        print(logbook.stream)

        if gen % CHECKPOINT_FREQUENCY == 0:
            checkpoint_data = dict(logbook=logbook, population=pop, generation=gen, random_state=random.getstate())
            with open(f'{WORKING_DIR}/checkpoint.pkl', 'wb') as f:
                pickle.dump(checkpoint_data, f)

    lg.info('Finished evolutionary process!')

    logbook_df = pd.DataFrame.from_records(logbook)
    for column in ['std', 'min', 'avg', 'max']:
        columns = [f'{column}-{x}' for x in range(NUMBER_OF_OBJECTIVES)]
        logbook_df[columns] = pd.DataFrame(logbook_df[column].tolist(), index=logbook_df.index)
    logbook_df.drop(['std', 'min', 'avg', 'max'], axis=1, inplace=True)
    with open(f'{WORKING_DIR}/logbook.csv', 'w') as f:
        logbook_df.to_csv(index=False, path_or_buf=f)

    gen = logbook.select('gen')

    # Unpack fitness of objectives into separate lists
    avg_fit_good_cluster, avg_fit_error, avg_fit_median_cluster_size = list(
        zip(*[tuple(gen_record['avg']) for gen_record in logbook])
    )

    plot_2d_fitness(
        gen=gen,
        y_axis_data=avg_fit_good_cluster,
        y_axis_label='Avg. Number of Good Clusters',
        y_line_label='Avg. No. Good Clusters',
        filename='deap_avg_fit_cluster_count'
    )
    plot_2d_fitness(
        gen=gen,
        y_axis_data=avg_fit_error,
        y_axis_label='Avg. Sum of the SSE',
        y_line_label='Avg. Scaled SSSE',
        filename='deap_avg_fit_error'
    )
    plot_2d_fitness(
        gen=gen,
        y_axis_data=avg_fit_median_cluster_size,
        y_axis_label='Avg. Median Cluster Size',
        y_line_label='Avg. Med. Cluster Size',
        filename='deap_avg_fit_cluster_size'
    )
    plot_3d_fitness(
        avg_fit_good_cluster, avg_fit_error, avg_fit_median_cluster_size,
        'Avg. Number of Good Clusters', 'Avg. Sum of the SSE', 'Avg. Median Cluster Size'
    )


if __name__ == '__main__':
    if not path.exists(WORKING_DIR):
        mkdir(WORKING_DIR)
    lg.basicConfig(
        format='[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=lg.INFO
    )
    lg.info(f'Using {CLUSTERING_ALGORITHM} as clustering algorithm')
    main(checkpoint=CHECKPOINT_FILE)
