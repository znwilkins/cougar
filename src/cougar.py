
##############################################################################################################
#                                                                                                            #
#    File: cougar.py                                                                                         #
#    Date: May 25, 2020                                                                                      #
#    Purpose:                                                                                                #
#      This file contains functionality to reduce high dimensional data and cluster it.                      #
#                                                                                                            #
#    Copyright (c) 2020 Zachary Wilkins                                                                      #
#    COUGAR is licensed under the MIT Licence.                                                               #
#                                                                                                            #
##############################################################################################################

from argparse import ArgumentParser
from hashlib import md5
from json import loads, dumps
import logging as lg
import pickle
import sqlite3
from datetime import datetime
from os import mkdir, path, stat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.sql import DatabaseError
from scipy import sparse
from scipy.special import comb
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from umap import UMAP

SQLITE_FILE_NAME = 'cougar_db.sqlite3'
SQLITE_MD5_TABLE = 'md5'
HOLDOUT_SUFFIX = '_holdout'
SQLITE_MD5_TABLE_HOLDOUT = SQLITE_MD5_TABLE + HOLDOUT_SUFFIX
VECTORIZATION_FILE = 'vectorized.npz'
REDUCER_PICKLE_FILE = 'reducer.pkl'
RANDOM_STATE = 2112  # We've taken care of everything, the words you read, the songs you sing!
QUALITY_THRESHOLD = 0.8
UTF_8 = 'UTF-8'
intro = [
    'This is COUGAR!',
    'Clustering Of Unknown malware using Genetic Algorithm Routines',
    '',
    'Script last modified: {}'.format(datetime.fromtimestamp(stat(__file__).st_mtime).strftime('%Y-%m-%d %H:%M:%S')),
]
summary = [
    'Clustering Result #{} Summary',
    '===',
    'Quality Cluster Count: {} / {}',
    'Error: {}',
    'Median Cluster Sample Count: {}',
    '===',
    '{} Reduction Parameters: {}',
    '{} Clustering Parameters: {}\n\n'
]


def main(cougar_instance):
    lg.info('\n' + '\n'.join('### {:^72} ###'.format(s) for s in intro))
    cougar_instance.vectorize()
    cougar_instance.reduce()
    cougar_instance.cluster()
    cougar_instance.evaluate()


class Cougar:
    REDUCTION_ALGORITHMS = {
        'umap': UMAP
    }
    CLUSTER_ALGORITHMS = {
        'dbscan': DBSCAN,
        'optics': OPTICS,
        'kmeans': KMeans
    }
    info_df = None
    imports_df = None
    vectorization = None

    def __init__(self, dataset: int, location: str, working: str, reduction: str, params1: dict, cluster: str,
                 params2: dict, save: bool = False):
        """
        :param dataset: training dataset partition, an int in the range [0, 5]
        :param location: Parquet files directory
        :param working: working directory to store output
        :param reduction: dimension reduction algorithm, one of ['umap']
        :param params1: dimension reduction parameters as dict
        :param cluster: clustering algorithm, one of ['dbscan', 'optics', 'kmeans']
        :param params2: clustering parameters as dict
        :param save: save output to disk
        """
        self.dataset = (
            f'{location}/train_features_{dataset}_2018_info.parquet',
            f'{location}/train_features_{dataset}_2018_imports.parquet'
        )
        self.working = working
        self.reduction_algorithm = Cougar.REDUCTION_ALGORITHMS[reduction]
        self.reduction_params = params1
        self.reduction_params_md5 = None
        self.cluster_algorithm = Cougar.CLUSTER_ALGORITHMS[cluster]
        self.clustering_params = params2
        self.clustering_params_md5 = None
        self.embedding = None
        self.md5s = list()
        self.md5s_holdout = None
        self.labels = None
        self.save = save
        if self.save and not path.exists(self.working):
            mkdir(self.working)

    @staticmethod
    def load_parquet(dataset: tuple):
        """
        Load Parquet files from a specified location to be used by all COUGAR instances.

        :param dataset: the paths of the info and import Parquet files
        """
        Cougar.info_df = pd.read_parquet(dataset[0])
        Cougar.info_df.num_imports = Cougar.info_df.num_imports.astype(np.int32)
        Cougar.imports_df = pd.read_parquet(dataset[1])

    def vectorize(self, load_from_disk: bool = False, subset: int = None, holdout: int = None):
        """
        Vectorize features of the dataset for further processing. The vectorized
        DataFrame is stored as a static class member, so multiple Cougar instances can share the data.

        :param load_from_disk: a bool indicating whether the vectorization should be loaded or generated
        :param subset: an int specifying the number of malware to choose from the dataset
        :param holdout: an int specifying the number of malware to choose as a holdout (testing) dataset
        """
        if Cougar.info_df is None:
            Cougar.load_parquet(self.dataset)

        if not load_from_disk:
            vectorizer = CountVectorizer(tokenizer=lambda t: filter(None, t.split('\n')))
            corpus = list()

            import_index = 0
            imports_per_md5 = Cougar.info_df.num_imports
            holdout_imports_per_md5 = None
            if subset is not None:
                imports_per_md5 = imports_per_md5[:subset]
            if holdout is not None:
                holdout_imports_per_md5 = Cougar.info_df.num_imports[subset:subset + holdout]

            lg.debug('Loading imports...')
            imports_list = [imports_per_md5]
            md5s_list = [self.md5s]
            if holdout is not None:
                self.md5s_holdout = list()
                imports_list.append(holdout_imports_per_md5)
                md5s_list.append(self.md5s_holdout)

            for imports_per_md5_i, md5s_i in zip(imports_list, md5s_list):
                for import_count in tqdm(imports_per_md5_i):
                    md5_df = Cougar.imports_df[import_index:import_index + import_count]
                    if len(md5_df) > 0:
                        md5s_i.append(md5_df.md5[import_index])
                        csv = md5_df.to_csv(index=False, header=False, columns=['library', 'function'])
                        corpus.append(csv)
                    import_index += import_count

            self.md5s = pd.DataFrame(self.md5s)
            if holdout is not None:
                self.md5s_holdout = pd.DataFrame(self.md5s_holdout)
            lg.debug('Vectorizing...')
            Cougar.vectorization = vectorizer.fit_transform(corpus)

            if self.save:
                sparse.save_npz(VECTORIZATION_FILE, Cougar.vectorization)
                with sqlite3.connect(SQLITE_FILE_NAME) as cnxn:
                    self.md5s.to_sql(name=SQLITE_MD5_TABLE, con=cnxn, index=False)
                    if holdout is not None:
                        self.md5s_holdout.to_sql(name=SQLITE_MD5_TABLE_HOLDOUT, con=cnxn, index=False)
                    lg.debug('Saved MD5s')

        else:
            Cougar.vectorization = sparse.load_npz(VECTORIZATION_FILE)
            with sqlite3.connect(SQLITE_FILE_NAME) as cnxn:
                try:
                    self.md5s = pd.read_sql(f'SELECT * from {SQLITE_MD5_TABLE}', cnxn)
                    if holdout is not None:
                        self.md5s_holdout = pd.read_sql(f'SELECT * from {SQLITE_MD5_TABLE_HOLDOUT}', con=cnxn)
                except DatabaseError:
                    lg.error(f'Failed to load {SQLITE_MD5_TABLE} from Sqlite3 DB')
                    return

    def compute_md5_params(self):
        """
        Compute MD5 hashes of the desired parameters and store them in the Castor instance.
        """
        self.reduction_params_md5 = md5(dumps(self.reduction_params, sort_keys=True).encode(UTF_8)).hexdigest()
        self.clustering_params_md5 = md5(dumps(self.clustering_params, sort_keys=True).encode(UTF_8)).hexdigest()

    def reset_models(self):
        """
        Reset the embedding and computed labels before a new computation.
        """
        self.embedding = None
        self.labels = None

    def set_params(self, params: dict, is_reduce: bool):
        if is_reduce:
            self.reduction_params = params
        else:
            self.clustering_params = params
        self.compute_md5_params()

    def set_save(self, save: bool = True):
        self.save = save
        if self.save and not path.exists(self.working):
            mkdir(self.working)

    def load_prebuilt_model(self, is_reduce: bool):
        """
        Load a previously built model from disk, which is then stored as part of this Castor instance.

        :param is_reduce: a bool indicating whether the desired model is a dimension reduction embedding
        """
        md5sum = self.reduction_params_md5 if is_reduce else self.reduction_params_md5 + self.clustering_params_md5
        with sqlite3.connect(SQLITE_FILE_NAME) as cnxn:
            try:
                df = pd.read_sql(f'SELECT * from \"{md5sum}\"', cnxn)
            except DatabaseError:
                lg.debug(f'Failed to load {md5sum} from Sqlite3 DB')
                return

        if is_reduce:
            lg.debug(f'Loaded embedding with MD5: {md5sum}')
            self.embedding = df
        else:
            lg.debug(f'Loaded labels with MD5: {md5sum}')
            self.labels = df

    def reduce(self, use_holdout: bool = False):
        """
        Compute a dimension reduction embedding and store it as part of this Cougar instance.

        :param use_holdout: a bool indicating whether holdout data should be transformed into an existing embedding
        """
        if use_holdout:
            with open(REDUCER_PICKLE_FILE, 'rb') as f:
                reducer = pickle.load(f)
            # Using holdout assumes a model has already been built. We want to add new data in that case!
            embedding = pd.DataFrame(reducer.transform(Cougar.vectorization[len(self.md5s):]))
            embedding.columns = ['x', 'y']
            self.embedding.columns = ['x', 'y']
            self.embedding = pd.concat([self.embedding, embedding], ignore_index=True)
            return
        elif self.embedding is not None:
            return

        reducer = self.reduction_algorithm() if self.reduction_params is None else \
            self.reduction_algorithm(**self.reduction_params)
        lg.debug('Fitting dimension reduction...')
        X = Cougar.vectorization if self.md5s_holdout is None else Cougar.vectorization[:len(self.md5s)]
        embedding = reducer.fit_transform(X)
        lg.debug(f'Embedding shape: {embedding.shape}')
        self.embedding = pd.DataFrame(embedding)

        if self.save:
            with open(REDUCER_PICKLE_FILE, 'wb') as f:
                pickle.dump(reducer, f)
            with sqlite3.connect(SQLITE_FILE_NAME) as cnxn:
                self.embedding.to_sql(name=self.reduction_params_md5, con=cnxn, index=False)
                lg.debug(f'Saved embedding with MD5: {self.reduction_params_md5}')

    def cluster(self):
        """
        Compute a clustering labels and store them as part of this Cougar instance.
        """
        clusterer = self.cluster_algorithm(**self.clustering_params)
        lg.debug('Clustering...')
        self.labels = pd.DataFrame(clusterer.fit_predict(self.embedding))

    def evaluate(self, index: int = None, use_holdout: bool = False) -> tuple:
        """
        Evaluate the quality of the computed embedding and labels by calculating the
        number of clusters and the purity of each cluster, where the purity is the
        total sum of the sum-squared errors of each cluster.

        :param index: an int for identifying the output files, optional
        :param use_holdout: a bool indicating whether holdout data should be evaluated
        :return: a tuple, the number of clusters, the total sum of the sum-squared errors of each cluster,
        and the median cluster count
        """
        suffix = HOLDOUT_SUFFIX if use_holdout else ''
        if not use_holdout:
            cluster_df = pd.concat([self.md5s, self.embedding, self.labels], axis=1)
            cluster_df.columns = ['md5', 'x', 'y', 'cluster']
        else:
            holdout_label_series = pd.Series([0] * len(self.md5s) + [1] * len(self.md5s_holdout))
            cluster_df = pd.concat([
                self.md5s.append(self.md5s_holdout, ignore_index=True),
                self.embedding, self.labels, holdout_label_series
            ], axis=1)
            cluster_df.columns = ['md5', 'x', 'y', 'cluster', 'is_holdout']

        if index is None:
            index = 0

        if self.save:
            plt.figure()
            sct = plt.scatter(cluster_df.x, cluster_df.y, c=cluster_df.cluster, cmap='Spectral')
            plt.title('Reduce/cluster on the EMBER dataset', fontsize=20)
            plt.legend(*sct.legend_elements(), loc='lower left', title='Clusters')
            plt.savefig(f'{self.working}/clusters{suffix}-{index}.png')
            plt.close()

        # Remove non-clustered points
        cluster_df = cluster_df[cluster_df.cluster != -1]

        cluster_sizes = cluster_df.cluster.value_counts()
        median_cluster_sample_count = np.median(cluster_sizes) if len(cluster_sizes) > 0 else 0

        cluster_values = sorted(cluster_df.cluster.unique())
        vectorized = self.vectorization

        total_cluster_sum_squared_error = 0.0

        cluster_results = list()

        for cluster in cluster_values:
            cluster_i_df = cluster_df[cluster_df.cluster == cluster]
            sample_count = len(cluster_i_df)
            cluster_i_vectorized = vectorized[cluster_i_df.index.values]
            # If they aren't at least 2 samples, then disregard this cluster
            # This also serves to mitigate divide by zero errors in sim_sum_scaled
            if cluster_i_vectorized.shape[0] <= 1:
                sim_sum_scaled = 0.0
            else:
                sim = cosine_similarity(cluster_i_vectorized, dense_output=False)
                sim = sim[np.triu_indices(n=sim.shape[0], m=sim.shape[1], k=1)]
                sim_sum = np.sum(sim)
                sim_sum_scaled = sim_sum / comb(N=cluster_i_vectorized.shape[0], k=2)
            sse = (1 - sim_sum_scaled) ** 2
            total_cluster_sum_squared_error += sse
            cluster_results.append((cluster, sim_sum_scaled, sse, sample_count))

        evaluation_results = (
            len([1 for x in cluster_results if x[1] >= QUALITY_THRESHOLD]),
            total_cluster_sum_squared_error,
            median_cluster_sample_count
        )

        if self.save:
            with open(f'{self.working}/summary{suffix}-{index}.txt', 'w') as f:
                f.write('\n'.join(summary).format(
                    index, evaluation_results[0], len(cluster_results), evaluation_results[1], evaluation_results[2],
                    self.reduction_algorithm.__name__, str(self.reduction_params),
                    self.cluster_algorithm.__name__, str(self.clustering_params)
                ))
            with open(f'{self.working}/cluster_points{suffix}-{index}.csv', 'w') as f:
                cluster_df.to_csv(index=False, path_or_buf=f)
            with open(f'{self.working}/results{suffix}-{index}.csv', 'w') as f:
                f.write(','.join(['cluster_number', 'similarity', 'error', 'sample_count\n']))
                for cluster, sim_sum_scaled, error, sample_count in cluster_results:
                    f.write(','.join([str(cluster), str(sim_sum_scaled), str(error), str(sample_count)]) + '\n')

        return evaluation_results


if __name__ == '__main__':
    lg.basicConfig(
        format='[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=lg.INFO
    )
    parser = ArgumentParser(description='Convert a collection of import statements into clusters')
    parser.add_argument('-d', '--dataset', type=int, required=True, help='Specify training dataset partition: [0-5]')
    parser.add_argument('-l', '--location', help='Specify Parquet files directory', default='../EMBER/Parquet')
    parser.add_argument('-w', '--working', help='Working directory to store output',
                        default='../Cougar_Output/Run_{}'.format(datetime.now().strftime('%b-%d-%Y-%H-%M')))
    parser.add_argument('-r', '--reduction', required=True, help='Dimension reduction algorithm')
    parser.add_argument('-p1', '--params1', help='Dimension reduction parameters as JSON dict')
    parser.add_argument('-c', '--cluster', required=True, help='Clustering algorithm')
    parser.add_argument('-p2', '--params2', help='Clustering algorithm parameters as JSON dict')
    parser.add_argument('-s', '--save', help='Save intermediate output to disk')
    args = parser.parse_args()
    cougar = Cougar(args.dataset, args.location, args.working, args.reduction, loads(args.params1), args.cluster,
                    loads(args.params2), args.save)
    main(cougar)
