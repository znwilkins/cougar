
##############################################################################################################
#                                                                                                            #
#    File: score.py                                                                                          #
#    Date: May 25, 2020                                                                                      #
#    Purpose:                                                                                                #
#      This file contains functionality to score COUGAR runs and output a summary HTML page.                 #
#                                                                                                            #
#    Copyright (c) 2020 Zachary Wilkins                                                                      #
#    COUGAR is licensed under the MIT Licence.                                                               #
#                                                                                                            #
##############################################################################################################

import pandas as pd
from sklearn.metrics import f1_score, homogeneity_completeness_v_measure, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from argparse import ArgumentParser
from base64 import b64encode
from datetime import datetime
import logging as lg
import matplotlib.pyplot as plt
from multiprocessing import Pool
from operator import itemgetter
from os import listdir, path
import re

from cougar import Cougar, QUALITY_THRESHOLD, UTF_8, HOLDOUT_SUFFIX, RANDOM_STATE

DATA_SUBSET_SIZE = 2000
DATA_HOLDOUT_SIZE = 1000


TOP_N = 16
INDEX_RE = re.compile(r'.+(\d+?).csv')
MD5_INDEX_LOOKUP: dict


def main(root: str, evaluate_holdout: bool):
    lg.info(f'Root is {root}')
    lg.info('Loading Parquet data from disk, please wait...')
    init()

    if not evaluate_holdout:
        score(root)
    else:
        score_holdout(root)


def init():
    dataset_partition = 2
    location = '../EMBER/Parquet'
    Cougar.load_parquet((
        f'{location}/train_features_{dataset_partition}_2018_info.parquet',
        f'{location}/train_features_{dataset_partition}_2018_imports.parquet'
    ))
    Cougar.info_df['avnum'] = LabelEncoder().fit(Cougar.info_df.avclass).transform(Cougar.info_df.avclass)
    global MD5_INDEX_LOOKUP
    MD5_INDEX_LOOKUP = {v: k for k, v in Cougar.info_df.md5.to_dict().items()}


def score(root: str):
    results = list()

    for p in tqdm(listdir(root)):
        run_dir = path.join(root, p)
        if path.isdir(run_dir):
            files_run_dir = listdir(run_dir)
            results_files = sorted([path.join(run_dir, f) for f in files_run_dir if 'results' in f])
            with Pool(processes=4) as pool:
                results.extend(pool.map(pool_eval_results_file, results_files))

    # Why can't I define a comparator like in Java? Let's sort three times instead >.<
    results = sorted(results, key=itemgetter(3), reverse=True)
    results = sorted(results, key=itemgetter(2))
    results = sorted(results, key=itemgetter(1), reverse=True)

    total_runs = len(results)
    results = results[:TOP_N]

    time_now = datetime.now().strftime('%b-%d-%Y-%H-%M')
    html_out = build_html(results, time_now, total_runs, root)
    pd.DataFrame(results, columns=[
        'results_file', 'quality_count', 'error', 'median_sample_count',
        'precision', 'recall', 'fscore_macro', 'fscore_micro', 'fscore_weighted',
        'homogeneity', 'completeness', 'v_measure'
    ]).to_csv(f'{root}/topN_stats_{time_now}.csv', index=False)

    with open(f'{root}/results_{time_now}.html', 'w') as f:
        f.write(html_out)


def pool_eval_results_file(results_file: str) -> tuple:
    print(results_file)
    # Start with stats from COUGAR.evaluate()
    df = pd.read_csv(results_file)
    quality_count = df.similarity.where(df.similarity > QUALITY_THRESHOLD).count()
    error = df.error.sum()
    median_sample_count = df.sample_count.median() if len(df) > 0 else 0

    # Then stats from ground truth
    cluster_points_file = results_file.replace('/results', '/cluster_points')
    truth_stats = compute_ground_truth_stats(cluster_points_file, save=True)

    return (results_file, quality_count, error, median_sample_count, *truth_stats)


def score_holdout(root: str):
    working_dir = path.join(root, 'ScoreHoldout')
    cougar_instance = Cougar(
        dataset=2,
        location='../EMBER/Parquet',
        working=working_dir,
        reduction='umap',
        params1={'n_neighbors': int(DATA_SUBSET_SIZE * 0.01), 'min_dist': 0.1, 'random_state': RANDOM_STATE},
        # These parameters need to be customized based on the chosen COUGAR individual
        cluster='dbscan',
        params2={'eps': 0.056232240210074926, 'min_samples': 5},
        save=True
    )
    cougar_instance.compute_md5_params()
    lg.info('Loading vectorization...')
    cougar_instance.vectorize(load_from_disk=True, holdout=DATA_HOLDOUT_SIZE)
    cougar_instance.load_prebuilt_model(is_reduce=True)
    lg.info('Clustering initial data...')
    cougar_instance.cluster()
    lg.info('Fitting holdout data...')
    cougar_instance.reduce(use_holdout=True)
    lg.info('Clustering/evaluating holdout data...')
    cougar_instance.cluster()
    holdout_index = 1
    cougar_instance.evaluate(index=holdout_index, use_holdout=True)
    cluster_file = path.join(working_dir, f'cluster_points_holdout-{holdout_index}.csv')
    compute_ground_truth_stats(cluster_file, save=True, use_holdout=True)


def compute_ground_truth_stats(cluster_file: str, save: bool = False, use_holdout: bool = False) -> tuple:
    """
    Compute a number of metrics based on a clustering generated from COUGAR.

    :param cluster_file: path to csv file containing clustered samples
    :param save: a bool indicating whether output should be saved from these calculations
    :param use_holdout: a bool indicating whether holdout data is present and should be evaluated
    :return: a tuple, containing precision, recall, fscore_macro, fscore_micro, fscore_weighted,
    homogeneity, completeness, and v_measure calculations for the given cluster file
    """
    suffix = HOLDOUT_SUFFIX if use_holdout else ''
    attrs_to_merge = ['md5', 'label', 'is_holdout'] if use_holdout else ['md5', 'label']
    cluster_root = path.split(cluster_file)[0]
    index_extraction_result = INDEX_RE.findall(cluster_file)
    clustering_index = index_extraction_result[0] if len(index_extraction_result) > 0 else 0

    cluster_df = pd.read_csv(cluster_file)
    if len(cluster_df) == 0:
        # Short-circuit when the clustering is rubbish
        return tuple([0.0] * 8)

    md5_cluster_lookup = {row[0]: row[1] for row in cluster_df[['md5', 'cluster']].values}
    if use_holdout:
        md5_holdout_lookup = {row[0]: row[1] for row in cluster_df[['md5', 'is_holdout']].values}

    cluster_membership = dict()

    for md5, index in MD5_INDEX_LOOKUP.items():
        if md5 in md5_cluster_lookup:
            record = Cougar.info_df.iloc[index]
            cluster_number = md5_cluster_lookup[md5]
            if use_holdout:
                record = record.append(pd.Series([md5_holdout_lookup[md5]], index=['is_holdout']))

            if cluster_number in cluster_membership:
                cluster = cluster_membership[cluster_number]
                cluster.append(record)
            else:
                cluster_membership[cluster_number] = [record]

    cluster_membership = {k: pd.DataFrame(v) for k, v in cluster_membership.items()}
    for cluster, df in list(cluster_membership.items()):
        og_df = df
        if use_holdout:
            # Only allow non-holdout data to 'vote' on cluster label
            df = df[df.is_holdout == 0]
            if len(df) == 0:
                del cluster_membership[cluster]
                continue
        cluster_numeric_label = df.avnum.value_counts().index[0]
        og_df['label'] = cluster_numeric_label

    labeled_cluster_df = pd.concat(cluster_membership)
    y_true = labeled_cluster_df.avnum
    y_pred = labeled_cluster_df.label

    if use_holdout:
        holdout_df = labeled_cluster_df[labeled_cluster_df.is_holdout == 1]
        y_true_ho = holdout_df.avnum
        y_pred_ho = holdout_df.label
        precision_ho, recall_ho, fscore_macro_ho, _ = precision_recall_fscore_support(y_true=y_true_ho,
                                                                                      y_pred=y_pred_ho,
                                                                                      average='macro')
        fscore_micro_ho = f1_score(y_true=y_true_ho, y_pred=y_pred_ho, average='micro')
        fscore_weighted_ho = f1_score(y_true=y_true_ho, y_pred=y_pred_ho, average='weighted')
        homogeneity_ho, completeness_ho, v_measure_ho = homogeneity_completeness_v_measure(labels_true=y_true_ho,
                                                                                           labels_pred=y_pred_ho)

    precision, recall, fscore_macro, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro')
    fscore_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    fscore_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels_true=y_true, labels_pred=y_pred)

    if save:
        with open(f'{cluster_root}/stats{suffix}-{clustering_index}.csv', 'w') as f:
            f.write('\n'.join([
                ','.join([
                    'precision', 'recall', 'fscore_macro', 'fscore_micro',
                    'fscore_weighted', 'homogeneity', 'completeness', 'v_measure'
                ]),
                ','.join([
                    str(precision), str(recall), str(fscore_macro), str(fscore_micro),
                    str(fscore_weighted), str(homogeneity), str(completeness), str(v_measure)
                ])
            ]) + '\n')
            if use_holdout:
                f.write('\n'.join([
                    ','.join([
                        str(precision_ho), str(recall_ho), str(fscore_macro_ho), str(fscore_micro_ho),
                        str(fscore_weighted_ho), str(homogeneity_ho), str(completeness_ho), str(v_measure_ho)
                    ]),
                    'First row is overall. Second row is holdout only.'
                ]))

        with open(f'{cluster_root}/cluster_points_labeled{suffix}-{clustering_index}.csv', 'w') as f:
            merged = pd.merge(cluster_df[['md5', 'x', 'y']], labeled_cluster_df[attrs_to_merge], on='md5')
            merged.to_csv(index=False, path_or_buf=f)

        plt.figure()
        sct = plt.scatter(merged.x, merged.y, c=merged.label, cmap='Spectral')
        plt.title('Cluster labels on EMBER', fontsize=20)
        plt.legend(*sct.legend_elements(), loc='lower left', title='AVClass Clusters')
        plt.savefig(f'{cluster_root}/clusters_labeled{suffix}-{clustering_index}.png')
        plt.close()

    return precision, recall, fscore_macro, fscore_micro, fscore_weighted, homogeneity, completeness, v_measure


def build_html(results: list, time_now: str, total_runs: int, root: str) -> str:
    """
    Build an HTML webpage giving an overview of all clustering results.

    :param results: a list of tuples, where each tuple consists of a results file path,
    the quality count, error, and median sample count objective scores, as well as all of the
    metrics from compute_ground_truth_stats for that results file.
    :param time_now: the present date and time
    :param total_runs: the total number of runs considered
    :param root: the root directory where all results and stats are stored
    :return: an HTML webpage giving an overview of all results
    """
    header = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>COUGAR Results</title>
        <style>
        table, th {
          border: 1px solid black;
        }

        table tr:nth-child(even) {
          background-color: #eee;
        }

        table tr:nth-child(odd) {
          background-color: #fff;
        }

        table {
          width: 100%%;
        }

        td {
          text-align: center;
        }
        
        .stats {
          text-align: right;
          padding: 10px;
        }
        </style>
        <script>
        function swapSizeImage(element) {
            element.style.width = element.style.width === "240px" ? "480px" : "240px";
        }
        </script>
    </head>
    <body>
        <h3>Top %d individuals of %d runs in directory %s. Compiled on %s</h3>
        <table>
            <tr>
                <th>Run Info</th>
                <th>Image</th>
                <th>Results</th>
                <th>Stats</th>
            </tr>
    ''' % (TOP_N, total_runs, root, time_now)

    footer = '''
        </table>
    </body>
    </html>
    '''

    row_template = '''
            <tr>
                <td>%s</td>
                <td><img style="width: 240px; height: auto;" src="data:image/png;base64, %s" alt="Visualization of embedding" onclick="swapSizeImage(this)"></td>
                <td>(%d, %.3f, %.2f)</td>
                <td class="stats">
                    precision: %.3f<br>
                    recall: %.3f<br>
                    fscore_macro: %.3f<br>
                    fscore_micro: %.3f<br>
                    fscore_weighted: %.3f<br>
                    homogeneity: %.3f<br>
                    completeness: %.3f<br>
                    v_measure: %.3f
                </td>
            </tr>
    '''

    rows = list()
    for result in results:
        (results_file, quality_count, error, median_sample_count,
         precision, recall, fscore_macro, fscore_micro, fscore_weighted,
         homogeneity, completeness, v_measure) = result
        png_file = results_file.replace('/results', '/clusters').replace('.csv', '.png')
        with open(png_file, 'rb') as f:
            b64bytes = b64encode(f.read())
        b64str = str(b64bytes, UTF_8)
        rows.append(row_template % (
            results_file[len(root):], b64str, quality_count, error, median_sample_count,
            precision, recall, fscore_macro, fscore_micro, fscore_weighted, homogeneity,
            completeness, v_measure
        ))

    return header + ''.join(rows) + footer


if __name__ == '__main__':
    lg.basicConfig(
        format='[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=lg.INFO
    )
    parser = ArgumentParser(description='Score COUGAR runs and output a summary HTML page.')
    parser.add_argument('-r', '--root', help='Root of test results')
    parser.add_argument('-t', '--holdout', action='store_true', help='Score holdout data')
    args = parser.parse_args()
    main(args.root, args.holdout)
