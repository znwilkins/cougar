
##############################################################################################################
#                                                                                                            #
#    File: anova.py                                                                                          #
#    Date: May 21, 2020                                                                                      #
#    Purpose:                                                                                                #
#      This file contains functionality to perform analysis of variance on COUGAR results.                   #
#                                                                                                            #
#    Copyright (c) 2020 Zachary Wilkins                                                                      #
#    COUGAR is licensed under the MIT Licence.                                                               #
#                                                                                                            #
##############################################################################################################

# Big props to: https://reneshbedre.github.io/blog/anova.html

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from os import listdir
from os.path import join
from sys import argv, stderr

OBJECTIVE_NAMES = ['count', 'error', 'median']


def main():
    if len(argv) < 2:
        """
        The passed value should be the top-most directory in a tree structured as:

        passed_directory/
        ├── run_set_1
        │   ├── final_avgs.csv
        │   └── ...
        ├── run_set_2
        │   ├── final_avgs.csv
        │   └── ...
        └── ...
        
        """
        print('Usage: python3 anova.py /path/to/test/results', file=stderr)
        exit(1)
    root = argv[1]
    print(root)

    stats_algos = list()
    stats_dfs = list()

    for directory in listdir(root):
        file_path = join(root, directory, 'final_avgs.csv')
        df = pd.read_csv(file_path)
        stats_algos.append(directory)
        stats_dfs.append(df)

    with open(join(root, 'anova_results.txt'), 'w') as f:
        for column in OBJECTIVE_NAMES:
            print(column, file=f)
            df = pd.concat([x[column] for x in stats_dfs], axis=1)
            df.columns = stats_algos
            fvalue, pvalue = stats.f_oneway(*[df[algo] for algo in stats_algos])
            print('F-Value: {}\nP-Value: {}'.format(fvalue, pvalue), file=f)
            df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=stats_algos)
            df_melt.columns = ['index', 'algorithm', 'value']
            m_comp = pairwise_tukeyhsd(endog=df_melt['value'], groups=df_melt['algorithm'], alpha=0.05)
            print(m_comp, file=f)
            print('', file=f)


if __name__ == '__main__':
    main()
