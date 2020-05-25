
##############################################################################################################
#                                                                                                            #
#    File: score.py                                                                                          #
#    Date: May 21, 2020                                                                                      #
#    Purpose:                                                                                                #
#      This file contains functionality to extract the final averages for each objective over many runs.     #
#                                                                                                            #
#    Copyright (c) 2020 Zachary Wilkins                                                                      #
#    COUGAR is licensed under the MIT Licence.                                                               #
#                                                                                                            #
##############################################################################################################

import pandas as pd

from os import listdir
from os.path import join
from sys import argv, stderr


def main():
    if len(argv) < 2:
        """
        The passed value should be the top-most directory in a tree structured as:
        
        passed_directory/
        ├── run_set_1
        │   ├── Run_Jan-31-2020-12-12
        │   │   ├── logbook.csv
        │   │   └── ...
        │   │
        │   └── ...
        ├── run_set_2
        │   ├── Run_Jan-31-2020-13-01
        │   │   ├── logbook.csv
        │   │   └── ...
        │   │
        │   └── ...
        └── ...
        
        """
        print('Usage: python3 extract_stats.py /path/to/test/results', file=stderr)
        exit(1)
    root = argv[1]
    print(root)

    for directory in listdir(root):
        directory_full_path = join(root, directory)
        records = list()
        for run_dir in listdir(directory_full_path):
            logbook_path = join(directory_full_path, run_dir, 'logbook.csv')
            df = pd.read_csv(logbook_path)
            # From last row, extract avgs of all three objectives
            df = df[-1:][['avg-0', 'avg-1', 'avg-2']]
            df['run'] = join(directory, run_dir)
            records.append(df)
        combined = pd.concat(records, ignore_index=True)
        combined.columns = ['count', 'error', 'median', 'run']
        with open(join(directory_full_path, 'final_avgs.csv'), 'w') as f:
            combined.to_csv(f, index=False)


if __name__ == '__main__':
    main()
