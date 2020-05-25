
##############################################################################################################
#                                                                                                            #
#    File: reduce_to_disk.py                                                                                 #
#    Date: May 25, 2020                                                                                      #
#    Purpose:                                                                                                #
#      This file contains functionality to reduce high dimensional data to be used with COUGAR.              #
#                                                                                                            #
#    Copyright (c) 2020 Zachary Wilkins                                                                      #
#    COUGAR is licensed under the MIT Licence.                                                               #
#                                                                                                            #
##############################################################################################################

import logging as lg
from datetime import datetime

from cougar import Cougar, RANDOM_STATE

USE_HOLDOUT = True
DATA_SUBSET_SIZE = 2000
DATA_HOLDOUT_SIZE = 1000


def main():
    cougar_instance = Cougar(
        dataset=2,
        location='../EMBER/Parquet',
        working='../Cougar_Output/Run_{}'.format(datetime.now().strftime('%b-%d-%Y-%H-%M')),
        reduction='umap',
        params1={'n_neighbors': int(DATA_SUBSET_SIZE * 0.01), 'min_dist': 0.1},
        cluster='dbscan',
        params2=None,
        save=True
    )
    if USE_HOLDOUT:
        cougar_instance.reduction_params['random_state'] = RANDOM_STATE
    cougar_instance.compute_md5_params()
    lg.info('Vectorizing...')
    cougar_instance.vectorize(subset=DATA_SUBSET_SIZE, holdout=DATA_HOLDOUT_SIZE if USE_HOLDOUT else None)
    lg.info('Fitting dimension reduction...')
    cougar_instance.reduce()


if __name__ == '__main__':
    lg.basicConfig(
        format='[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=lg.INFO
    )
    main()
