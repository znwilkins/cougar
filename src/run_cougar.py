
##############################################################################################################
#                                                                                                            #
#    File: run_cougar.py                                                                                     #
#    Date: May 25, 2020                                                                                      #
#    Purpose:                                                                                                #
#      This file contains examples of how to run COUGAR in standalone mode (without evolution).              #
#                                                                                                            #
#    Copyright (c) 2020 Zachary Wilkins                                                                      #
#    COUGAR is licensed under the MIT Licence.                                                               #
#                                                                                                            #
##############################################################################################################

import logging as lg
from datetime import datetime

from cougar import Cougar

FIRST_RUN = True
DATA_SUBSET_SIZE = 2000


def main():
    cougar_instance = Cougar(
        dataset=2,
        location='../EMBER/Parquet',
        working='../Cougar_Output/Run_{}'.format(datetime.now().strftime('%b-%d-%Y-%H-%M-%S')),
        reduction='umap',
        params1={'n_neighbors': int(DATA_SUBSET_SIZE * 0.01), 'min_dist': 0.1},
        cluster='dbscan',
        params2={'eps': 0.5, 'min_samples': 20},
        save=FIRST_RUN
    )
    cougar_instance.compute_md5_params()
    if FIRST_RUN:
        lg.info('Building vectorization...')
        cougar_instance.vectorize(subset=DATA_SUBSET_SIZE)
        lg.info('Building dimension reduction embedding...')
        cougar_instance.reduce()
    else:
        lg.info('Loading vectorization...')
        cougar_instance.vectorize(load_from_disk=True)
        lg.info('Loading dimension reduction...')
        cougar_instance.load_prebuilt_model(is_reduce=True)

    cougar_instance.cluster()
    lg.info(cougar_instance.evaluate())


if __name__ == '__main__':
    lg.basicConfig(
        format='[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=lg.INFO
    )
    main()
