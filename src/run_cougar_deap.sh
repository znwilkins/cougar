#!/usr/bin/env bash
##############################################################################################################
#                                                                                                            #
#    File: run_cougar_deap.sh                                                                                #
#    Date: May 25, 2020                                                                                      #
#    Purpose:                                                                                                #
#      This file contains an example of how to run COUGAR with DEAP/SCOOP                                    #
#      to evolve clustering parameters.                                                                      #
#                                                                                                            #
#    Copyright (c) 2020 Zachary Wilkins                                                                      #
#    COUGAR is licensed under the MIT Licence.                                                               #
#                                                                                                            #
##############################################################################################################

source ../venv/bin/activate

export CLUSTERING_ALGORITHM=dbscan
# Optionally, recover from a pickled checkpoint file in case of system crash
# export CHECKPOINT_FILE=../Cougar_Output/Run_Jan-31-2020-12-12/checkpoint.pkl
export DATASET_SIZE=2000
export HOLDOUT_PRESENT=0  # Indicate whether holdout data is present

# -n flag controls the number of SCOOP workers
python3 -m scoop -n 4 deap_optimize_cougar.py

deactivate
