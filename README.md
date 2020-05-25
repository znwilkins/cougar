# COUGAR
**Clustering Of Unknown malware using Genetic Algorithm Routines**

COUGAR 
is a system capable of reducing high-dimensional malware behavioural data,
and optimizing the clustering of that data with the assistance
of a multi-objective genetic algorithm, for the purposes of labelling
unknown malware.

This [repo](https://github.com/znwilkins/cougar) is associated with the
following paper:
* [Zachary Wilkins](https://znwilkins.ca) and
[Nur Zincir-Heywood](https://web.cs.dal.ca/~zincir). 2020.
COUGAR: Clustering Of Unknown malware using Genetic Algorithm Routines.
In _Genetic and Evolutionary Computation Conference
([GECCO '20](https://gecco-2020.sigevo.org/)), July 8–12, 2020, Cancún,
Mexico_. ACM, New York, NY, USA, 9 pages.
https://doi.org/10.1145/3377930.3390151

## Setup

To setup the virtualenv:
```bash
# This may require you to install the python3-venv package
# You can do so on a Debian-based system with: sudo apt install python3-venv
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

You'll also need to download/install UMAP and its dependencies, as this repo uses
a [pre-release version](https://github.com/lmcinnes/umap/tree/d214e5dbaa30f63a0bf7608d9de96cfa94de21e1):
```bash
cd umap-d214e5dbaa30f63a0bf7608d9de96cfa94de21e1
pip install -r requirements.txt
python setup.py install
cd ..
```

To exit the virtual environment, run
```bash
deactivate
```

## Usage

Before using COUGAR, you must first supply malware data in the form of
Parquet tables. After acquiring the
[EMBER](https://github.com/endgameinc/ember) data, use
`convert_ember_to_parquet.py` to generate the required files. This codebase
assumes data from `train_features_2.jsonl` in the 2018 dataset is available.

A simple example of running COUGAR is given in `run_cougar.py`. Ensure
that the `FIRST_RUN` flag is appropriately set, and a directory with
results will be saved in `$REPO/Cougar_Output`. In addition, a SQLite3
Database file containing the UMAP embedding and a compressed NumPy array
file representing the vectorization will be saved in `$REPO/src`. These
can be reused on subsequent calls to COUGAR, saving work when testing
clustering algorithms or parameters for them.

If you wish to prepare data in advance, consult `reduce_to_disk.py` for
an example of saving an embedding to disk without evaluating.

Running with DEAP expects that the aforementioned files have already been
generated. `run_cougar_deap.sh` demonstrates how to run COUGAR in
evolutionary mode.

To score the resulting data, use `score.py`, which contains usage info
in the argparser configuration. It is designed such that you can call
the script twice on the same directory: the first time to evaluate the
training data, and the second with `--holdout` to evaluate the holdout
data.

## Supplementary material

The interested reader can find supplementary README.md files in
each of:

* StatisticalSignificance
* RealWorldScenario
