import numpy as np
import os
import pickle as p
import sys
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import  accuracy_score
from utils.build_datasets import make_datasets
from utils.create_data_for_testing import ThreeDatasets


def main():
    """
    Creates dictionary (pickled) of logistic regression accuracy results at
    specified frequencies, for the three conditions:
                            - All data
                            - All data but age
                            - Only age
    This pickled dictionary can be parsed by running process_results.py

    Returns:
       Pickled dictionary of logistic regression accuracy scores, indexed by
       frequency. This file is parsed by process_results.py
    """
    # freqs for downsampling
    freqs = np.array([8, 30, 480])

    # build datasets
    datasets_at_freqs = make_datasets(freqs)

    # all data, all but age, age only datasets
    make_three_datasets = ThreeDatasets(datasets_at_freqs)
    three_datasets = make_three_datasets.get_three_datasets()

    results = [{}, {}, {}]
    for i, dataset in enumerate(three_datasets):
        for freq, data in dataset.items():
            # for each frequency and each dataset, find accuracy
            X_train, y_train = data[0][0], data[0][1]
            X_test, y_test = data[1][0], data[1][1]
            log_model = LogisticRegression(C=1000)
            log_model.fit(X_train, y_train)
            logistic_results = log_model.predict(X_test)
            acc = accuracy_score(y_test, logistic_results)
            results[i][freq] = acc

    p.dump(results, open("results/logregr_results.p", "wb"))


if __name__ == "__main__":
    main()
