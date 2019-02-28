import numpy as np
import os
import pickle as p
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from utils.build_datasets import make_datasets
from utils.create_data_for_testing import ThreeDatasets


def main():
    """
    Creates dictionary (pickled) of random forest accuracy results at
    specified frequencies, for the three conditions:
                            - All data
                            - All data but age
                            - Only age
    This pickled dictionary can be parsed by running process_results.py

    Returns:
       Pickled dictionary of accuracy scores, indexed by
       frequency. This file is parsed by process_results.py
    """
    # frequencies used for downsampling
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
            # Create the parameter grid based on the results of random search
            param_grid = {
                'bootstrap': [True],
                'max_depth': [80, 90, 100, 110],
                'max_features': [2, 3],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [100, 200, 300, 1000]
            }

            # Create a based model
            model = RandomForestClassifier()
            # Instantiate the grid search model
            grid_search = GridSearchCV(estimator = model, param_grid = param_grid,
                                      cv = 5, n_jobs = -1, verbose = 1)
            grid_search.fit(X_train, y_train)
            best_grid = grid_search.best_estimator_
            y_pred = best_grid.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[i][freq] = acc

    p.dump(results, open("results/rf_results.p", "wb"))


if __name__ == "__main__":
    main()
