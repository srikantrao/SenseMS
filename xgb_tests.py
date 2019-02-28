import numpy as np
import os
import pickle as p
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from utils.build_datasets import make_datasets
from utils.create_data_for_testing import ThreeDatasets
import xgboost


def main():
    """
    Creates dictionary (pickled) of xgboost accuracy results at
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
                          'min_child_weight': [1, 5, 10],
                          'gamma': [0.5, 1, 1.5, 2, 5],
                          'subsample': [0.6, 0.8, 1.0],
                          'colsample_bytree': [0.6, 0.8, 1.0],
                          'max_depth': [3, 4, 5]
                         }

            # Create a based model
            model = XGBClassifier()
            # Instantiate the grid search model
            grid_search = GridSearchCV(estimator = model, param_grid = param_grid,
                                      cv = 5, n_jobs = -1, verbose = 1)
            grid_search.fit(X_train, y_train)
            best_grid = grid_search.best_estimator_
            y_pred = best_grid.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[i][freq] = acc

    p.dump(results, open("results/xgb_results.p", "wb"))


if __name__ == "__main__":
    main()
