import matplotlib.pyplot as plt
import numpy as np
import pickle


def main():
    """
    Plots accuracy results for each of the models at
    specified frequencies, for the three conditions:
                    - All data
                    - All data but age
                    - Only age

    Returns:
       Matplotlib plots in the '/results' directory.
    """

    # load test results
    gb_results = pickle.load(open('results/gb_results.p', 'rb'))
    xgb_results = pickle.load(open('results/gb_results.p', 'rb'))
    rf_results = pickle.load(open('results/rf_results.p', 'rb'))
    lr_results = pickle.load(open('results/logregr_results.p', 'rb'))
    all_results = [gb_results, xgb_results, rf_results, lr_results]

    # maps for labeling plots and creating filenames
    test_map = {0: "Gradient Boosting",
                1: "XGBoost",
                2: "Random Forest",
                3: "Logistic Regression"}

    condition_map = {0: "All Data",
                     1: "All Data But Age",
                     2: "Age Only",}

    for i, results in enumerate(all_results):
        for j, condition in results:
            # for each model and dataset, create plot of freq. (x) and accuracy (y)
            Hz = []
            accuracy = []
            for freq in condition:
                Hz.append(freq)
                accuracy.append(results[freq])

            # create plot
            plt.figure()
            plt.xlim(max(Hz), min(Hz))  # decreasing frame-rate
            plt.xlabel('Frame-rate (Hz)')
            plt.ylabel("Accuracy")
            plt.title(f"{test_map[i]} Accuracy with {condition_map[j]}")
            plt.plot(Hz, accuracy)
            # example filename: "gradient_boosting_with_all_data"
            filename = (test_map[i].replace(' ', '_').lower() + "_with_" +
                         condition_map[j].replace(' ', '_').lower())
            plt.savefig(f"results/{filename}.png")


if __name__ == "__main__":
    main()
