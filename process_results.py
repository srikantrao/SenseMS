import matplotlib.pyplot as plt
import numpy as np
import pickle


results = pickle.load(open('results_all.p', 'rb'))

# for each of the 'final'  metrics
val_map = {0:"Mean Train Accuracy",
           1:"Mean Val. Accuracy",
           2:"Mean Sensitivity",
           3:"Mean Specificity"}

for i in range(4):
    Hz = []
    metric = []
    for key in results:
        Hz.append(key)
        metric.append(results[key][i])

    plt.figure()
#    plt.xlim(2, 1)  # decreasing frame-rate
    metr = str(val_map[i])
    plt.xlabel('Frame-rate (Hz)')
    plt.ylabel("{}".format(metr))
    plt.title("{}".format(metr))
    plt.plot(Hz, metric)
    plt.savefig("results/best_params_all_data/{}.png".format(metr.replace(' ', '_')))

