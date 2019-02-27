from xgboost import XGBClassifier
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import itertools
import pickle as p
import tensorflow as tf
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import utils.data_formatutils as dfu
import utils.readinutils as readinutils
from utils.dataset import Dataset
from utils.dataset import DatasetGroup
from temp_blur import DatasetAtFrequency

## Models
from models.conv_mlp import CONV_MLP as conv_mlp
from models.fc_mlp import FC_MLP as fc_mlp
from models.regression import REGRESSION as reg

## Params
from params.conv_mlp_params import params as conv_mlp_params
from params.fc_mlp_params import params as fc_mlp_params
from params.regression_params import params as lr_params

"""--------------------------------------
  Model specification is done here:
--------------------------------------"""
model_type = "fc_mlp"
#model_type = "conv_mlp" # TODO: Not implemented yet
#model_type = "reg" # TODO: Not implemented yet
"""--------------------------------------
--------------------------------------"""

"""--------------------------------------
  Data path specification is done here:
--------------------------------------"""
figpath = os.path.expanduser("~")+'/figs/'
datadir = os.path.expanduser("~")+'/envision_working_traces/'
patientfile = './data/patient_stats.csv'
"""--------------------------------------
--------------------------------------"""

## Param selection
if model_type == "conv_mlp":
    params = conv_mlp_params
elif model_type == "fc_mlp":
    params = fc_mlp_params
elif model_type == "reg":
    params = lr_params
else:
    assert False, ("model_type must be 'conv_mlp', 'fc_mlp', or 'reg'")

"""--------------------------------------
  Params sweeping options
--------------------------------------"""
sweep_params = dict()
sweep_params["model_name"] = ["split_30_arch_60_40_30_20_10_2_5crossval"]
#sweep_params["model_name"] = ["split_30_arch_50_30_10_2"]
#cannot make crossvalidations a sweepable parameter, as Dataset instantiation is outside loop
sweep_params["weight_learning_rate"] = [0.006]
sweep_params["lr_staircase"] = [False]
sweep_params["lr_decay_rate"] = [0.98]
sweep_params["weight_decay_mults"] = [[3.00001E-05, 3.00001E-05, 3.00001E-05, 3.00001E-05, 3.00001E-05, 3.00001E-05]]
sweep_params["orth_mults"] = [[5.00001E-06, 5.00001E-06, 5.00001E-06, 5.00001E-06, 5.00001E-06,
5.00001E-06]]
sweep_params["do_batch_norm"] = [True]
sweep_params["norm_decay_mult"] = [0.6] # dont need to sweep if we aren't using batch norm
params.num_crossvalidations = 1
"""--------------------------------------
--------------------------------------"""

## Data setup
trials = readinutils.readin_traces(datadir, patientfile)
trials = [trial for trial in trials if(trial.sub_ms.size>0)]
if params.truncate_trials:
    trials = dfu.truncate_trials(trials)
if params.trial_split_multiplier is not None:
    trials = dfu.split_trials(trials, multiplier=params.trial_split_multiplier)

patient_trials = [trial for trial in trials if trial.sub_ms == '1']
control_trials = [trial for trial in trials if trial.sub_ms == '0']

# freqs for temporal blur
#freqs = np.arange(8, 481, 8)
freqs = np.array([8, 30, 480])
new_datasets = {}
full_dataset = [patient_trials, control_trials]

# build datasets
for i in range(len(freqs)):
    data = DatasetAtFrequency(freqs[i], full_dataset)
    new_datasets[freqs[i]] = data.get_new_data()

print("all data")
results = {}
for freq, data in new_datasets.items():
    patient_trials = data[0]
    control_trials = data[1]
    print("length of points in first patient trial", patient_trials[0].num_strips)
    if(len(patient_trials + control_trials) != 0):
        data, labels, stats = dfu.make_mlp_eyetrace_matrix(patient_trials, control_trials, params.fft_subsample)
        dataset_group_list = DatasetGroup(data, labels, stats, params)

        #if hasattr(params, "flip_trials_on_y_axis") and params.flip_trials_on_y_axis:
        #    dataset_group_list.flip_trials_y()
        #if hasattr(params, "reduce_data") and params.reduce_data:
        #    dataset_group_list.pca_whiten_reduce(params.required_explained_variance, params.rand_state)
        if hasattr(params, "concatenate_age") and params.concatenate_age:
            dataset_group_list.concatenate_age_information()
        #if hasattr(params, "concatenate_fft") and params.concatenate_fft:
        #    dataset_group_list.concatenate_fft_information()
        #if hasattr(params, "concatenate_vel") and params.concatenate_vel:
        #    dataset_group_list.concatenate_vel_information()
        #if hasattr(params, "concatenate_nblinks") and params.concatenate_nblinks:
        #    dataset_group_list.concatenate_nblinks_information()
        
# add ids just to check for leakage
        dataset_group_list.concatenate_id_information() 
        sys.stdout.flush()

X_train, y_train = (dataset_group_list.get_dataset(0)["train"].data, 
                    dataset_group_list.get_dataset(0)["train"].labels)
X_test, y_test = (dataset_group_list.get_dataset(0)["val"].data, 
                    dataset_group_list.get_dataset(0)["val"].labels)

model = XGBClassifier()
model.fit(X_train[:,:-1], y_train[:,1])
y_pred = model.predict(X_test[:,:-1])
acc = accuracy_score(y_test[:,1], y_pred)
print(acc)
