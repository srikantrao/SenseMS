#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import os
import utils.data_formatutils as dfu
import utils.readinutils as readinutils
#from utils.dataset import Dataset
#from utils.dataset import DatasetGroup
from temp_blur import DatasetAtFrequency

## Params
from params.fc_mlp_params import params as fc_mlp_params

"""--------------------------------------
  Data path specification is done here:
--------------------------------------"""
datadir = os.path.expanduser("~")+'/envision_working_traces/'
patientfile = './data/patient_stats.csv'
"""--------------------------------------
--------------------------------------"""

## Param selection
params = fc_mlp_params


def make_datasets(freqs):
    """
    Creates DatasetGroup at listed frame-rates (frequencies).
       freqs: list
    Returns:
       Dictionary of downsamples datasets, indexed by frequency, with values in the form:
              [patient_trails, control_trials]
    """

    ## Data setup
    trials = readinutils.readin_traces(datadir, patientfile)
    trials = [trial for trial in trials if(trial.sub_ms.size>0)]
    if params.truncate_trials:
        trials = dfu.truncate_trials(trials)
    if params.trial_split_multiplier is not None:
        trials = dfu.split_trials(trials, multiplier=params.trial_split_multiplier)

    patient_trials = [trial for trial in trials if trial.sub_ms == '1']
    control_trials = [trial for trial in trials if trial.sub_ms == '0']

    new_datasets = {}
    full_dataset = [patient_trials, control_trials]

    ## Build datasets at specified frequencies by downsampling eye-traces
    for freq in freqs:
        data = DatasetAtFrequency(freq, full_dataset)
        new_datasets[freq] = data.get_new_data()

    return new_datasets
