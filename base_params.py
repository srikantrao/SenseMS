import os
import numpy as np

class Base_Params(object):
    # Meta
    model_name = "name"
    trial_split_multiplier = None # split trials into multiplier more samples (accepts None as 1)
    truncate_trials = False # if True, truncate all tirals to the length of the shortest
    flip_trials_on_y_axis = False # if True, double dataset size by flipping all trials on y axis
    num_train = None # use the rest after setting up val & test
    num_val = 50
    num_test = 50
    num_epochs = 1
    batch_size = 1
    num_crossvalidations = 5
    # Other
    max_cp_to_keep = 1
    val_frequency = 5 # after how many epochs should the validation be run
    eps = 1e-10
    rand_seed = 12345
    rand_state = np.random.RandomState(rand_seed)
    out_dir = os.path.expanduser("~")+"/models/"
    device = "/gpu:0"
