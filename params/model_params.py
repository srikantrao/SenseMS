import numpy as np
from params.base_params import Base_Params

class params(Base_Params):
    # Meta
    model_name = "fc_mlp"
    num_val = 500
    num_test = 100
    num_epochs = 800
    num_crossvalidations = 5
    batch_size = 500
    vectorize = True # whethe or not we should vectorize the data, collapsing all dim
    trial_split_multiplier = 30 # split trials into multiplier more samples (accepts None as 1)
    truncate_trials = True # if True, truncate all tirals to the length of the shortest
    reduce_data = True # if True, perform dimensionality reduction & whitening on data
    # (scalar from 0 to 1) Reduce dimensionality such that this much data can still be explained
    required_explained_variance = 0.995
    concatenate_age = True # if True, concatenate the age of the subject as a feature
    concatenate_fft = True #if True, concatenate the Fourier Transform as a feature
    concatenate_vel = True #if True, concatenate the average velocity as a feature
    concatenate_nblinks = False #if True, concatenate the number of blinks as a feature
    fft_subsample = 100 #number of FT components to use (likely want less than other features)
    flip_trials_on_y_axis = True # if True, double dataset size by flipping all trials on y axis
    num_train = None # use the rest after setting up val & test
    # Architecture
    #fc_output_channels = [30, 20, 2]
    fc_output_channels = [60, 40, 20, 10, 2]
    # Training
    optimizer_type = "sgd"
    weight_learning_rate = 5e-4
    lr_staircase = False
    lr_decay_rate = 0.98
    lr_decay_steps = num_epochs*0.8
    # Regularizers
    weight_decay_mults = [1e-6 for idx in range(len(fc_output_channels))]
    orth_mults = [1e-5 for idx in range(len(fc_output_channels))]
    do_batch_norm = True
    # exponential moving average decay multiplier for batch norm
    # lower is "less forgetful"
    norm_decay_mult = 0.4
    # Other
    val_frequency = 10 # after how many epochs should the validation be run
    rand_seed = 123456789
    rand_state = np.random.RandomState(rand_seed)
