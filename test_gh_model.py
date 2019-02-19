# Libraries
import numpy as np
import os
import pickle as p
import sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Modules
import build_datasets
from temp_blur import DatasetAtFrequency
import utils.data_formatutils as dfu
import utils.readinutils as readinutils
from utils.dataset import Dataset
from utils.dataset import DatasetGroup

## Models
from models.fc_mlp import FC_MLP as fc_mlp

## Params
from params.model_params import params


"""--------------------------------------
  Model specification is done here:
--------------------------------------"""
model_type = "fc_mlp"
"""--------------------------------------
--------------------------------------"""

"""--------------------------------------
  Data path specification is done here:
--------------------------------------"""
datadir = os.path.expanduser("~")+'/envision_working_traces/'
patientfile = './data/patient_stats.csv'
"""--------------------------------------
--------------------------------------"""

"""--------------------------------------
  Params sweeping options
--------------------------------------"""
sweep_params = dict()
sweep_params["model_name"] = ["split_30_arch_60_40_30_20_10_2_5crossval"]
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

# freqs for downsampling
freqs = np.arange(8, 481, 8)

# build datasets
new_datasets = build_datasets.make_datasets(freqs)

results = {}
for freq, data in new_datasets.items():
    patient_trials = data[0]
    control_trials = data[1]
    print("length of points in first patient trial", patient_trials[0].num_strips)
    if(len(patient_trials + control_trials) != 0):
        data, labels, stats = dfu.make_mlp_eyetrace_matrix(patient_trials, control_trials, params.fft_subsample)
        dataset_group_list = DatasetGroup(data, labels, stats, params)

        if hasattr(params, "flip_trials_on_y_axis") and params.flip_trials_on_y_axis:
            dataset_group_list.flip_trials_y()
        if hasattr(params, "reduce_data") and params.reduce_data:
            dataset_group_list.pca_whiten_reduce(params.required_explained_variance, params.rand_state)
        if hasattr(params, "concatenate_age") and params.concatenate_age:
            dataset_group_list.concatenate_age_information()
        if hasattr(params, "concatenate_fft") and params.concatenate_fft:
            dataset_group_list.concatenate_fft_information()
        if hasattr(params, "concatenate_vel") and params.concatenate_vel:
            dataset_group_list.concatenate_vel_information()
        if hasattr(params, "concatenate_nblinks") and params.concatenate_nblinks:
            dataset_group_list.concatenate_nblinks_information()

        params.data_shape = list(dataset_group_list.get_dataset(0)["train"].data.shape[1:])

        print("Training on", dataset_group_list.get_dataset(0)["train"].data.shape[0], "data points.")
        print("Validating on", dataset_group_list.get_dataset(0)["val"].data.shape[0], "data points.")
        print(freq, "Hz")

        ## Training
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        num_sweeps = np.prod([len(val) for val in sweep_params.values()])
        keys, values = zip(*sweep_params.items())
        for experiment_number, experiment in enumerate(itertools.product(*values)):
            print("\n\n-------\nStarting experiment", experiment_number+1, " out of ", num_sweeps)

            ## Set new params for this experiment
            for key, value in zip(keys, experiment):
                print(key, "is set to", value)
                setattr(params, key, value)

            dataset_group_list.reset_counters()
            dataset_group_list.init_results()

            ## Model selection
            if model_type == "conv_mlp":
                model = conv_mlp(params)
            elif model_type == "fc_mlp":
                model = fc_mlp(params)
            elif model_type == "reg":
                model = reg(params)
            else:
                assert False, ("model_type must be 'conv_mlp', 'fc_mlp', or 'reg'")

            ## Perform Monte Carlo Cross Validation (resample and train multiple times)
            while dataset_group_list.crossvalids_completed < params.num_crossvalidations:
                #get the dataset for this draw
                data = dataset_group_list.get_dataset(dataset_group_list.crossvalids_completed)

                print(f"Crossvalidation run {dataset_group_list.crossvalids_completed + 1} / {params.num_crossvalidations}")
                sys.stdout.flush()
                ## TODO: Store these in Dataset structure (don't want to create conflicts now)
                pulls_max_acc = 0 #best validation accuracy
                pulls_max_acc_sens = 0 #sensitivity at best val accuracy
                pulls_max_acc_spec = 0 #specificty at best val accuracy

                ## New Session for each Draw
                with tf.Session(config=config, graph=model.graph) as sess:
                    ## Need to provide shape if batch_size is used in graph
                    sess.run(model.init_op,
                        feed_dict={model.x:np.zeros([params.batch_size]+params.data_shape,
                        dtype=np.float32)})
                    model.write_graph(sess.graph_def)
                    sess.graph.finalize() # Graph is read-only after this statement

                    while data["train"].epochs_completed < params.num_epochs:
                        data_batch = data["train"].next_batch(model.batch_size)
                        #print("dataset as x",data_batch[0])
                        feed_dict = {model.x:data_batch[0], model.y:data_batch[1]}

                        ## Update weights
                        sess.run(model.apply_grads, feed_dict)

                        if (data["train"].epochs_completed % params.val_frequency == 0
                            and data["train"].batches_this_epoch==1):

                            global_step = sess.run(model.global_step)
                            current_loss = sess.run(model.total_loss, feed_dict)

                            weight_cp_filename, full_cp_filename = model.write_checkpoint(sess)

                            with tf.Session(graph=model.graph) as tmp_sess:
                                val_feed_dict = {model.x:data["val"].data, model.y:data["val"].labels}
                                tmp_sess.run(model.init_op, val_feed_dict)
                                cp_load_file = tf.train.latest_checkpoint(model.cp_save_dir,
                                    model.cp_latest_filename+"_weights")
                                model.load_weights(tmp_sess, cp_load_file)
                                run_list = [model.merged_summaries, model.accuracy, model.sensitivity, model.specificity]
                                summaries, val_accuracy, val_sensitivity, val_specificity = tmp_sess.run(run_list,
                                    val_feed_dict)
                                model.writer.add_summary(summaries, global_step)

                            with tf.Session(graph=model.graph) as tmp_sess:
                                tr_feed_dict = {model.x:data["train"].data, model.y:data["train"].labels}
                                tmp_sess.run(model.init_op, tr_feed_dict)
                                cp_load_file = tf.train.latest_checkpoint(model.cp_save_dir,
                                    model.cp_latest_filename+"_weights")
                                model.weight_saver.restore(tmp_sess, cp_load_file)
                                train_accuracy = tmp_sess.run(model.accuracy, tr_feed_dict)

                            ## check for best accuracy
                            ## TODO: Think about doing this more often to reduce likelyhood of missing best value
                            if(pulls_max_acc < val_accuracy):
                                pulls_max_acc = val_accuracy
                                pulls_max_acc_sens = val_sensitivity
                                pulls_max_acc_spec = val_specificity

                            num_decimals = 5
                            print("epoch:", str(data["train"].epochs_completed).zfill(3),
                                 "\tbatch loss:", np.round(current_loss, decimals=num_decimals),
                                 "\ttrain accuracy:", np.round(train_accuracy, decimals=num_decimals),
                                 "\tval accuracy:", np.round(val_accuracy, decimals=num_decimals),
                                 "\tval sensitivity:", np.round(val_sensitivity, decimals=num_decimals),
                                 "\tval specificity:", np.round(val_specificity, decimals=num_decimals))
                            sys.stdout.flush()

                    ## Report results for this pull so we can calculate mean and sd for cross validation
                    dataset_group_list.record_results(train_accuracy, val_accuracy, val_sensitivity, val_specificity,
                                                      pulls_max_acc, pulls_max_acc_sens, pulls_max_acc_spec,
                                                      dataset_group_list.crossvalids_completed)

            ## Report Cross Validated Result
            # result is array trainacc,valacc,sens,spec, maxtracc, sensatmaxtracc, specatmaxtracc
            xvald_means = dataset_group_list.mean_results()
            xvald_sds = dataset_group_list.sd_results()
            results[freq] = xvald_means
            print("Cross Validation Completed!", freq, "Hz",
                "\nMean Final Train accuracy:", np.round(xvald_means[0], decimals=num_decimals), "(SD:",np.round(xvald_sds[0],decimals=num_decimals),")",
                "\nMean Final Val accuracy:", np.round(xvald_means[1], decimals=num_decimals), "(SD:",np.round(xvald_sds[1],decimals=num_decimals),")",
                "\nMean Final sensitivity:", np.round(xvald_means[2], decimals=num_decimals), "(SD:",np.round(xvald_sds[2],decimals=num_decimals),")",
                "\nMean Final specificity:", np.round(xvald_means[3], decimals=num_decimals), "(SD:",np.round(xvald_sds[3],decimals=num_decimals),")",
                "\nMean Best Val accuracy:", np.round(xvald_means[4], decimals=num_decimals), "(SD:",np.round(xvald_sds[4],decimals=num_decimals),")",
                "\nMean Best sensitivity at Best Val acc:", np.round(xvald_means[5], decimals=num_decimals), "(SD:",np.round(xvald_sds[5],decimals=num_decimals),")",
                "\nMean Best specificity at Best Val acc:", np.round(xvald_means[6], decimals=num_decimals), "(SD:",np.round(xvald_sds[6],decimals=num_decimals),")")

            p.dump(results, open("results.p", "wb"))
