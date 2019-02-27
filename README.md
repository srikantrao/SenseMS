# SenseMS
*****Private GitHub only available to C. Light*****

Since this work was part of a private Insight fellowship project, the model structure and training code could not be open-sourced.

Eye-tracking data follows the movement of the eye: <br />
![Eye-tracking](https://media.giphy.com/media/blle4NCmxmMne/giphy.gif)

I downsampled the eyetraces from 480 Hz down to 8 Hz to create new datasets for training:
![Model](img/downsampling.gif)

This repository contains results from deep-dive into the classification accuracy of the current model; inference can be run on GPU using the frozen graph.

`utils/build_datasets.py` takes list of frequencies and returns dictionary of downsampled Eyetrace objects at each of the specified frequencies using `utils/downsample.py`.

`create_data_for_testing.py` takes the downsampled eye-traces and builds three
dictionaries of training data, defined in `utils/datasets.py`, to be fed into
the models representing:
	- All data,
	- All data except Age, and
	- Age only

`logistic_tests.py`, `RF_tests.py`, and `GB_tests.py` use the training data to return cross-validated accuracy scores for each model. These tests were used for sanity checks when the DNN was returning unusual results.

`train_model.py` demonstrates usage of the NN model as well as basic results from simpler ML models. These results also contain work verifying suspicions of data leakage in the pre-existing NN architecture.

`process_results.py` takes the results of `train_model.py` and generates graphs stored in the `results` directory.

`exploration.html` shows the exploration that resulted in data leakage findings.



## Dependencies
Anaconda or Miniconda are required. Download [here](https://conda.io/en/latest/miniconda.html) and install with:
```
bash Miniconda3-latest-Linux-x86_64.sh
bash Anaconda-latest-Linux-x86_64.sh
```


## Setup
Create a conda environment called `sensorframerate`:
```
conda env create -f build/environment.yml
conda activate sensorframerate
```


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```


## Run
If you have a GPU, select sample input data and a simple model to replicate the process I used to create downsampled datasets.
```
python run_wavernn.py samples/LJ016-0277.wav
```


## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
