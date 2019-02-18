# SenseMS
*****Private GitHub only available to C. Light*****

Since this work was part of a private Insight fellowship project, the model structure and training code could not be open-sourced.

![Model](img/downsampling.gif)

## Motivation for this project format:
- **Noise-DNN-And-Sensor-Efficacy** : Put all source code for production within structured directory
- **tests** : Put all source code for testing in an easy to find location
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : Include example a small amount of data in the Github repository so tests can be run to validate installation
- **build** : Include scripts that automate building of a standalone environment
- **static** : Any images or content to include in the README or web framework if part of the pipeline

This repository contains results from deep-dive into the classification accuracy of the current model; inference can be run on GPU using the frozen graph.

`logistic_tests.py`, `RF_tests.py`, and `GB_tests.py` take an input WAV file, applies an FFT to produce an 80-band
mel spectrogram, then uses the spectrogram to generate 16 kHz audio with a
frozen WaveRNN model.

`build_datasets.py` take...

`cudnn_gru.ipynb` demonstrates usage of the NN model as well as basic results from simpler ML models. These results also contain work verifying suspicions of data leakage in the pre-existing NN architecture.


## Setup
Create a virtualenv, activate then run
```
pip install -r requirements.txt
```

## Requisites

- List all packages and software needed to build the environment
- This could include cloud command line tools (i.e. gsutil), package managers (i.e. conda), etc.

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
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
