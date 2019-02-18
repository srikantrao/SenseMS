# SenseMS
*****Private GitHub only available to C. Light*****

Consulting project to show the effects of noise on a deep neural network used to classify multiple sclerosis. Since this work was part of a private Insight fellowship project, the model structure and training code could not be open-sourced.

![Model](img/downsampling.png)

## Motivation for this project format:
- **Noise-DNN-And-Sensor-Efficacy** : Put all source code for production within structured directory
- **tests** : Put all source code for testing in an easy to find location
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : Include example a small amount of data in the Github repository so tests can be run to validate installation
- **build** : Include scripts that automate building of a standalone environment
- **static** : Any images or content to include in the README or web framework if part of the pipeline

This repository contains a demo of a WaveRNN-based waveform generator trained
on 13000 spoken sentences; inference can be run on CPU or GPU using the frozen
graph.

`run_wavernn.py` takes an input WAV file, applies an FFT to produce an 80-band
mel spectrogram, then uses the spectrogram to generate 16 kHz audio with a
frozen WaveRNN model.

`cudnn_gru.ipynb` demonstrates usage of the (poorly documented) CuDNN GRU cell
in TensorFlow, and some of the tricks and workarounds needed to get the network
up and running. See [this link](http://htmlpreview.github.io/?https://github.com/austinmoehle/wavernn/blob/master/cudnn_gru.html)
for an HTML-rendered version.


## Setup
On AWS EC2 Ubuntu Deep Learning AMI version 20, run:
```shell
bash environment.sh
```

## Requisites

- List all packages and software needed to build the environment
- This could include cloud command line tools (i.e. gsutil), package managers (i.e. conda), etc.

#### Dependencies

- [Streamlit](streamlit.io)

#### Installation
To install the package above, please run:
```shell
pip install -r requirements
```

## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

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
