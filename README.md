# SenseMS
<br>

***SenseMS*** was completed as an Insight Artifical Intelligence Fellow, consulting for a company that uses AI and eye-motion data to detect neurodegenerative diseases, like multiple sclerosis.
<br>

By examining classification accuracy of a neural network on downsampled sensor data, I mimiced using cheaper hardware to help the company decide how to productionize their device. ***SenseMS*** helped the company move one step closer to going to market: a $15B opportunity.

<br>

Eye-tracking time-series data *(x, y, t)*    |  Downsampled eye-traces
:------------------------:|:-------------------:
![Eye-tracking](https://media.giphy.com/media/blle4NCmxmMne/giphy.gif)  |  ![Model](img/downsampling.gif)
<br>

Unfortunately, due to the highly sensitive nature of the data and model confidentiality, I am not able to open-source this project. This repository was created for the benefit of the company I consulted for to conduct future testing.
<br>

`utils/build_datasets.py` takes list of frequencies and returns dictionary of downsampled eye-traces at each of the specified frequencies using `utils/downsample.py`.
<br>

As EDA showed that almost all of the pre-built model's signal was strangely coming from age data, `utils/create_data_for_testing.py` takes the downsampled eye-traces and builds datasets for testing at each frequency, as defined in `utils/datasets.py`, representing:
<center>

	1. All data
	2. All data except age
	3. Age only 
</center>
<br>

`logisticregr_tests.py`, `gb_tests.py`, `rf_tests.py`, and `xgb_test.py` use the training data to return cross-validated accuracy scores for each model. These tests were used for sanity checks when the DNN was returning unusual results.
<br>

`train_model.py` demonstrates usage of the NN model as well as basic results from simpler ML models. These results also contain work verifying suspicions of data leakage in the pre-existing NN architecture.
<br>

`process_results.py` takes the results of `train_model.py` and generates plots stored in the `results` directory.
<br>

`exploration.html` shows the exploration that helped to identify data leakage.
<br>

## Dependencies
Anaconda is required. Download [here](https://conda.io/en/latest/miniconda.html) and install with:
```
bash Anaconda-latest-Linux-x86_64.sh
```

## Setup
Create a conda environment called `sensorframerate`:

```
conda env create -f build/environment.yml
conda activate sensorframerate
```

## Test
- After loading the conda environment, propogate the `data`, `models`, and `params` directories with the company's modules and run:

```
python logisticregr_tests.py &&
python gb_tests.py &&
python rf_tests.py &&
python xgb_test.py
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results

```
# Step 1
# Step 2
```
