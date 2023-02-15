# About
This repository contains all codes, datasets for the paper `A New Modeling Framework for Continuous, Sequential Domains` published in AISTATS 2023.


# Setup
- Clone this repository to your local disk.
- Download the zipped dataset from [Here](https://drive.google.com/file/d/1_iGbjvUxCrJrZ_zaxVSciqbgUmoPfSgk/view?usp=share_link) and extract all content into the `data` folder.
- Source `core/set_env.profile` before executing any command. Otherwise, import error will be occurred.
- Install other dependencies prompted if needed (such as `networkx`).

# Usage
- `core` contains the some custom implemented library function and modules specifically for this project. That's why you need to source `core/set_env.profile` before use this set of codes. 
- `data` is currently empty, please download the dataset from the link in Setup and extract all contents to this directory.
- `models` contains all implementation of our model and the competitors. 
- `output` contains some output generated for specific experiment. It also contains the raw results we reported in the paper. 
- `analysis.ipynb` is the jupyter notebook used to analysis the results of sequence completion and trajectory prediction experiment. It also generates the graphs (REC curve and summary bar plot) shown in the paper. 
- `main.py` is the main script used for running experiment. It can take multiple command line arguments and the details usage can be checked by running it without any argument. 
- `run.sh` is the bash script used for training and prediction task on all of the tasks. 
- `train.py` is the script used for train all four models using Optuna in parallel.

The command in `run.sh` have demonstrated the basic usage of our codes for training or predicting. 


If you need to train our model on new dataset, please prepare the data into the similar format. The data is a picked file where the data structure is a tuple like (train, test). For train/test, it is a list of numpy array where each array represent a sequence. Note that the length of these sequences can be different in our model. 


