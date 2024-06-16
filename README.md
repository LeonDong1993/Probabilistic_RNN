# About
This repository contains all codes, datasets for the paper `A New Modeling Framework for Continuous, Sequential Domains` published in AISTATS 2023.

In this work, we propose a novel temporal probabilistic model for continuous domains that is able to handle long term dependencies 
by leveraging power of Recurrent Neural Networks. We show that various inference tasks can be efficiently implemented using forward filtering with simple gradient ascent. 


# Getting Started
## Setup

- Clone this repository to your local disk.
- Download the zipped dataset from [Here][data_url] and extract all content into the `data` folder.
- Source `core/set_env.profile` before executing any command. Otherwise, import error will be occurred.
- Install necessary dependencies if prompted (such as `networkx`).

## File Structure
- `core` contains the some custom implemented library function and modules specifically for this project. That's why you need to source `core/set_env.profile` before use this set of codes. 
- `data` is currently empty, please download the dataset from the link in `Setup` section and extract all contents to this directory.
- `models` contains all implementation of our model and the competitors. 
- `output` contains some output generated for specific experiment. It also contains the raw results we reported in the paper. 
- `analysis.ipynb` is the jupyter notebook used to analysis the results of sequence completion and trajectory prediction experiment. It also generates the graphs (REC curve and summary bar plot) shown in the paper. 
- `main.py` is the main script used for running experiment. It can take multiple command line arguments and the details usage can be checked by running it without any argument. 
- `run.sh` is the bash script used for training and prediction task on all of the tasks. 
- `train.py` is the script used for train all four models using Optuna in parallel.

## Data Format
- All datasets are first pre-processed, formatted and then being pickled into a `.fdt` file. 
- You can load the data through `pickle.load`, and it is tuple of train and test data where each of them is a list of numpy arrays. 
- Each numpy array is essentially a sequence where the row represent the time axis and column represent the variables.

For more information about how to create such a pickled dataset, please refer to the python scripts located in the `data` directory.
If you need to train our model on new dataset, please prepare the data into the similar format. Note that the length of these sequences can be different in our model. 

## Run Experiments
We use `main.py` as our entry script for all evaluating experiments. It takes three command line inputs:
1. the path to the data file, it should be `.fdt` file
2. the output directory
3. the trained model checkpoint file

We use `train.py` for training and tuning the models, it usually takes two parameters.
1. the path to the data file
2. the output directory

The commands in `run.sh` have demonstrated the basic usage of our codes for training or predicting. 



# Results


![experiment results on trajectory prediction][res_trajpred]


![experiment results on sequence completion][res_seqcomp]



# Citation
Please cite our work if you find it is helpful for your research!

```
@InProceedings{pmlr-v206-dong23a,
  title = 	 {A New Modeling Framework for Continuous, Sequential Domains},
  author =       {Dong, Hailiang and Amato, James and Gogate, Vibhav and Ruozzi, Nicholas},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {11118--11131},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v206/dong23a/dong23a.pdf},
  url = 	 {https://proceedings.mlr.press/v206/dong23a.html},
  abstract = 	 {Temporal models such as Dynamic Bayesian Networks (DBNs) and Hidden Markov Models (HMMs) have been widely used to model time-dependent sequential data. Typically, these approaches limit focus to discrete domains, employ first-order Markov and stationary assumptions, and limit representational power so that efficient (approximate) inference procedures can be applied. We propose a novel temporal model for continuous domains, where the transition distribution is conditionally tractable: it is modelled as a tractable continuous density over the variables at the current time slice only, while the parameters are controlled using a Recurrent Neural Network (RNN) that takes all previous observations as input. We show that, in this model, various inference tasks can be efficiently implemented using forward filtering with simple gradient ascent. Our experimental results on two different tasks over several real-world sequential datasets demonstrate the superior performance of our model against existing competitors.}
}
```


# Contact
If you have any questions or need help regarding our work, you can email us and we are happy to discuss the work (the email addresses of each author are included in the paper). 

In case my school email being deactivated, you can email me using my personal email address `HailiangDong@hotmail.com`.


[data_url]:https://utdallas.box.com/s/73kq6gu9v5depnx7z6zat9el429nsfz5
[res_seqcomp]:https://github.com/LeonDong1993/Probabilistic_RNN/blob/main/figs/res-seqcomp.png
[res_trajpred]:https://github.com/LeonDong1993/Probabilistic_RNN/blob/main/figs/res-trajpred.png




