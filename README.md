# mlrap: Machine Learning Regression Analyse Packages
## Table of Contents
+ [Introduction](#Introduction)
+ [Installation](#Installation)
+ [Usage](#Usage)
+ [Contributors](#Contributors)
## Introduction
MLRAP (Machine Learning Regression Analysis Package) is designed to assist materials scientists in constructing regression models and establishing structure-property relationships. This software offers a user-friendly command-line interface and automates the machine learning workflow in Python. It encompasses modules for data preprocessing, feature engineering, model evaluation, model optimization, and model interpretation. This automation significantly reduces the workload for users and enhances the efficiency of materials research and development.  
<img src="https://github.com/NianSan-H/mlrap/blob/master/image/workflow.png" alt="fig" title="workflow">  

## Installation
MLRAP supports Python 3.8+. It is recommended to install it within a virtual environment using virtual environment management tools.  
First create a conda environment: Install miniconda environment from https://conda.io/miniconda.html based on your system requirements. Then, create a virtual environment:  
```
conda create -n mlrap python==3.8.0
conda activate mlrap
```
Extract the compressed archive to a directory, and then execute the following command:  
```
python setup.py install
```
We highly recommend installing in editable mode:  
```
pip install --editable .
```

## Usage
After installation, execute the command `mlrap -h` to view the help interface.  
```
Usage: mlrap [OPTIONS] COMMAND [ARGS]...

  Machine learning regression analyse packages

Options:
  -h, --help  Show this message and exit.

Commands:
  run     Global run base config file.
  subrun  Run step by step.
```
If you have already prepared your dataset (a CSV file containing chemical formulas and target properties), simply execute the command `mlrap run train` in the directory where your dataset is located. MLRAP will automatically perform feature engineering, hyperparameter optimization, and model training for you. Process data will be output in CSV format, and five images will be generated as follows:  

<img src="https://github.com/NianSan-H/mlrap/blob/master/image/output.png" alt="fig" title="output">

### Note
During hyperparameter optimization, we employ Bayesian hyperparameter optimization, where the best hyperparameters are selected for the final model training.  

## Contributors
+ [Gang Tang](https://github.com/obaica)
+ [Tao Hu](https://github.com/NianSan-H)
+ [Chunbao Feng](https://lxy.cqupt.edu.cn/info/1191/6711.htm)

This Project was supported by Beijing Institute of Technology Research Fund Program for Young Scholars (Grant No. XSQD-202222008) and Guangdong Key Laboratory of Electronic Functional Materials and Devices Open Fund (EFMD2023004M).
