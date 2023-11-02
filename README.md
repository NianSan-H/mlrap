# mlrap: Machine Learning Regression Analyse Packages
MLRAP (Machine Learning Regression Analysis Package) is a specialized software tool designed to assist materials scientists in building regression models and establishing structure-property relationships. The software features a user-friendly command-line interface and automates the machine learning workflow in Python. It includes modules for data preprocessing, feature engineering, model evaluation, model optimization, and model interpretation, significantly reducing user workload and enhancing the efficiency of materials research and development.  
<img src="https://github.com/NianSan-H/mlrap/blob/master/image/workflow.png" alt="fig" title="workflow">  
## Installation
MLRAP supports Python 3.8+. It is recommended to install it within a virtual environment using virtual environment management tools.  
First create a conda environment: Install miniconda environment from https://conda.io/miniconda.html based on your system requirements. Then, create a virtual environment:  
```
conda create mlrap
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
## Getting started

```
mlrap -h
```
<img src="https://github.com/NianSan-H/mlrap/blob/master/image/help.png" alt="fig" title="workflow">

## Contributors
+ [Gang Tang](https://github.com/obaica)
+ [Tao Hu](https://github.com/NianSan-H)
+ [Chunbao Feng](https://lxy.cqupt.edu.cn/info/1191/6711.htm)
