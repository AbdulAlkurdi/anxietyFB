# Set up environment


## Build the environment
Run the following commands to create and activate your environment
```
conda env create -f environment_mod.yml -n TFwesad
conda activate TFwesad
```
 
## Python Version Check
Check the version of python (the output should be Python 3.8.13)
```
python --version
```

## Install additional packages:
Run the following commands to install neccesary packages
```
pip install git+https://www.github.com/keras-team/keras-contrib.git
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
pip install git+https://github.com/xflr6/graphviz.git@0.13.2
```
