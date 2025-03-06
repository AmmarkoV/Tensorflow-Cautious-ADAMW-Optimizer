# Cautious ADAMW Optimizer for Keras/Tensorflow

This is an implementation for the paper : [Cautious Optimizers: Improving Training with One Line of Code](https://arxiv.org/abs/2411.16085)

###Creating a python environment

To create a python3 venv to try this:
```
sudo apt install python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install tensorflow==2.18.0 tf_keras
```

###Test

To perform a small check and see if the optimizer use:
```
source venv/bin/activate
python3 AdamWCautious.py
```

For CPU/GPU/CUDA/CUDNN compatibility check: 
#https://www.tensorflow.org/install/source#linux

