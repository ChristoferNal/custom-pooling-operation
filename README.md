# custom-pooling-operation
Learn how to create a pooling operation from scratch using Pytorch (python) or building your own C++ extension.

The tutorial in [a relative link](custom_pooling.ipynb) includes:

- Implementation of max pool using the python API of pytorch.
- Implementation of max pool using the C++ API of pytorch and instructions on how to build a python binding.
- Performance comparison of the custom max pool in python, the C++ extension and the native pytorch max pool operation.

## Setup

Install the both the python and the C++ distribution of pytorch. 
Instructions for the C++ distribution can be found here https://pytorch.org/cppdocs/installing.html
In order to build the custom model run the following commands from your pytorch environment:

- cd /cpp
- python setup.py install 

You should get something like this:

Installed ~/anaconda2/envs/torch/lib/python3.7/site-packages/pooling_cpp-0.0.0-py3.7-linux-x86_64.egg
Processing dependencies for pooling-cpp==0.0.0
Finished processing dependencies for pooling-cpp==0.0.0
