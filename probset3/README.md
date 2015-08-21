#### Description
Motivated by the [Stanford Deep Learning Tutorial](http://ufldl.stanford.edu/tutorial/), we train a neural networks for the two tasks

* Identification of individuals from the NimStim dataset
* Identification of emotions from the POFA dataset

The primary code is in multilayer_supervised, but functionality borrowed from the tutorial, specifically, minFunc is utilized from common.  The mathematics behind the neural network, implemented in multilayer_supervised/supervised_dnn_cost.m, is found in [Multilayer Supervised Network](http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/) and [Supervised Neural Networks](http://ufldl.stanford.edu/tutorial/supervised/ExerciseSupervisedNeuralNetwork/).

#### Code Base
In multilayer_supervised/run_train, configure the parameters of a neural network, select either the NimStim or POFA dataset and then train and test simply by executing from MATLAB command window,  

```
run_train
```

If the requisite data preprocessing has not yet been done, MATLAB will also call,

```
prepare_data
```

Which will considerably extend the run-time of the program.  This will not be called each time run_train is called, provided the data and labels are in the workspace. There are two optimizations available, minFunc and Stochastic Gradient Descent.  
