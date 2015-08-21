#### Description
We perform logistic regression and softmax regression on the MNIST dataset.  There are two primary functions of this code.  

* Logisitic regresssion for the classification of handwritten {'0','1'}
* Softmax regression for the classification of handwritten digits {'0',...,'9'}

For reference of the mathematics implemented, please refer to the [Stanford Deep Learning Tutorial](http://ufldl.stanford.edu/tutorial/)

#### Dataset
The data comes from the MNIST dataset of handwritten digits.  Data has been processed from IDX format into Numpy format.

#### Code Base
To run logistic regression, execute
```
python run_logisticregression.py
```

And similarly, to run softmax regression, simply execute
```
python run_softmaxregression.py
```

Math functionality (probability, loss functions, gradients, gradient descents) are stored in the respective files:
* math_logistic.py
* math_softmax.py

Other code for preprocessing may be found in helper_functions.py