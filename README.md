# checkGPclass
Matlab implementation of the methods described in [Robustness Quantification for Classification with Gaussian Processes](https://arxiv.org/abs/1905.11876). We provide example code for adversarial analyses and interpretability analyses of the results described in the paper, i.e., on the Synthetic2D, SPAM and MNIST datasets. 

## Requirements
The main requirement for running the code is of course a:
- Matlab installation (the code was tested on Matlab versions >= 2016a)

Additional toolbox required:
- The [GPML toolbox](http://www.gaussianprocess.org/gpml/code/matlab/doc/) is used for training of two-class GPCs and to perform inference.
- The [GPstuff toolbox](https://research.cs.aalto.fi/pml/software/gpstuff/) is used for training of multi-class GPCs and to perform inference.
- The optimization toolbox is used to solve quadratic programming problems.
- The parallel toolbox may additionally be used to speed up computations.
- The [vlfeat toolbox](http://www.vlfeat.org/install-matlab.html) is used for automatic feature extraction using SIFT. This is used in the experiments on the MNIST dataset.


Additionally, to run the experiments with the MNIST and SPAM dataset it is necessary to first download the two datasets. The code expect the training and testing data to be stored into the `data` directory  as comma separated values:
- `mnist38_train.csv`: [or similar name] a subsample of N images among all the ones included in the MNIST training dataset having classes 3 and 8. Data are assumed to be organised as a Nx785 matrix, where the first element of each row is the class label.
- `mnist38_test.csv`: [or similar name] a similarly formatted test set.

Similar data files are required for the SPAM dataset. More details can be read in the data loading files provided, that is, `utils/data_handling/load_mnist38`, `utils/data_handling/load_mnist358` and `utils/data_handling/load_spam_dataset.m` 

## Usage
