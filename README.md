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

The code expect GPML, GPstuff and vlfeat to be stored in the `toolboxes` directory.
Additionally, to run the experiments with the MNIST and SPAM dataset it is necessary to first download the two datasets. The code expect the training and testing data to be stored into the `data` directory  as comma separated values:
- `mnist38_train.csv`: [or similar name] a subsample of N images among all the ones included in the MNIST training dataset having classes 3 and 8. Data are assumed to be organised as a Nx785 matrix, where the first element of each row is the class label.
- `mnist38_test.csv`: [or similar name] a similarly formatted test set.

Similar data files are required for the SPAM dataset. More details can be found in the data loading m-files provided, that is, `utils/data_handling/load_mnist38.m`, `utils/data_handling/load_mnist358.m` and `utils/data_handling/load_spam_dataset.m` 

## Usage
- Adversarial Local safety results (that is Figures 2 and 3 of the paper) can be obtained by running the three `AdversarialLocalSafety` m-files from the main directory of the repo. 
- The Adversarial Local Robustness results (that is Figure 4) can be obtained by iterating the adversarial local safety code for more than one test sample (50 test samples have been used in the paper). The parameters that we changed in the experiments reported in the paper are `train_iter` and `infFun`.
- Interpretability results (that is Figures 5 and 6) can be obtained by running the three `interpretability` m-files from the main directory of the repo. Notice that the MNIST one is implemented for multi-class, thus GPstuff is required for the training part

## Contributors
- [Andrea Patane](https://github.com/andreapatane) 
- [Arno Blaas](https://github.com/arblox) 
