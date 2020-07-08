# dynamic-inversion
Code for Podlaski &amp; Machens 2020

Last updated: 11 June 2020

***Stay tuned for pytorch and tensorflow implementations

This folder contains five python files (.py), and one empty subfolder (data).

Code dependencies:
- Python libraries: numpy, matplotlib, scipy, numba, mlxtend
- MNIST dataset: download the training and test datasets (http://yann.lecun.com/exdb/mnist/) into the 'data' subfolder

General notes:
- IMPORTANT: Files are run with a random seed, and occasionally will produce differing results. Notably, the SSA optimization algorithm used to stabilize the weight matrices will sometimes (very rarely) fail, causing the script to crash. When this happens, please re-run.
- Model type acronyms: backpropagation, BP; feedback alignment, FA; pseudobackpropagation, PBP; non-dynamic inversion, NDI; dynamic inversion, DI; single-loop dynamic inversion (SDI).
- Hyper-parameters for all files
	- model_name: unique name for all models to be run
	- model_type: specifies the type of error backprop algorithm to be used, one of the model acronyms above
	- fback_type: initialization of feedback weights -- 'random', random Gaussian; 'neg-transpose', B=-W^T; 'optim': random Gaussian, followed by stability optimization with the SSA algorithm (in sim_tools.py)
	- leak_vals: alpha value for NDI and DI algorithms
	- eta: learning rate
	- w_decay: additional regularizer decay on weight

File descriptions: file runtimes were estimated on a MacBook (3GHz i7, 16GB RAM)
- sim_tools.py: contains helper functions for the other four files, including dynamic inversion algorithms
- linear_regression.py: reproduces Fig. 3a-f from the main paper. Note that for Fig. 3b, the variable norm_delW must be set to True. This file takes ~1-2 minutes to run.
- nonlinear_regression.py: reproduces Fig. 3g-k from the main paper. This file takes ~30-40 minutes to run 10000 iterations.
- mnist_classification.py: reproduces Fig. 4a from the main paper. This file takes ~60 minutes per epoch.
- mnist_autoencoding.py: reproduces Fig. 4b from the main paper. Note the parameter w_init_type ('normal' or 'orthogonal') -- File has to be run twice for each weight initialization. This file takes ~45 minutes per epoch.
