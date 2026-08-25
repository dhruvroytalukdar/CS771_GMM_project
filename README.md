# Gaussian Mixture Model Classification on MNIST

## Project overview

This project develops a generative image classifier for the MNIST handwritten-digit dataset using class-conditional Gaussian Mixture Models (GMMs).

Instead of learning a discriminative boundary directly, the project models the distribution of images for each digit class. For every class $c\in\{0,\ldots,9\}$, it learns a mixture of $K$ multivariate Gaussians:

$$
p(x\mid y=c)=\sum_{k=1}^{K_c}\pi_{ck}\,\mathcal{N}(x\mid\mu_{ck},\Sigma_{ck}),
$$

where:

- $x\in\mathbb{R}^{784}$ is a flattened 28-by-28 image,
- $\pi_{ck}$ is the mixture weight for component $k$ of class $c$,
- $\mu_{ck}$ is the component mean image,
- $\Sigma_{ck}$ is the full covariance matrix,
- $K_c$ is the number of Gaussian components assigned to class $c$.

Classification is performed by Bayes/MAP inference:

$$
\hat y=\arg\max_c\left[\log p(y=c)+\log p(x\mid y=c)\right].
$$

The main contribution is the implementation of the GMM classifier and its numerically stable inference pipeline. The workspace also extends the classifier to censored images by hiding a central block of pixels, classifying from the remaining pixels, and reconstructing the missing region using Gaussian conditional means.

## What is included in this curated project folder

The original `lec13` directory has been preserved. This `project_files` folder is a focused copy of the files relevant to GMM classification and its MNIST inference experiments.

### Core classifier implementation

#### `gmm_classification_python.py`

This is the most complete script for the GMM classifier. It contains:

- A NumPy implementation of K-Means for GMM initialization.
- GMM parameter initialization from K-Means clusters.
- Numerically stable multivariate Gaussian log-density evaluation.
- E-step responsibility computation using the log-sum-exp trick.
- M-step updates for mixture weights, means, and covariance matrices.
- Training of one GMM per digit class.
- Bayes/MAP prediction using class priors and mixture likelihoods.
- A binary MNIST IDX file loader.
- Normalization and flattening of 28-by-28 images into 784-dimensional vectors.

The script currently loads the saved model `gmm_params_python_special.pkl` for inference rather than retraining every GMM. The configured per-class component counts are:

```text
K = [5, 20, 1, 10, 1, 2, 1, 2, 5, 5]
```

This means the model has a different number of mixture components for different digit classes, allowing more complex digit distributions to receive more modeling capacity.

### Censored-image classification and reconstruction

#### `gmm_inference.py`

This is the cleanest implementation of the censored-image extension. It:

1. Loads a saved class-conditional GMM model.
2. Zeros a central 12-by-12 region of each test image, removing 144 of the 784 pixels.
3. Uses the remaining 640 observed pixels for classification.
4. Computes the observed-space marginal likelihood for every class and mixture component.
5. Predicts the digit using the maximum posterior class score.
6. Reconstructs the missing pixels using a mixture of Gaussian conditional means.

For each component, the conditional mean of the missing pixels is:

$$
\mathbb{E}[x_m\mid x_o,c,k]
=\mu_m+\Sigma_{mo}\Sigma_{oo}^{-1}(x_o-\mu_o),
$$

where $x_o$ denotes observed pixels and $x_m$ denotes the missing central block. The component conditional means are combined using posterior component responsibilities:

$$
\hat{x}_m=\sum_k\gamma_k\,\mathbb{E}[x_m\mid x_o,c,k].
$$

This file uses Cholesky factorization, covariance submatrices, and log-sum-exp calculations to make the high-dimensional Gaussian computations more stable.

#### `gmm_classification_inferencing.py`

This is an earlier/alternate censored-inference implementation. It includes:

- Full-feature prediction.
- Prediction with missing features.
- Gaussian likelihood evaluation in log space.
- A central-image censoring function.
- An alternate missing-feature reconstruction routine.

It uses the (K=10) saved model in its main execution path. It is useful for understanding the evolution of the implementation, while `gmm_inference.py` is the cleaner reference for the final censored-classification logic.

#### `gmm_inference_separate.py`

This script evaluates multiple component counts:

```text
K = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
```

For each (K), it computes:

- Full-image training accuracy.
- Full-image test accuracy.
- Censored-test accuracy.
- Correct and incorrect reconstruction grids.

It writes reconstruction visualizations into `recon_grids/`. The full version of this experiment requires the original 9 GB `gmm_params/` directory, which was intentionally not duplicated into this curated folder.

### Exploratory notebook

#### `GMM_Classification.ipynb`

This notebook documents the development process:

- Loading the raw MNIST IDX files.
- Reshaping images into 28-by-28 arrays.
- Normalizing and flattening images.
- Implementing K-Means initialization.
- Creating per-class GMM parameters.
- Running EM updates.
- Computing Gaussian log-likelihoods.
- Testing classification performance during EM.
- Saving and reloading learned parameters.

The notebook contains both a single-Gaussian-per-class baseline and the multi-component GMM approach. The multi-component model is the main classification contribution.

### Saved model parameters

The model files store dictionaries keyed by digit class. Each class entry is stored as:

```python
(mu, Sigma, pi)
```

with shapes:

```text
mu:    (K, 784)
Sigma: (K, 784, 784)
pi:    (K,)
```

Included models are:

| File | Intended use |
|---|---|
| `gmm_params_python_5.pkl` | Five-component-per-class censored classification and reconstruction |
| `gmm_params_python_10.pkl` | Ten-component-per-class alternate inference script |
| `gmm_params_python_special.pkl` | Final class-specific component configuration `[5,20,1,10,1,2,1,2,5,5]` |

These files are large because every component stores a full 784-by-784 covariance matrix. They are included so the saved-model inference path can be inspected and reproduced without rerunning the expensive EM training process.

### Accuracy results

#### `gmm_classification_results_python.pkl`

This file stores the component-count sweep:

| Components (K) | Training accuracy | Test accuracy |
|---:|---:|---:|
| 1 | 86.13% | 83.44% |
| 2 | 90.14% | 85.44% |
| 5 | 95.31% | 87.28% |
| 10 | 99.00% | 88.23% |
| 15 | 99.87% | 86.54% |
| 20 | 99.99% | 83.62% |
| 25 | 100.00% | 84.77% |
| 30 | 100.00% | 86.74% |
| 35 | 100.00% | 89.07% |
| 40 | 100.00% | 88.14% |

The best recorded full-image test accuracy in this sweep is **89.07% at (K=35)**. The results also show the bias-variance trade-off: training accuracy reaches 100% as the mixture becomes more expressive, while test accuracy is non-monotonic.

#### `plot_results.py`

This script plots the component-count experiment. It visualizes test accuracy as a function of the number of Gaussian components and highlights that adding components does not guarantee better generalization.

### Visualization artifacts

#### `recon_grids/`

This directory contains saved reconstruction grids for multiple component counts. The images compare reconstructions obtained from correctly and incorrectly classified censored MNIST samples.

### MNIST data

#### `mnist/`

This directory contains the raw MNIST IDX files used by the loaders:

- Training images and labels.
- Test images and labels.
- The nested paths expected by the Python scripts, such as `mnist/train-images-idx3-ubyte/train-images-idx3-ubyte`.

Each image is 28-by-28 grayscale pixels and each flattened sample has 784 features.

### Local helper package

#### `cs771/`

This contains the small plotting/helper package used by the notebooks and visualization scripts:

- `plotData.py`
- `utils.py`
- `__init__.py`

### Environment metadata

#### `pyproject.toml`

This records the Python project metadata and dependencies used in the original environment, including NumPy, Matplotlib, Pandas, Seaborn, and tqdm.

## Detailed GMM classification method

### 1. Data loading and representation

The MNIST files are stored in the standard IDX binary format. The loader validates the magic numbers, reads the image dimensions, converts the byte stream into arrays, and returns the images and labels.

Each image is then:

1. Reshaped into a 28-by-28 matrix.
2. Converted to floating point.
3. Normalized approximately to the range ([0,1]).
4. Flattened into a 784-dimensional vector.

The code uses `/255.0` in the standalone Python classifier and `/256` in parts of the exploratory notebook. For a clean rerun, `/255.0` is the conventional normalization choice.

### 2. Class-conditional modeling

The model is generative rather than discriminative. A separate GMM is fitted for each digit class:

```text
class 0 -> GMM for digit 0
class 1 -> GMM for digit 1
...
class 9 -> GMM for digit 9
```

### 3. K-Means initialization

For each class, K-Means first divides the class-specific training images into (K) clusters. The cluster centroids initialize the Gaussian means. Cluster frequencies initialize the mixture weights, and each cluster covariance initializes the corresponding covariance matrix.

Empty clusters are handled with fallback means and covariance estimates. A diagonal regularizer is added to every covariance:

$$
\Sigma_k\leftarrow\Sigma_k+\lambda I,
$$

which makes the matrices more likely to be invertible in 784 dimensions.

### 4. EM optimization

For every class, EM alternates between:

#### E-step

The responsibility of component $k$ for sample $x_i$ is:

$$
\gamma_{ik}
=
\frac{\pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}
{\sum_j\pi_j\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}.
$$

The implementation computes these values in log space and normalizes them using the log-sum-exp trick.

#### M-step

With effective component count $N_k=\sum_i\gamma_{ik}$, the updates are:

$$
\pi_k=\frac{N_k}{N},
$$

$$
\mu_k=\frac{1}{N_k}\sum_i\gamma_{ik}x_i,
$$

$$
\Sigma_k=\frac{1}{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^T+\lambda I.
$$

The standalone implementation supports a configurable EM iteration limit and covariance regularization.

### 5. Numerically stable inference

Directly evaluating 784-dimensional Gaussian densities can underflow or fail because the covariance determinants and quadratic forms are extreme. The code addresses this by:

- Computing log determinants with `np.linalg.slogdet`.
- Evaluating Gaussian log densities instead of raw densities.
- Adding diagonal covariance regularization.
- Combining mixture components with log-sum-exp.
- Using Cholesky solves in the censored-image implementation instead of explicitly inverting covariance matrices.
- Adding small floors to mixture weights and class priors before taking logarithms.

### 6. MAP prediction

For every test image and digit class, the classifier evaluates:

$$
\log p(y=c)+
\log\left(\sum_k\pi_{ck}\mathcal{N}(x\mid\mu_{ck},\Sigma_{ck})\right).
$$

The predicted digit is the class with the largest score.

## End-to-end execution

### 1. Enter the curated project folder

PowerShell:

```powershell
cd "C:\Users\dhruv\Desktop\IIT-Kanpur\Sem 1\intro_to_ML\lec13\project_files"
```

### 2. Install the required packages

```powershell
python -m pip install numpy matplotlib tqdm jupyter pandas seaborn
```

The core classifier uses NumPy, while Matplotlib and tqdm support plotting and progress reporting. Jupyter is only needed for the notebook workflow.

### 3. Run saved-model full-image classification

```powershell
python gmm_classification_python.py
```

This loads `gmm_params_python_special.pkl`, loads the MNIST data, computes class priors, predicts the test set, predicts the training set, and prints both accuracies.

The script's training block is commented because fitting full 784-by-784 covariance matrices is computationally expensive. To retrain, enable the call to `train_gmm_classifier()`, choose the desired per-class component counts, and save the resulting parameters before running inference.

### 4. Run censored-image classification and reconstruction

```powershell
python gmm_inference.py
```

This uses the five-component model, removes the central 12-by-12 pixel block, classifies using only the observed pixels, reconstructs the missing pixels with the GMM conditional mean, and prints censored-test accuracy.

### 5. Inspect the alternate K=10 censored implementation

```powershell
python gmm_classification_inferencing.py
```

This runs the alternate missing-feature classifier using `gmm_params_python_10.pkl` and reports accuracy on centrally censored images.

### 6. Explore the notebook implementation

```powershell
jupyter notebook GMM_Classification.ipynb
```

Run the notebook cells in order to reproduce the data-loading, K-Means initialization, GMM parameter creation, EM updates, inference, and model serialization workflow. Full training is computationally heavy because it operates in the original 784-dimensional pixel space with dense covariance matrices.

### 7. Plot the component-count results

```powershell
python plot_results.py
```

This opens a plot of accuracy against the number of Gaussian components. The numerical source for the full-image sweep is `gmm_classification_results_python.pkl`.

### 8. Reproduce the complete K sweep

The original folder contains a `gmm_params/` directory with saved models for $K\in\{1,2,5,10,15,20,25,30,35,40\}$. It is approximately 9 GB and was deliberately not duplicated into this showcase folder. To run `gmm_inference_separate.py` across every $K$, run it from the original `lec13` folder or copy the required parameter files into `project_files/gmm_params/`.

## Important implementation notes

- The project uses full covariance matrices, which are expressive but expensive in 784 dimensions.
- The number of mixture components is a model-capacity parameter: larger (K) improves training fit but may overfit.
- The saved sweep reaches 100% training accuracy for larger (K), while the best recorded test accuracy is 89.07% at (K=35).
- The class-specific model in `gmm_params_python_special.pkl` uses different component counts per digit rather than one global K.
- `gmm_inference.py` is the preferred reference for censored classification because it uses observed-space marginal likelihoods, Cholesky factorization, and mixture-of-conditionals reconstruction.
- The notebook is valuable as a development record, but the standalone Python scripts are easier to rerun and inspect.
- This curated folder was copied from the original workspace; the original source files remain unchanged.
