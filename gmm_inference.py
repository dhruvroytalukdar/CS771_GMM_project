import pickle
import numpy as np
import struct
from os.path import join
from array import array
from numpy.linalg import inv
from cs771 import plotData as pd
import matplotlib.pyplot as plt

# --- Provided helpers (unchanged) ---
def censorImages(X):
    # Wipe out 21% of the pixels from the central part of the image
    X[:, 8:20, 8:20] = 0
    return X

def truncatePixels(X, low=0, high=1):
    X[X < low] = low
    X[X > high] = high
    return X

# --- Index helpers for the fixed missing window ---
def missing_mask_28x28():
    mask = np.zeros((28,28), dtype=bool)
    mask[8:20, 8:20] = True
    return mask

_MISS_MASK_2D = missing_mask_28x28()
_MISS_IDX = np.flatnonzero(_MISS_MASK_2D.ravel())      # missing indices (length 144)
_OBS_IDX  = np.flatnonzero(~_MISS_MASK_2D.ravel())     # observed indices (length 640)
d_obs = len(_OBS_IDX)

# --- Stable log N(x | mu, Sigma) via Cholesky on the observed subspace ---
def log_gaussian_obs(x_obs, mu_obs, Sigma_obs, jitter=1e-6):
    # x_obs: (d_o,), mu_obs: (d_o,), Sigma_obs: (d_o, d_o)
    # returns scalar log density
    # Add small ridge to ensure PSD
    S = Sigma_obs + jitter * np.eye(Sigma_obs.shape[0])
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # If still fails, bump jitter
        L = np.linalg.cholesky(S + 1e-3 * np.eye(S.shape[0]))
    v = x_obs - mu_obs
    # Solve L * w = v, then compute ||w||^2
    w = np.linalg.solve(L, v)
    quad = np.dot(w, w)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d_obs * np.log(2.0*np.pi) + logdet + quad)

def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

# --- Per-class marginal log-likelihood of observed pixels ---
def class_log_marginal(models_c, x_obs):
    # models_c: dict with 'pi' (K,), 'mu' (K,d), 'Sigma' (K,d,d)
    pi, mu, Sigma = models_c[2], models_c[0], models_c[1]
    K, d = mu.shape
    mu_o = mu[:, _OBS_IDX]                 # (K, d_o)
    # Pre-slice Sigma_oo for all k
    # (K, d_o, d_o); efficient gather:
    Sigma_oo = Sigma[:, _OBS_IDX][:, :, _OBS_IDX]
    log_comp = np.empty(K)
    for k in range(K):
        log_comp[k] = np.log(pi[k]) + log_gaussian_obs(x_obs, mu_o[k], Sigma_oo[k])
    return logsumexp(log_comp), log_comp  # returns log p(x_o|c), and component-wise logs

# --- Classify a censored image ---
def classify_censored(models, class_priors, x_flat):
    # x_flat: (784,), but only OBS_IDX are used
    x_obs = x_flat[_OBS_IDX]
    best_c = None
    best_score = -np.inf
    per_class_scores = {}
    for c, mdl in models.items():
        log_px_c, _ = class_log_marginal(mdl, x_obs)
        score = np.log(class_priors[c]) + log_px_c
        per_class_scores[c] = score
        if score > best_score:
            best_score, best_c = score, c
    return best_c, per_class_scores

# --- Reconstruct missing pixels for a chosen class (usually the predicted one) ---
def reconstruct_for_class(models_c, x_flat):
    # returns reconstructed full 784-d vector (with missing block filled)
    pi, mu, Sigma = models_c[2], models_c[0], models_c[1]
    K, d = mu.shape
    x_obs = x_flat[_OBS_IDX]

    mu_o = mu[:, _OBS_IDX]           # (K, d_o)
    mu_m = mu[:, _MISS_IDX]          # (K, d_m)
    Sigma_oo = Sigma[:, _OBS_IDX][:, :, _OBS_IDX]     # (K, d_o, d_o)
    Sigma_mo = Sigma[:, _MISS_IDX][:, :, _OBS_IDX]    # (K, d_m, d_o)

    # Compute component posteriors gamma_k ∝ pi_k * N(x_o | mu_o, Sigma_oo)
    log_comp = np.array([
        np.log(pi[k]) + log_gaussian_obs(x_obs, mu_o[k], Sigma_oo[k])
        for k in range(K)
    ])
    log_norm = logsumexp(log_comp)
    gamma = np.exp(log_comp - log_norm)              # (K,)

    # For each component, conditional mean of missing given observed
    d_m = mu_m.shape[1]
    cond_means = np.empty((K, d_m))
    for k in range(K):
        # Solve (Sigma_oo)^{-1} (x_o - mu_o)
        S_oo = Sigma_oo[k] + 1e-6 * np.eye(Sigma_oo[k].shape[0])
        L = np.linalg.cholesky(S_oo)
        r = x_obs - mu_o[k]
        w = np.linalg.solve(L, r)
        alpha = np.linalg.solve(L.T, w)             # alpha = S_oo^{-1} (x_o - mu_o)
        cond_means[k] = mu_m[k] + Sigma_mo[k] @ alpha

    # Mixture mean (MMSE) for missing block
    x_recon = x_flat.copy()
    x_recon[_MISS_IDX] = np.dot(gamma, cond_means)  # weighted sum over k

    # Clip to valid range and return
    return np.clip(x_recon, 0.0, 1.0)

# --- End-to-end: classify + reconstruct for a batch of censored images ---
def classify_and_reconstruct_batch(models, class_priors, X28, y_pred_correct):
    # X28: (N,28,28), assumed ALREADY censored (use censorImages on a copy)
    N = X28.shape[0]
    X_flat = X28.reshape(N, -1)
    y_pred = []
    X_rec = np.empty_like(X_flat)
    correct_predictions = 0
    for i in range(N):
        print(f"Processing image {i+1}/{N}")
        print("-----------------------")
        print(f"Accuracy so far: {correct_predictions}/{i} = {correct_predictions / i if i > 0 else 0:.4f}")
        x = X_flat[i]
        c_hat, _ = classify_censored(models, class_priors, x)
        correct_predictions += (c_hat == y_pred_correct[i])
        y_pred.append(c_hat)
        x_rec = reconstruct_for_class(models[c_hat], x)
        X_rec[i] = x_rec
    X_rec = X_rec.reshape(N, 28, 28)
    X_rec = truncatePixels(X_rec, 0, 1)
    return np.array(y_pred), X_rec

def flattenTensor( X ):
    n = X.shape[0]
    d = np.prod( X.shape[1:] )
    return X.reshape( n, d )

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img.flatten() # Flatten the image data
        return images, labels


    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

if __name__ == "__main__":
    # Load MNIST data
    input_path = 'mnist/'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                 test_images_filepath, test_labels_filepath)
    (XTrain, yTrain), (XTest, yTest) = dataloader.load_data()

    XTest = np.array(XTest)
    XTrain = np.array(XTrain)
    yTest = np.array(yTest)
    yTrain = np.array(yTrain)

    with open(f"gmm_params_python_5.pkl", 'rb') as f:
        gmm_params = pickle.load(f)

    XTest = XTest.reshape( XTest.shape[0], 28, 28 )
    XTest_censored = censorImages( XTest.copy() )

    class_priors = {}
    total_samples = len(yTrain)
    for c in gmm_params.keys():
        class_priors[c] = np.sum( yTrain == c ) / total_samples

    y_pred, XTest_reconstructed = classify_and_reconstruct_batch(
        gmm_params,
        class_priors,
        XTest_censored,
        yTest
    )

    acc = (y_pred == yTest).mean()
    print("Censored-test accuracy:", acc)

    # for c in gmm_params.keys():
    #     print(f"Class {c}:")
    #     print(f"  mu shape: {gmm_params[c][0].shape}")
    #     print(f"  sigma shape: {gmm_params[c][1].shape}")
    #     print(f"  pi shape: {gmm_params[c][2].shape}")

