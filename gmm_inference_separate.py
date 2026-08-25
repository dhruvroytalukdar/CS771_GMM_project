import pickle
import numpy as np
import struct
from os.path import join
import os
from array import array
from numpy.linalg import inv
from cs771 import plotData as pd
import matplotlib.pyplot as plt

# -----------------------
# Provided helpers
# -----------------------
def censorImages(X):
    X[:, 8:20, 8:20] = 0
    return X

def truncatePixels(X, low=0, high=1):
    X[X < low] = low
    X[X > high] = high
    return X

# -----------------------
# Missing window indices
# -----------------------
def missing_mask_28x28():
    mask = np.zeros((28,28), dtype=bool)
    mask[8:20, 8:20] = True
    return mask

_MISS_MASK_2D = missing_mask_28x28()
_MISS_IDX = np.flatnonzero(_MISS_MASK_2D.ravel())      # 144
_OBS_IDX  = np.flatnonzero(~_MISS_MASK_2D.ravel())     # 640
_DOBS = len(_OBS_IDX)
_DALL = 28*28

def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

# =======================================================
# FULL-FEATURE PREDICTION (train/test uncensored)
# =======================================================
def prepare_full_models(models, jitter=1e-6):
    """
    Precompute Cholesky and log-dets for full-d covariances.
    """
    prep = {}
    I = np.eye(_DALL)
    for c, mdl in models.items():
        pi, mu, Sigma = mdl[2], mdl[0], mdl[1]
        K = mu.shape[0]
        L = np.empty_like(Sigma)
        logdet = np.empty(K)
        for k in range(K):
            S = Sigma[k] + jitter * I
            try:
                Lk = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                Lk = np.linalg.cholesky(S + 1e-3 * I)
            L[k] = Lk
            logdet[k] = 2.0 * np.sum(np.log(np.diag(Lk)))
        prep[c] = {
            'logpi': np.log(pi + 1e-300),
            'mu': mdl[0],           # (K, 784)
            'L': L,                    # (K, 784, 784)
            'logdet': logdet
        }
    return prep

def predict_full_batch(prep_models, class_priors, X28):
    """
    Vectorized argmax_c [log p(c) + log p(x|c)] for uncensored images.
    """
    N = X28.shape[0]
    X = X28.reshape(N, -1)  # (N,784)
    classes = list(prep_models.keys())
    per_class_scores = {}

    const_term = -0.5 * _DALL * np.log(2.0 * np.pi)
    for c in classes:
        pm = prep_models[c]
        mu   = pm['mu']      # (K,d)
        L    = pm['L']       # (K,d,d)
        logd = pm['logdet']  # (K,)
        logpi= pm['logpi']   # (K,)
        K    = mu.shape[0]
        comp_logs = np.empty((N, K))
        for k in range(K):
            r = X - mu[k]                          # (N,d)
            wT = np.linalg.solve(L[k], r.T)        # (d,N)
            quad = np.sum(wT * wT, axis=0)         # (N,)
            logN = const_term - 0.5 * (logd[k] + quad)
            comp_logs[:, k] = logpi[k] + logN
        log_px_c = logsumexp(comp_logs, axis=1)    # (N,)
        per_class_scores[c] = np.log(class_priors[c] + 1e-300) + log_px_c

    scores_mat = np.column_stack([per_class_scores[c] for c in classes])  # (N,C)
    idx = np.argmax(scores_mat, axis=1)
    y_pred = np.array([classes[j] for j in idx])
    return y_pred, per_class_scores

# =======================================================
# CENSORED-FEATURE PREDICTION (test with missing)
# =======================================================
def prepare_obs_models(models, jitter=1e-6):
    """
    Precompute Cholesky and log-dets for observed subspace.
    """
    prep = {}
    I = np.eye(_DOBS)
    for c, mdl in models.items():
        pi, mu, Sigma = mdl[2], mdl[0], mdl[1]
        K = mu.shape[0]
        mu_o = mu[:, _OBS_IDX]                             # (K, d_o)
        Sigma_oo = Sigma[:, _OBS_IDX][:, :, _OBS_IDX]      # (K, d_o, d_o)
        L_oo = np.empty_like(Sigma_oo)
        logdet = np.empty(K)
        for k in range(K):
            S = Sigma_oo[k] + jitter * I
            try:
                L = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(S + 1e-3 * I)
            L_oo[k] = L
            logdet[k] = 2.0 * np.sum(np.log(np.diag(L)))
        prep[c] = {
            'logpi': np.log(pi + 1e-300),
            'mu_o':  mu_o,
            'L_oo':  L_oo,
            'logdet': logdet
        }
    return prep

def predict_censored_batch(prep_models, class_priors, X28_censored):
    """
    Vectorized argmax_c [log p(c) + log p(x_o|c)] for censored images.
    """
    N = X28_censored.shape[0]
    Xo = X28_censored.reshape(N, -1)[:, _OBS_IDX]  # (N, d_o)
    classes = list(prep_models.keys())
    per_class_scores = {}

    const_term = -0.5 * _DOBS * np.log(2.0 * np.pi)
    for c in classes:
        pm = prep_models[c]
        mu_o   = pm['mu_o']          # (K, d_o)
        L_oo   = pm['L_oo']          # (K, d_o, d_o)
        logdet = pm['logdet']        # (K,)
        logpi  = pm['logpi']         # (K,)
        K = mu_o.shape[0]
        comp_logs = np.empty((N, K))
        for k in range(K):
            r = Xo - mu_o[k]                     # (N, d_o)
            wT = np.linalg.solve(L_oo[k], r.T)   # (d_o, N)
            quad = np.sum(wT * wT, axis=0)       # (N,)
            logN = const_term - 0.5 * (logdet[k] + quad)
            comp_logs[:, k] = logpi[k] + logN
        log_px_c = logsumexp(comp_logs, axis=1)
        per_class_scores[c] = np.log(class_priors[c] + 1e-300) + log_px_c

    scores_mat = np.column_stack([per_class_scores[c] for c in classes])  # (N,C)
    idx = np.argmax(scores_mat, axis=1)
    y_pred = np.array([classes[j] for j in idx])
    return y_pred, per_class_scores

# =======================================================
# PER-IMAGE RECONSTRUCTION (from censored)
# =======================================================
def reconstruct_for_class(models_c, x_flat):
    """
    Mixture-of-conditionals reconstruction for a SINGLE censored image.
    Use predicted class for models_c, fill the missing block.
    """
    pi, mu, Sigma = models_c[2], models_c[0], models_c[1]
    K, d = mu.shape
    x_obs = x_flat[_OBS_IDX]

    mu_o = mu[:, _OBS_IDX]           # (K, d_o)
    mu_m = mu[:, _MISS_IDX]          # (K, d_m)
    Sigma_oo = Sigma[:, _OBS_IDX][:, :, _OBS_IDX]     # (K, d_o, d_o)
    Sigma_mo = Sigma[:, _MISS_IDX][:, :, _OBS_IDX]    # (K, d_m, d_o)

    # Component posteriors
    log_comp = np.empty(K)
    I = np.eye(_DOBS)
    const = -0.5 * _DOBS * np.log(2.0*np.pi)
    for k in range(K):
        S_oo = Sigma_oo[k] + 1e-6 * I
        L = np.linalg.cholesky(S_oo)
        r = x_obs - mu_o[k]
        w = np.linalg.solve(L, r)
        quad = np.dot(w, w)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        logN = const - 0.5 * (logdet + quad)
        log_comp[k] = np.log(pi[k] + 1e-300) + logN
    gamma = np.exp(log_comp - logsumexp(log_comp))      # (K,)

    # Conditional means and mixture
    d_m = mu_m.shape[1]
    cond_means = np.empty((K, d_m))
    for k in range(K):
        S_oo = Sigma_oo[k] + 1e-6 * I
        L = np.linalg.cholesky(S_oo)
        r = x_obs - mu_o[k]
        w = np.linalg.solve(L, r)
        alpha = np.linalg.solve(L.T, w)                 # S_oo^{-1}(x_o - mu_o)
        cond_means[k] = mu_m[k] + Sigma_mo[k] @ alpha

    x_recon = x_flat.copy()
    x_recon[_MISS_IDX] = gamma @ cond_means
    return np.clip(x_recon, 0.0, 1.0)

# =======================================================
# DRIVER: accuracies + 2x10 grids for each K
# =======================================================
def evaluate_and_visualize_all_K(
    models_by_K,
    class_priors_by_K,
    K_list,
    X_train, y_train,
    X_test, y_test,
    out_dir="recon_grids"
):
    os.makedirs(out_dir, exist_ok=True)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Precompute censored test set once
    X_test_cens = censorImages(X_test.copy())

    # Hold accuracies
    records = []
    priors = class_priors_by_K
    for K in K_list:
        models = models_by_K[K]

        # --- Train accuracy (full)
        prep_full = prepare_full_models(models)
        y_pred_train, _ = predict_full_batch(prep_full, priors, X_train)
        acc_train = (y_pred_train == y_train).mean()

        # --- Test accuracy (full)
        y_pred_test, _ = predict_full_batch(prep_full, priors, X_test)
        acc_test = (y_pred_test == y_test).mean()

        # --- Censored-test accuracy
        prep_obs = prepare_obs_models(models)
        y_pred_test_cens, _ = predict_censored_batch(prep_obs, priors, X_test_cens)
        acc_test_cens = (y_pred_test_cens == y_test).mean()

        print(f"K={K}: Train Acc={acc_train:.4f}, Test Acc={acc_test:.4f}, Test Cens Acc={acc_test_cens:.4f}")

        records.append({
            "K": K,
            "acc_train": float(acc_train),
            "acc_test": float(acc_test),
            "acc_test_censored": float(acc_test_cens)
        })

        # ------------- Build 2x10 grid for censored predictions -------------
        # Find indices for correct and incorrect
        correct_idx = np.flatnonzero(y_pred_test_cens == y_test)
        wrong_idx   = np.flatnonzero(y_pred_test_cens != y_test)

        # Pick up to 6 each
        n_show = 6
        correct_pick = correct_idx[:n_show]
        wrong_pick   = wrong_idx[:n_show]

        # If not enough, pad with repeats (optional)
        if correct_pick.size < n_show and correct_pick.size > 0:
            correct_pick = np.pad(correct_pick, (0, n_show - correct_pick.size), mode='wrap')
        if wrong_pick.size < n_show and wrong_pick.size > 0:
            wrong_pick = np.pad(wrong_pick, (0, n_show - wrong_pick.size), mode='wrap')

        # If one of them is empty, skip image (no wrong or no correct)
        if correct_pick.size == 0 or wrong_pick.size == 0:
            print(f"[K={K}] Skipping grid image (insufficient correct/incorrect).")
            continue

        # Reconstruct the chosen images (per-image reconstruction)
        recon_correct = []
        for i in correct_pick:
            x = X_test_cens[i].reshape(-1)
            c_hat = y_pred_test_cens[i]
            x_rec = reconstruct_for_class(models[c_hat], x).reshape(28,28)
            x_rec = 1 - x_rec  # Invert colors for better visibility
            print(x_rec)
            x_rec *= 255.0

            recon_correct.append(x_rec)

        recon_wrong = []
        for i in wrong_pick:
            x = X_test_cens[i].reshape(-1)
            c_hat = y_pred_test_cens[i]
            x_rec = reconstruct_for_class(models[c_hat], x).reshape(28,28)
            x_rec = 1 - x_rec  # Invert colors for better visibility
            x_rec *= 255.0
            recon_wrong.append(x_rec)

        # Plot 2x10 grid
        fig, axes = plt.subplots(2, n_show, figsize=(10, 5))
        fig.suptitle(f"K={K}  |  Top: Correct Recon  |  Bottom: Incorrect Recon", y=0.98)

        for j in range(n_show):
            axes[0, j].imshow(recon_correct[j], cmap='gray', vmin=0, vmax=1)
            axes[0, j].axis('off')
            axes[1, j].imshow(recon_wrong[j], cmap='gray', vmin=0, vmax=1)
            axes[1, j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_path = os.path.join(out_dir, f"recon_grid_K{K}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[K={K}] Saved grid to {save_path}")

    # Print summary table
    print("\n=== Accuracy Summary ===")
    for r in records:
        print(f"K={r['K']:>2} | train={r['acc_train']:.4f} | test={r['acc_test']:.4f} | test(cens)={r['acc_test_censored']:.4f}")

    return records  # list of dicts

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
    K_list = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]

    # Make sure you have these populated:
    # models_by_K: dict[K][class] -> {'pi','mu','Sigma'}
    # class_priors_by_K: dict[K][class] -> prior
    # X_train, y_train, X_test, y_test

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


    XTest = XTest.reshape( XTest.shape[0], 28, 28 )
    XTrain = XTrain.reshape( XTrain.shape[0], 28, 28 )

    gmm_params_by_K = {}
    for k in K_list:
        # Load models and priors for each K
        with open(f'gmm_params/gmm_params_python_{k}.pkl', 'rb') as f:
            gmm_params = pickle.load(f)
        gmm_params_by_K[k] = gmm_params

    class_priors_by_K = {}
    for c in range(10):
        class_priors_by_K[c] = yTrain.tolist().count(c) / len(yTrain)

    acc_table = evaluate_and_visualize_all_K(
        gmm_params_by_K,
        class_priors_by_K,
        K_list,
        XTrain, yTrain,
        XTest, yTest,
        out_dir="recon_grids"
    )

    with open("gmm_evaluation_records.pkl", 'wb') as f:
        pickle.dump(acc_table, f)