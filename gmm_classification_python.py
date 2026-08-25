import numpy as np
import time
import struct
from array import array
from os.path import join
from numpy.linalg import inv
import pickle

# --- 1. K-Means for GMM Initialization ---

def run_kmeans(data, K, max_iters=100, random_seed=42):
    """
    Implements a simple K-Means clustering algorithm using only NumPy.

    Args:
        data (np.ndarray): N x D_pca array of data points.
        K (int): The number of clusters.

    Returns:
        tuple: (centroids, labels)
            centroids (np.ndarray): K x D_pca array of final cluster centers.
            labels (np.ndarray): N-dimensional array of cluster assignments (0 to K-1).
    """
    N, D = data.shape
    np.random.seed(random_seed)

    initial_indices = np.random.choice(N, K, replace=False)
    centroids = data[initial_indices]
    labels = np.zeros(N, dtype=int)

    for _ in range(max_iters):
        labels_old = np.copy(labels)

        # E-Step (Assignment)
        distances = np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2, axis=2)
        labels = np.argmin(distances, axis=1)

        # M-Step (Update)
        centroids_new = np.zeros((K, D))
        for k in range(K):
            cluster_data = data[labels == k]
            if len(cluster_data) > 0:
                centroids_new[k] = np.mean(cluster_data, axis=0)
            else:
                # Re-initialize empty clusters
                centroids_new[k] = data[np.random.choice(N)]

        centroids = centroids_new

        if np.array_equal(labels, labels_old):
            break

    return centroids, labels

# --- 2. GMM Core Functions (Numerically Stable) ---

def initialize_gmm_params(data_c, K, reg_covar=1e-6):
    """Initializes GMM parameters using K-Means."""
    N_c, D = data_c.shape

    mu_c, labels = run_kmeans(data_c, K)

    pi_c = np.zeros(K)
    Sigma_c = np.zeros((K, D, D))

    for k in range(K):
        cluster_data = data_c[labels == k]
        N_k = len(cluster_data)

        if N_k == 0:
            pi_c[k] = 1.0 / K # Assign uniform prob if empty
            mu_c[k] = np.mean(data_c, axis=0) # Use overall mean
            Sigma_c[k] = np.cov(data_c, rowvar=False) + reg_covar * np.eye(D)
        else:
            pi_c[k] = N_k / N_c

            if N_k > 1:
                cov_k = np.cov(cluster_data, rowvar=False)
            else:
                cov_k = np.cov(data_c, rowvar=False) # Fallback

            Sigma_c[k] = cov_k + reg_covar * np.eye(D)

    pi_c = pi_c / np.sum(pi_c)
    return pi_c, mu_c, Sigma_c

def log_gaussian_pdf(X, mu, Sigma):
    """Computes the log-PDF of a multivariate Gaussian."""
    N, D = X.shape

    try:
        sign, log_det_Sigma = np.linalg.slogdet(Sigma)
        if sign != 1:
             return np.full(N, -1e100)
        Sigma_inv = inv(Sigma)
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix in log_gaussian_pdf")
        # Add more regularization if this happens
        Sigma = Sigma + 1e-4 * np.eye(D)
        sign, log_det_Sigma = np.linalg.slogdet(Sigma)
        Sigma_inv = inv(Sigma)
        if sign != 1:
            return np.full(N, -1e100)

    log_norm_const = -0.5 * D * np.log(2 * np.pi) - 0.5 * log_det_Sigma
    X_minus_mu = X - mu
    term1 = np.dot(X_minus_mu, Sigma_inv)
    quadratic_form = np.sum(term1 * X_minus_mu, axis=1)
    log_pdf_values = log_norm_const - 0.5 * quadratic_form

    return log_pdf_values

def e_step(data_c, pi_c, mu_c, Sigma_c):
    """Performs the E-Step using the log-sum-exp trick."""
    N_c, D = data_c.shape
    K = pi_c.shape[0]

    log_weighted_pdfs = np.zeros((N_c, K))
    log_pi_c = np.log(pi_c + 1e-300)

    for k in range(K):
        log_pdf_values = log_gaussian_pdf(data_c, mu_c[k], Sigma_c[k])
        log_weighted_pdfs[:, k] = log_pi_c[k] + log_pdf_values

    # Log-Sum-Exp Trick
    log_max_per_data_point = np.max(log_weighted_pdfs, axis=1, keepdims=True)
    exp_log_probs_stable = np.exp(log_weighted_pdfs - log_max_per_data_point)
    sum_exp_stable = np.sum(exp_log_probs_stable, axis=1, keepdims=True)

    responsibilities = exp_log_probs_stable / sum_exp_stable
    responsibilities[np.isnan(responsibilities)] = 1.0 / K # Handle NaNs

    # Calculate log-likelihood for convergence check
    log_likelihood = np.sum(log_max_per_data_point + np.log(sum_exp_stable))

    return responsibilities, log_likelihood

def m_step(data_c, responsibilities, reg_covar=1e-6):
    """Performs the M-Step."""
    N_c, D = data_c.shape
    K = responsibilities.shape[1]

    # 1. Calculate N_k (effective number of points per cluster)
    N_k = np.sum(responsibilities, axis=0) # K-dim vector

    # 2. Update pi_c
    pi_c = N_k / N_c

    # 3. Update mu_c
    # (K x N_c) @ (N_c x D) -> (K x D)
    mu_c = (responsibilities.T @ data_c) / (N_k[:, np.newaxis] + 1e-300)

    # 4. Update Sigma_c
    Sigma_c = np.zeros((K, D, D))
    for k in range(K):
        X_minus_mu_k = data_c - mu_c[k] # N_c x D
        resp_k = responsibilities[:, k:k+1] # N_c x 1

        # (D x N_c) * (N_c x 1).T -> (D x N_c)
        # ( (X-mu).T * resp.T ) @ (X-mu) -> D x D
        Sigma_k = (X_minus_mu_k.T * resp_k.T) @ X_minus_mu_k
        Sigma_c[k] = Sigma_k / (N_k[k] + 1e-300)

        # Add regularization
        Sigma_c[k] += reg_covar * np.eye(D)

    return pi_c, mu_c, Sigma_c

# --- 3. Main Training Function ---

def train_gmm_classifier(X_train, y_train, K, max_em_iters=20, reg_covar=1e-6):
    """
    Trains the full GMM generative classifier.
    """
    C = len(np.unique(y_train)) # Number of classes (10)

    # 2. Calculate class priors P(y=c)
    class_priors = np.zeros(C)
    for c in range(C):
        class_priors[c] = np.mean(y_train == c)

    # 3. Train one GMM per class
    gmm_params = {}

    for c in range(C):
        print(f"\nTraining GMM for class {c} (K={K})...")
        data_c = X_train[y_train == c]

        # Initialize
        pi_c, mu_c, Sigma_c = initialize_gmm_params(data_c, K[c], reg_covar)

        last_log_likelihood = -np.inf

        for i in range(max_em_iters):
            # E-Step
            responsibilities, log_likelihood = e_step(data_c, pi_c, mu_c, Sigma_c)

            # M-Step
            pi_c, mu_c, Sigma_c = m_step(data_c, responsibilities, reg_covar)

            # Check for convergence
            if i > 0 and np.abs(log_likelihood - last_log_likelihood) < 1e-4:
                print(f"Class {c} converged after {i+1} iterations.")
                break
            last_log_likelihood = log_likelihood

            if (i+1) % 5 == 0:
                print(f"  Class {c}, Iter {i+1}/{max_em_iters}, Log-Likelihood: {log_likelihood:.6f}")

        gmm_params[c] = (pi_c, mu_c, Sigma_c)

    print("\n--- Training Complete ---")

    # Return all learned parameters
    return gmm_params, class_priors

# --- 5. Inference (Prediction) ---

def predict(X_test, gmm_params, class_priors, K):
    """
    Performs inference using the trained GMM classifier.
    """
    print("Starting prediction...")
    N_test = X_test.shape[0]
    C = len(class_priors)
    K = gmm_params[0][0].shape[0] # Get K from class 0's pi vector

    # Store log P(x | y=c) + log P(y=c)
    log_posteriors = np.zeros((N_test, C))

    log_priors = np.log(class_priors + 1e-300)

    for c in range(C):
        mu_c, Sigma_c, pi_c = gmm_params[c]

        # Store log P(x | z=k, y=c) + log P(z=k | y=c)
        log_probs_k = np.zeros((N_test, K[c]))
        log_pi_c = np.log(pi_c + 1e-300)

        for k in range(K[c]):
            log_pdf_k = log_gaussian_pdf(X_test, mu_c[k], Sigma_c[k])
            log_probs_k[:, k] = log_pi_c[k] + log_pdf_k

        # Calculate log P(x | y=c) = log-sum-exp( log_probs_k )
        log_max = np.max(log_probs_k, axis=1, keepdims=True)
        log_sum_exp = log_max + np.log(np.sum(np.exp(log_probs_k - log_max), axis=1, keepdims=True))

        # Bayes' Rule (in log-space)
        log_posteriors[:, c] = log_sum_exp.flatten() + log_priors[c]

    # Prediction is the class with the highest log-posterior
    predictions = np.argmax(log_posteriors, axis=1)

    print("Prediction complete.")
    return predictions

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

def flattenTensor( X ):
    n = X.shape[0]
    d = np.prod( X.shape[1:] )
    return X.reshape( n, d )

# --- 6. Main execution ---
if __name__ == "__main__":

    # --- Configuration ---
    # These are the values you will vary for your assignment
    # K_COMPONENTS = [1, 2, 5, 10, 15 ,20, 25, 30, 35 ,40]      # Number of GMM components (K)
    MAX_EM_ITERS = 65     # Max iterations for EM
    REG_COVAR = 1e-5     # Regularization for covariance

    # --- Data Loading ---
    # Assumes 'mnist_train.csv' and 'mnist_test.csv' are in the same folder

    input_path = 'mnist/'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    ((X_train, y_train), (X_test, y_test)) = mnist_dataloader.load_data()

    # Normalize pixel values
    X_train = np.array(X_train).astype(np.float64) / 255.0
    X_test = np.array(X_test).astype(np.float64) / 255.0

    X_train = flattenTensor( X_train )
    X_test = flattenTensor( X_test )
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # for K_COMPONENTS in [1, 2, 5, 10, 15 ,20, 25, 30, 35 ,40]:
    # for K_COMPONENTS in [5]:
    K_COMPONENTS = [5, 20, 1, 10, 1, 2, 1, 2, 5, 5]
    print("\nStarting GMM Classifier Training ---")
    print(f"K (Components) = {K_COMPONENTS}")

    # start_time = time.time()

    # # --- Train ---
    # gmm_params, class_priors = train_gmm_classifier(
    #     X_train, y_train,
    #     K=K_COMPONENTS,
    #     max_em_iters=MAX_EM_ITERS,
    #     reg_covar=REG_COVAR
    # )

    # end_time = time.time()
    # print(f"\nTotal training time: {end_time - start_time:.2f} seconds")

    # # --- Save Model ---
    # with open(f"gmm_params_python_special.pkl", 'wb') as f:
    #     pickle.dump(( {c: (gmm_params[c][1], gmm_params[c][2], gmm_params[c][0]) for c in gmm_params} ), f)

    # --- Predict and Evaluate ---

    with open(f"gmm_params_python_special.pkl", 'rb') as f:
        gmm_params = pickle.load(f)

    class_priors = np.zeros(10)
    for c in range(10):
        class_priors[c] = np.mean(y_train == c)

    # Test accuracy
    predictions_test = predict(X_test, gmm_params, class_priors, K_COMPONENTS)
    accuracy_test = np.mean(predictions_test == y_test)

    # Train accuracy
    predictions_train = predict(X_train, gmm_params, class_priors, K_COMPONENTS)
    accuracy_train = np.mean(predictions_train == y_train)

    print("\n--- Results ---")
    print(f"K = {K_COMPONENTS}")
    print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
    print(f"Test Accuracy:     {accuracy_test * 100:.2f}%")
