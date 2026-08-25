import pickle
import numpy as np
import struct
from os.path import join
from array import array
from numpy.linalg import inv
from cs771 import plotData as pd
import matplotlib.pyplot as plt

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

def predict_with_missing_features(X_test, gmm_params, class_priors, missing_mask):
    """
    Performs inference using the trained GMM classifier with missing features.
    missing_mask is a boolean array of length D where missing_mask[i] is True indicates a missing feature.
    """
    print("Starting prediction with missing features...")
    N_test = X_test.shape[0]
    C = len(class_priors)
    K = gmm_params[0][2].shape[0] # Get K from class 0's pi vector
    # D = X_test.shape[1]

    # Store log P(x | y=c) + log P(y=c)
    log_posteriors = np.zeros((N_test, C))

    log_priors = np.log(class_priors + 1e-300)

    for c in range(C):
        mu_c, sigma_c, pi_c = gmm_params[c]

        # Store log P(x | z=k, y=c) + log P(z=k | y=c)
        log_probs_k = np.zeros((N_test, K))
        log_pi_c = np.log(pi_c + 1e-300)

        for k in range(K):
            mu_ck = mu_c[k]
            sigma_ck = sigma_c[k]

            # Adjust mu and sigma for missing features
            mu_ck_missing = mu_ck[~missing_mask]
            sigma_ck_missing = sigma_ck[~missing_mask][:, ~missing_mask]
            X_test_missing = X_test[:, ~missing_mask]
            log_pdf_k = log_gaussian_pdf(X_test_missing, mu_ck_missing, sigma_ck_missing)
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

def censorImages( X ):
    print(X.shape)
    # Wipe out 21% of the pixels from the central part of the image
    X[:, 8:20, 8:20] = 0
    return X

# Make sure that reconstructed image does not have any negative-valued pixels
def truncatePixels( X, low = 0, high = 1 ):
    X[X < low] = low
    X[X > high] = high
    return X

# def reconstruct_missing_features(X_test, y_pred, gmm_params, missing_mask):
#     """
#     Reconstructs missing features in X_test based on predicted classes and GMM parameters.
#     missing_mask is a boolean array of length D where missing_mask[i] is True indicates a missing feature.
#     """
#     print("Starting reconstruction of missing features...")
#     XRecon = np.zeros( X_test.shape )
#     # Pixels observed are used as is in the reconstruction
#     XRecon[:, ~missing_mask] = X_test[:, ~missing_mask]
#     for i in range( X_test.shape[0] ):
#         ( mu, Sigma, c, p ) = gmm_params[y_pred[i]]
#         recon = mu[~missing_mask] + Sigma[~missing_mask,:][:,missing_mask].dot( lin.pinv( Sigma[missing_mask,:][:,missing_mask] ).dot( (XRecon[i,missing_mask] - mu[missing_mask]) ) )
#         XRecon[i, ~missing_mask] = recon
#     return truncatePixels( XRecon )

def predict(X_test, gmm_params, class_priors):
    """
    Performs inference using the trained GMM classifier.
    """
    print("Starting prediction...")
    N_test = X_test.shape[0]
    C = len(class_priors)
    K = gmm_params[0][2].shape[0] # Get K from class 0's pi vector

    # Store log P(x | y=c) + log P(y=c)
    log_posteriors = np.zeros((N_test, C))

    log_priors = np.log(class_priors + 1e-300)

    for c in range(C):
        mu_c, sigma_c, pi_c = gmm_params[c]

        # Store log P(x | z=k, y=c) + log P(z=k | y=c)
        log_probs_k = np.zeros((N_test, K))
        log_pi_c = np.log(pi_c + 1e-300)

        for k in range(K):
            log_pdf_k = log_gaussian_pdf(X_test, mu_c[k], sigma_c[k])
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

def e_step(data_c, pi_c, mu_c, Sigma_c):
    """Performs the E-Step using the log-sum-exp trick."""
    N_c, D = data_c.shape
    print("data_c shape:", data_c.shape)
    K = pi_c.shape[0]

    print("pi_c shape:", pi_c.shape)

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

def reconstruct_missing_features(X_test, y_pred, gmm_params, missing_mask, K):
    """
    Reconstructs missing features in X_test based on predicted classes and GMM parameters.
    missing_mask is a boolean array of length D where missing_mask[i] is True indicates a missing feature.
    """
    print("Starting reconstruction of missing features...")
    XRecon = np.zeros( X_test.shape )
    # Pixels observed are used as is in the reconstruction
    XRecon[:, ~missing_mask] = X_test[:, ~missing_mask]

    for i in range( X_test.shape[0] ):
        # Params for data point
        c = y_pred[i]
        mu_c, Sigma_c, pi_c = gmm_params[c]

        res_values = np.zeros( (K,) )

        for k in range(K):
            mu_ck = mu_c[k]
            sigma_ck = Sigma_c[k]

            # Adjust mu and sigma for missing features
            mu_ck_missing = mu_ck[~missing_mask]
            sigma_ck_missing = sigma_ck[~missing_mask][:, ~missing_mask]
            # convert X_test_missing from shape (D,) to (1, D_missing)
            X_test_missing = X_test[i, ~missing_mask][np.newaxis, :]

            responsibilities, _ = e_step(X_test_missing, pi_c, mu_ck_missing, sigma_ck_missing)
            res_values[k] = responsibilities
        
        component = np.argmax( res_values )
        mu = mu_c[component]
        Sigma = Sigma_c[component]

        recon = mu[~missing_mask] + Sigma[~missing_mask, :][:, missing_mask].dot(lin.pinv(Sigma[missing_mask, :][:, missing_mask]).dot((XRecon[i, missing_mask] - mu[missing_mask])))
        XRecon[i, ~missing_mask] = recon

    return truncatePixels( XRecon )


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
    XTestFlat = flattenTensor( XTest / 255 )
    XTrainFlat = flattenTensor( XTrain / 255 )
    yTest = np.array(yTest)
    yTrain = np.array(yTrain)


    # Class priors
    numClasses = 10
    class_priors = np.array([np.sum(yTrain == c) for c in range(numClasses)]) / yTrain.shape[0]
        
    # K_values = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    K_values = [10]
    train_accuracy = []
    test_accuracy = []

    for K in K_values:

        # Load the parameters from the pickle file
        with open(f"gmm_params_python_{K}.pkl", 'rb') as f:
            gmm_params = pickle.load(f)

        # Perform prediction
        # yPred_train = predict(XTrainFlat, gmm_params, class_priors)
        # yPred_test = predict(XTestFlat, gmm_params, class_priors)
        # Calculate accuracy
        # train_accuracy.append(np.mean(yPred_train == yTrain))
        # test_accuracy = np.mean(yPred_test == yTest)

        # print(f"Test Accuracy with K={K}: {test_accuracy*100:.2f}%")

        XTestFlat_censored = flattenTensor( censorImages( XTest.reshape(XTest.shape[0], 28, 28) ) / 255 )
        missing_mask = np.zeros( XTestFlat.shape[1], dtype=bool )
        missing_mask[ (8*28 + 8) : (20*28 + 20) ] = True


        # missing_mask1 = flattenTensor( censorImages( np.ones( imShape )[np.newaxis, :] ) )[0] == 1
        # print(missing_mask1)

        # Perform prediction on censored data
        yPred_test_censored = predict_with_missing_features(XTestFlat_censored, gmm_params, class_priors, missing_mask)
        # Calculate accuracy
        test_accuracy = np.mean(yPred_test_censored == yTest)
        print(f"Test Accuracy with censored images and K={K}: {test_accuracy*100:.2f}%")

        wrong_preds = 10
        right_preds = 10

        wrong_index = []
        right_index = []
        for i in range( XTestFlat.shape[0] ):
            if yPred_test_censored[i] != yTest[i] and wrong_preds > 0:
                wrong_index.append(i)
                wrong_preds -= 1
            elif yPred_test_censored[i] == yTest[i] and right_preds > 0:
                right_index.append(i)
                right_preds -= 1
            if wrong_preds == 0 and right_preds == 0:
                break

        XTest_wrong = np.array(XTestFlat_censored[wrong_index])
        XTest_right = np.array(XTestFlat_censored[right_index])

        recon_XTest_right = reconstruct_missing_features(XTest_wrong, yPred_test_censored[wrong_index], gmm_params, missing_mask, K)
        recon_XTest_wrong = reconstruct_missing_features(XTest_right, yPred_test_censored[right_index], gmm_params, missing_mask, K)

    
        # numRows = 4
        # numCols = 5
        # imShape = (28, 28)
        # fig14, axs14 = pd.getFigList( numRows, numCols, sizey = 3.2 )
        # labels = ["True Label: %s\nCensored Image" % yTest[i] for i in range( numRows * numCols )]
        # pd.showImagesNoAxes( axs14, XTestFlat_censored[:numRows*numCols], numRows, numCols, resize = True, imShape = imShape, labelList = labels )


    # with open("gmm_classification_results_python.pkl", 'wb') as f:
    #     pickle.dump((K_values, train_accuracy, test_accuracy), f)
