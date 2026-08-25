import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    # with open('gmm_classification_results_python.pkl', 'rb') as f:
    #     K_values, train_accuracies, test_accuracies = pickle.load(f)

    K_values = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    train_accuracies = [0.7731, 0.7849, 0.7789, 0.769, 0.7453, 0.7244, 0.7419, 0.7167, 0.729, 0.7089]
    # Censored-test accuracy: 0.7731
    # Censored-test accuracy: 0.7849
    # Censored-test accuracy: 0.7789
    # Censored-test accuracy: 0.769
    # Censored-test accuracy: 0.7453
    # Censored-test accuracy: 0.7244
    # Censored-test accuracy: 0.7419
    # Censored-test accuracy: 0.7167
    # Censored-test accuracy: 0.729
    # Censored-test accuracy: 0.7089

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, [x*100 for x in train_accuracies], label='Accuracy', marker='o')
    # plt.plot(K_values, [x*100 for x in test_accuracies], label='Testing Accuracy', marker='s')
    plt.title('GMM Classifier Inferencing over Censored Test Set')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('Accuracy (%)')
    plt.xticks(K_values)
    plt.ylim(60, 80)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()