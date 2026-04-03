import numpy as np
from features import image_to_vector


def train_perceptron(train_images, train_labels, epochs=5, lr=1.0):
    """
    Train a multiclass perceptron using the One-vs-All strategy.

    Args:
        train_images: list of 28x28 images
        train_labels: list of integer labels in {0, ..., 9}
        epochs: number of passes over the training set
        lr: learning rate

    Returns:
        weights: numpy array of shape (10, 784)
    """
    num_classes = 10
    input_dim = 28 * 28

    # TODO 1: Initialize the weight matrix for all 10 perceptrons.
    weights = np.zeros((num_classes, input_dim))
    
    for epoch in range(epochs):
        num_updates = 0

        for image, true_label in zip(train_images, train_labels):
            # TODO 2: Convert the image into a normalized 784-D vector.
            x = image_to_vector(image)
            # TODO 3: For each class c, compute the binary target (+1 or -1),
            # compute the perceptron score, and update the weights if misclassified.
            # Use the update rules from class:
            #   False Positive: w <- w - x
            #   False Negative: w <- w + x
            for c in range(num_classes):
                if c == true_label:
                    target = 1
                else:
                    target = -1
                score = np.dot(weights[c], x)

            if target * score <= 0:
                if target == 1:
                    weights[c] += lr * x
                else:
                    weights[c] -= lr * x
                num_updates += 1
        print(f"Epoch {epoch + 1}/{epochs} completed - updates: {num_updates}")

    return weights
