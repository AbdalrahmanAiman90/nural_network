import numpy as np
import os
import cv2

# Hamming Network implementation


class HammingNetwork:
    def __init__(self, num_neurons, input_size):
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.weights = np.zeros((num_neurons, input_size))

    def train(self, images, labels):
        for image, label in zip(images, labels):
            self.weights[label] += image

    def predict(self, image):
        distances = np.sum(np.abs(self.weights - image), axis=1)
        return np.argmin(distances)

# Load images from a directory


def load_images(directory):
    images = []
    labels = []
    label_mapping = {}

    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Extract the label from the filename
            label = filename.split("_")[0]

            # Check if the label already exists in the label mapping
            if label not in label_mapping:
                label_mapping[label] = len(label_mapping)

            # Load and preprocess the image
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (300, 300))
            image = image.flatten() / 255.0

            # Add the image and label to the lists
            images.append(image)
            labels.append(label_mapping[label])

    return images, labels, label_mapping

# Classify a test image


def classify_image(test_image, hamming_network, label_mapping):
    # Load and preprocess the test image
    test_image = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (300, 300))
    test_image = test_image.flatten() / 255.0

    # Predict the class label
    predicted_label = hamming_network.predict(test_image)

    # Map the predicted label back to the original label
    predicted_label = list(label_mapping.keys())[list(
        label_mapping.values()).index(predicted_label)]

    return predicted_label

# Main function


def main():
    # Load training images and labels
    train_directory = "path_to_train_directory"
    train_images, train_labels, label_mapping = load_images(train_directory)

    # Create and train the Hamming network
    num_neurons = len(label_mapping)
    input_size = len(train_images[0])
    hamming_network = HammingNetwork(num_neurons, input_size)
    hamming_network.train(train_images, train_labels)

    # Classify a test image
    test_image = "path_to_test_image"
    predicted_label = classify_image(
        test_image, hamming_network, label_mapping)
    print("Predicted label:", predicted_label)


if __name__ == '__main__':
    main()
