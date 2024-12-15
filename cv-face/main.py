import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split


# Load and prepare the dataset
def prepare_data(data_dir):
    faces = []
    labels = []
    label_map = {}  # Map folder names to numeric labels
    current_label = 0

    # Loop through all subdirectories in the dataset directory
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            # Assign a numeric label if not already mapped
            if label_dir not in label_map:
                label_map[label_dir] = current_label
                current_label += 1

            # Process all images in the folder
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                if image_name.endswith(".pgm"):
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
                    if img is not None:
                        img = cv2.resize(img, (92, 112))  # Resize for consistency
                        faces.append(img)
                        labels.append(label_map[label_dir])  # Use numeric label

    return faces, labels


# Train and evaluate a recognizer
def evaluate_recognizer(recognizer, faces_train, labels_train, faces_test, labels_test):
    recognizer.train(faces_train, np.array(labels_train))
    correct = 0
    total = len(faces_test)
    miss_count = 0

    for i in range(total):
        label, confidence = recognizer.predict(faces_test[i])
        if label == labels_test[i]:
            correct += 1
        else:
            miss_count += 1

    accuracy = (correct / total) * 100
    miss_rate = (miss_count / total) * 100

    return accuracy, miss_rate


# Load and prepare data
data_dir = 'ExtendedYaleB'  # Path to the dataset
faces, labels = prepare_data(data_dir)

# Split data into training and testing sets
faces_train, faces_test, labels_train, labels_test = train_test_split(faces, labels, test_size=0.3, random_state=42)

# Initialize the recognizers
# LBPH Recognizer - Accuracy: 99.98% | Miss Rate: 0.02%
recognizers = {
    # "LBPH": cv2.face.LBPHFaceRecognizer_create(),
    # "EigenFace": cv2.face.EigenFaceRecognizer_create(),
    "FisherFace": cv2.face.FisherFaceRecognizer_create(),
}

# Evaluate each recognizer
for name, recognizer in recognizers.items():
    print(name)
    accuracy, miss_rate = evaluate_recognizer(recognizer, faces_train, labels_train, faces_test, labels_test)
    print(f"{name} Recognizer - Accuracy: {accuracy:.2f}% | Miss Rate: {miss_rate:.2f}%")
