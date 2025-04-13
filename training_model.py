# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
import keras
import kagglehub
from sklearn.model_selection import train_test_split

# Download dataset from Kaggle
path = kagglehub.dataset_download("ayuraj/american-sign-language-dataset")
print("Path to dataset files:", path)

# Update to match actual folder inside the dataset
data_dir = os.path.join(path, 'asl')  # Adjust if folder name differs

print("Files in dataset path:", os.listdir(path))  # Debugging line

# Define the percentage of data to sample
sample_percentage = 0.1  # 0.5% of the data

# Function to sample a percentage of data
def sample_data(data_dir, sample_percentage):
    sampled_data_dir = path
    if not os.path.exists(sampled_data_dir):
        os.makedirs(sampled_data_dir)

    for folder_name in os.listdir(data_dir):
        if folder_name == 'asl' or not folder_name.isalpha() or len(folder_name) != 1:
            continue
        folder_path = os.path.join(data_dir, folder_name)
        sampled_folder_path = os.path.join(sampled_data_dir, folder_name)

        if os.path.isdir(folder_path):
            if not os.path.exists(sampled_folder_path):
                os.makedirs(sampled_folder_path)

            images = os.listdir(folder_path)
            sample_size = int(len(images) * sample_percentage)
            sampled_images = np.random.choice(images, sample_size, replace=False)

            for img_name in sampled_images:
                img_path = os.path.join(folder_path, img_name)
                sampled_img_path = os.path.join(sampled_folder_path, img_name)
                os.rename(img_path, sampled_img_path)

    return sampled_data_dir

# Sample the data
data_dir = sample_data(data_dir, sample_percentage)

# Check if the data directory exists
if not os.path.exists(data_dir):
    print(f"Data directory '{data_dir}' does not exist.")
    exit() 

def add_gaussian_noise(image, mean=0, stddev=0.1):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, stddev, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)  # Ensure the pixel values are in [0, 255]
    return noisy.astype(np.uint8)

def load_data(data_dir, add_noise=False):
    images = []
    labels = []
    label_map = {}  # To map gestures to integer labels

    for folder_name in os.listdir(data_dir):
        if folder_name == 'asl' or not folder_name.isalpha() or len(folder_name) != 1: # Only consider folders with single-letter names
            continue

        folder_path = os.path.join(data_dir, folder_name)
        
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                # Only consider .jpg or .jpeg files
                if img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    img_path = os.path.join(folder_path, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (224, 224))

                    # Extract the label (first letter of the filename)
                    label = folder_name.lower()


                    # Convert label to integer if not already mapped
                    if label not in label_map:
                        label_map[label] = len(label_map)

                    # Optionally add noise
                    if add_noise:
                        img = add_gaussian_noise(img)

                    images.append(img)
                    labels.append(label_map[label])

    images = np.array(images)
    labels = np.array(labels)

    return images, labels, label_map

# Load data with noise added
images, labels, label_map = load_data(data_dir, add_noise=True)

# Normalize pixel values to be between 0 and 1
images = images / 255.0

# One-hot encode the labels
labels = keras.utils.to_categorical(labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Number of classes: {len(label_map)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Define the model
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(label_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")

print(f"Test loss: {loss:.3f}")

# Save the model
model.save('model.keras')