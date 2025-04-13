# import necessary libraries
import os
import cv2
import numpy as np
import keras
import tf
from sklearn.model_selection import train_test_split


data_dir = './asl'

def load_data(data_dir):
    images = []
    labels = []
    label_map = {}  # To map gestures to integer labels

    for folder_name in os.listdir(data_dir):
        if folder_name.startswith('.'):  # Skip hidden directories
            continue
        folder_path = os.path.join(data_dir, folder_name)
        
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                # Only consider .jpg or .jpeg files
                if img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    img_path = os.path.join(folder_path, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)  # Resize to a consistent shape

                    # Extract the label (first letter of the filename)
                    # Validate the assumption that the first letter represents the gesture
                    label = img_name[0].lower() if img_name[0].isalpha() else 'unknown'
                    if label == 'unknown':
                        print(f"Warning: Assigning 'unknown' label to file '{img_name}' as it does not start with a valid letter.")
                    
                    # Convert label to integer if not already mapped
                    if label not in label_map:
                        label_map[label] = len(label_map)
                    
                    # Convert label to integer if not already mapped
                    if label not in label_map:
                        label_map[label] = len(label_map)

                    images.append(img)
                    labels.append(label_map[label])

    images = np.array(images)
    labels = np.array(labels)

    return images, labels, label_map

# Load data
images, labels, label_map = load_data(data_dir)

# Normalize pixel values to be between 0 and 1
images = images / 255.0

# One-hot encode the labels
labels = tf.keras.utils.to_categoricalto_cate(labels)

# Sample data
images = images[:1000]  # Use only the first 1000 samples for testing
labels = labels[:1000]  # Use only the first 1000 samples for testing

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Number of classes: {len(label_map)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Define the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(label_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")

print(f"Test loss: {loss:.3f}")

# Save the model
model.save('asl_model.h5')
