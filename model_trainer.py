import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(data_dir):
    images = []
    labels = []
    label_map = {}
    
    # Create label mapping
    for idx, folder in enumerate(sorted(os.listdir(data_dir))):
        if os.path.isdir(os.path.join(data_dir, folder)):
            label_map[folder] = idx
    
    # Load images and labels
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                if image_file.endswith('.jpeg'):
                    image_path = os.path.join(folder_path, image_file)
                    # Load and preprocess image
                    img = tf.keras.preprocessing.image.load_img(
                        image_path, 
                        target_size=(224, 224)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0  # Normalize
                    
                    images.append(img_array)
                    labels.append(label_map[folder])
    
    return np.array(images), np.array(labels), label_map

def create_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    # Load data
    data_dir = './asl'
    images, labels, label_map = load_data(data_dir)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Create and compile model
    model = create_model(len(label_map))
    
    # Train the model
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=32),
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Save the model
    model.save('asl_model.h5')
    
    # Save label mapping
    with open('label_map.json', 'w') as f:
        import json
        json.dump(label_map, f)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()

if __name__ == '__main__':
    train_model()
