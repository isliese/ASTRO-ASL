import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# Load the model
model = tf.keras.models.load_model('asl_model.h5')

# Load label map
with open('label_map.json', 'r') as f:
    label_map = json.load(f)
reverse_label_map = {v: k for k, v in label_map.items()}

# Predict on a new image
img = load_img('path/to/image.jpeg', target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

pred = model.predict(img_array)
predicted_label = reverse_label_map[np.argmax(pred)]
print("Predicted label:", predicted_label)
