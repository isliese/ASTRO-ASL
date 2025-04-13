import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import kagglehub
import os
import json


#generate label map for 
def generate_label_map(data_dir, output_file='label_map.json'):
    label_map = {}

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg')):
                    label = img_name[0].lower()
                    if label not in label_map:
                        label_map[label] = len(label_map)

    with open(output_file, 'w') as f:
        json.dump(label_map, f)

    return label_map

#Load the data set
path = kagglehub.dataset_download("ayuraj/american-sign-language-dataset")
print("Path to dataset files:", path)

#check if path exists
if os.path.exists(path):
    generate_label_map(path)

# Load the model
model = tf.keras.models.load_model('model.keras')

# Load label map
with open('label_map.json', 'r') as f:
    label_map = json.load(f)
reverse_label_map = {v: k for k, v in label_map.items()}

# Predict on a new image
img = load_img('test.jpeg', target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

pred = model.predict(img_array)
predicted_label = reverse_label_map[np.argmax(pred)]
print("Predicted label:", predicted_label)
