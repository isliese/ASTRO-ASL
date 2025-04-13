import os
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import argparse

class ASLPredictor:
    def __init__(self, model_path='model.keras', label_map_path='label_map.json'):
        # Load the model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load label mapping
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        
        # Create reverse mapping for predictions
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        
        # Image size expected by the model
        self.img_size = (224, 224)
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        img = Image.open(image_path)
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    
    def predict_single(self, image_path):
        """Make prediction on a single image"""
        # Preprocess image
        processed_img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get the corresponding letter
        predicted_letter = self.reverse_map[predicted_class]
        
        return {
            'letter': predicted_letter,
            'confidence': float(confidence),
            'all_predictions': {k: float(v) for k, v in zip(self.reverse_map.values(), predictions[0])}
        }
    
    def predict_directory(self, directory_path):
        """Make predictions on all images in a directory"""
        results = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory_path, filename)
                try:
                    prediction = self.predict_single(image_path)
                    results.append({
                        'image': filename,
                        'prediction': prediction
                    })
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        return results

def main():
    parser = argparse.ArgumentParser(description='ASL Image Predictor')
    parser.add_argument('--image', type=str, help='Path to a single image to predict')
    parser.add_argument('--directory', type=str, help='Path to directory containing images to predict')
    parser.add_argument('--model', type=str, default='model.keras', help='Path to the model file')
    parser.add_argument('--labels', type=str, default='label_map.json', help='Path to the label mapping file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ASLPredictor(args.model, args.labels)
    
    if args.image:
        # Single image prediction
        result = predictor.predict_single(args.image)
        print("\nSingle Image Prediction:")
        print(f"Image: {args.image}")
        print(f"Predicted Letter: {result['letter']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll Predictions:")
        for letter, conf in result['all_predictions'].items():
            print(f"{letter}: {conf:.2%}")
    
    elif args.directory:
        # Directory prediction
        results = predictor.predict_directory(args.directory)
        print("\nDirectory Predictions:")
        for result in results:
            print(f"\nImage: {result['image']}")
            print(f"Predicted Letter: {result['prediction']['letter']}")
            print(f"Confidence: {result['prediction']['confidence']:.2%}")
    
    else:
        print("Please provide either --image or --directory argument")
        parser.print_help()

if __name__ == '__main__':
    main()