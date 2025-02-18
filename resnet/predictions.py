import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def build_resnet50_model(input_shape=(224, 224, 3), num_classes=2):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Custom pipeline wrapper for Keras models
class CustomPipeline:
    def __init__(self, model_builder, **kwargs):
        self.model_builder = model_builder
        self.model = None
        self.kwargs = kwargs

    def fit(self, X, y, **fit_params):
        self.model = self.model_builder(**self.kwargs)
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)

# Function to load and preprocess images (same as before)
def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image
        images.append(img_array)
    return np.vstack(images)


# image_paths = [ "images\imagefour.jpg", "images\imagefive.jpg", "images\imagesix.jpg"]  # non def
image_paths= ["images\imageone.jpg", "images\imagetwo.jpg", "images\imagethree.jpg"] # def


# Load and preprocess the images
X_images = load_and_preprocess_images(image_paths)

# Load the saved pipeline
pipeline = joblib.load('resnet50_pipeline.pkl')

# Make predictions using the loaded pipeline
predictions = pipeline.predict(X_images)

# Output the predictions (class index for each image)
print("Predictions for the images:", predictions)

# You can also print class names if you have a mapping
class_names = ['Non - Defective', 'Defective']  # Example class names
predicted_labels = [class_names[pred] for pred in predictions]

print("Predicted labels for the images:", predicted_labels)
