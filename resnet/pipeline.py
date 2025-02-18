import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# Function to build the ResNet50 model
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

# Load image data (example code for image loading and preprocessing)
def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image
        images.append(img_array)
    return np.vstack(images)

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

# Prepare image data (replace with your own image paths)
image_paths= ["images\imagefour.jpg", "images\imagefive.jpg", "images\imagesix.jpg"]
X_images = load_and_preprocess_images(image_paths)
Y_labels = np.array([0, 0, 0])  

# One-hot encode the labels
Y_labels = to_categorical(Y_labels, num_classes=2)
# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_images, Y_labels, test_size=0.25, random_state=42)

# Initialize and train the pipeline
pipeline = CustomPipeline(model_builder=build_resnet50_model, input_shape=(224, 224, 3), num_classes=2)
pipeline.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(X_test, Y_test))

# Save the pipeline model
pipeline.save('resnet50_pipeline2.pkl')

print("ResNet50 Pipeline saved successfully!")


predictions = pipeline.predict(X_images)

# Output the predictions (class index for each image)
print("Predictions for the images:", predictions)

# You can also print class names if you have a mapping
class_names = ['Non - Defective', 'Defective']  # Example class names
predicted_labels = [class_names[pred] for pred in predictions]

print("Predicted labels for the images:", predicted_labels)