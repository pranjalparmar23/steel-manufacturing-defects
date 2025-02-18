import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
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





# Load and preprocess image
def load_and_preprocess_image(uploaded_image, target_size=(224, 224)):
    img = image.load_img(uploaded_image, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

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


# Load the saved pipeline
pipeline = joblib.load('resnet50_pipeline.pkl')
# Function to predict whether the image is defective or non-defective
def predict_image_class(img_array):
    predictions = pipeline.predict(img_array)
    class_names = ['Non - Defective', 'Defective']
    predicted_label = class_names[predictions[0]]
    return predicted_label

# Streamlit App
st.title("Defective or Non-Defective Image Classifier")
st.write("Upload an image to classify whether it is defective or non-defective.")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    img_array = load_and_preprocess_image(uploaded_file)

    # Predict the class of the uploaded image
    if st.button("Predict"):
        result = predict_image_class(img_array)
        st.write(f"The image is: {result}")
