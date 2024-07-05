import streamlit as st
import os
import numpy as np
import cv2
import math
import skimage
from skimage.feature import graycomatrix, graycoprops
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from IPython.display import display, Image

# Load your trained model for image classification
model = load_model("https://drive.google.com/file/d/1RsQJtu6xW_CRSzPSwj_6UlbuIZ-Vkkp7/view?usp=drive_link")

# Define your class names here
class_names = ['AP', 'ATP', 'IR20', 'KO50']


# Function to predict a single image
def predict_single_image(image_path):
    img = load_img(image_path, target_size=(100, 100))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]


# Function to calculate morphological features
def calculate_morphological_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(max_contour)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    orientation = ellipse[2]
    eccentricity = math.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))
    area = cv2.contourArea(max_contour)
    perimeter = cv2.arcLength(max_contour, True)
    roundness = (4 * math.pi * area) / perimeter
    aspect_ratio = major_axis / minor_axis

    if orientation > 90:
        orientation -= 180

    return major_axis, minor_axis, orientation, eccentricity, area, roundness, aspect_ratio


# Function to calculate texture features
def calculate_texture_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    distances = [1]
    angles = [0]
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').ravel()[0]
    correlation = graycoprops(glcm, 'correlation').ravel()[0]
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()[0]
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()[0]

    return contrast, correlation, dissimilarity, homogeneity


# Streamlit UI
def main():
    st.title("Rice Seed Classification and Germination Prediction System")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        st.write("")
        st.write("Classification successfully completed.")

        # Save the uploaded file temporarily
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Make prediction
        predicted_class = predict_single_image(temp_image_path)

        # Calculate morphological features
        major_axis, minor_axis, orientation, eccentricity, area, roundness, aspect_ratio = calculate_morphological_features(
            temp_image_path)

        # Calculate texture features
        contrast, correlation, dissimilarity, homogeneity = calculate_texture_features(temp_image_path)

        # Remove temporary file
        os.remove(temp_image_path)

        # Display results
        st.subheader("Prediction:")
        if predicted_class=='AP':
            st.write("Predicted Class: ",predicted_class," (English: Andhra Ponni) (Tamil: ஆந்திரா பொன்னி)")
        elif predicted_class=='ATP':
            st.write("Predicted Class: ", predicted_class, " (English: Atchaya Ponni) (Tamil: அட்சயா பொன்னி)")
        elif predicted_class=='IR20':
            st.write("Predicted Class: ", predicted_class, " (English: International Rice-20) (Tamil: சர்வதேச அரிசி-20)")
        else:
            st.write("Predicted Class: ", predicted_class, " (English: Coimbatore Rice-50) (Tamil: கோயம்புத்தூர் அரிசி-50)")

        if contrast >= 5 or roundness < 4000:
            st.write("Possibility of Germination: :x:")
        else:
            st.write("Possibility of Germination: :white_check_mark:")



if __name__ == "__main__":
    main()
