import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
import tempfile

# Paths to your directories and files
MODEL_PATH = "pest_identification_model.keras"
CSV_PATH = "Plant disease.csv"
# Paths to your directories and files
image_dir = os.path.expanduser('C:/Users/USER/Downloads/New plant/Images')
MODEL_PATH = "pest_identification_model.keras"
CSV_PATH = "Plant disease.csv"

# Prepare the ImageDataGenerator for training and validation
image_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    rotation_range=30,
    horizontal_flip=True
)

train_data = image_gen.flow_from_directory(
    image_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)


# Load the model
@st.cache_resource  # Cache the model to avoid reloading it multiple times
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load the CSV file
@st.cache_data  # Cache the CSV file to reduce reloading overhead
def load_csv():
    return pd.read_csv(CSV_PATH)

pest_df = load_csv()

# Function to predict pest and fetch details
def predict_pest(uploaded_file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        # Load and preprocess the image
        img = load_img(temp_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the pest
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        class_labels = list(train_data.class_indices.keys())
        pest_name = class_labels[predicted_class]

        # Fetch pest details from the CSV
        pest_info = pest_df[pest_df['crop_disease'] == pest_name]
        return pest_name, pest_info
    finally:
        # Clean up temporary file
        os.remove(temp_path)

# Streamlit App
st.title("Crop Disease Identification and Treatment Recommendation System")
st.write("Upload an image of the crop to identify the disease and provide recommendations.")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with st.spinner('Predicting...'):
        try:
            pest_name, pest_info = predict_pest(uploaded_file)
            st.success(f"Identified: {pest_name}")

            if not pest_info.empty:
                st.write(f"**Disadvantages:** {pest_info['Disadvantages'].values[0]}")
                st.write(f"**First Aid (Short Term):** {pest_info['First_Aid_Short_Term'].values[0]}")
                st.write(f"**First Aid (Long Term):** {pest_info['First_Aid_Long_Term'].values[0]}")
                st.write(f"**Pesticides/Treatments:** {pest_info['Pesticides/ Treatments'].values[0]}")
                st.write(f"**Maintenance Recommendations:** {pest_info['Maintenance_Recommendations'].values[0]}")
            else:
                st.error("Disease details for the identified crop are not available.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.write("Powered by TensorFlow and Streamlit")

