import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model('keras_model.h5')

model = load_trained_model()

# Define the size of the input images
img_size = (224, 224)

# Function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, img_size)  # Resize image
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    return img

# Define the Streamlit app
def app():
    st.title('Skin Cancer Classification App')
    st.write("Upload an image to classify whether it has cancer or not.")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Convert file to OpenCV format
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Invalid image file. Please upload a valid image.")
            return

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Process image & Predict on button click
        if st.button("Predict"):
            try:
                img_processed = preprocess_image(img)
                pred = model.predict(img_processed)
                pred_prob = pred[0][0]
                pred_label = 'Cancer' if pred_prob > 0.5 else 'Not Cancer'
                
                # Display Results
                st.success(f'Prediction: **{pred_label}**')
                st.info(f'Probability of Skin Cancer: **{pred_prob * 100:.2f}%**')

            except Exception as e:
                st.error(f"Error making prediction: {e}")

# Run the app
if __name__ == '__main__':
    app()

