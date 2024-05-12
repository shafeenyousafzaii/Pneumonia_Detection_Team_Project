import streamlit as st
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load your Keras model
model = load_model('my_model3.h5')

# Function to make prediction
def predict_pneumonia(file):
    # Load and preprocess the image
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
            
    # Make prediction
    prediction = model.predict(img_array)
    threshold = 0.55
    pred_class = (prediction > threshold).astype(int)
    if pred_class[0][0] == 1:
        diagnosis = "Pneumonia"
        probability = prediction[0][0]  # Assuming prediction is the probability of pneumonia
    else:
        diagnosis = "Pneumonia Not Detected"
        probability =  prediction[0][0]
        
    return diagnosis, probability

# Streamlit app
def main():
    st.title('Pneumonia Detection')
    st.write('Upload a chest X-ray image to detect pneumonia.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        diagnosis, probability = predict_pneumonia(file_path)
        if diagnosis == "Pneumonia":
            st.warning(f"Diagnosis: {diagnosis}, Probability: {probability}")
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        else:
            st.success(f"Diagnosis: {diagnosis}, Probability: {probability}")
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
           
if __name__ == "__main__":
    main()
