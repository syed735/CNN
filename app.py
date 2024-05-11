import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('models/monumentclassifier.h5')

def classify_image(img):
    img = img.resize((256, 256))  # Resize image for model input
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = img_array / 255.0  # Normalize image
    
    # Classify image
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    # Return result
    if predicted_class >0.5:
        result = "European"
    else:
        result = "Indian"
   
    return result
def main():
    st.title("Image Classifier")
    st.write("Upload an image and I'll classify it as either European or Indian Monument.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        result = classify_image(image)
        st.write(f"Predicted Monument: {result}")

if __name__ == "__main__":
    main()
