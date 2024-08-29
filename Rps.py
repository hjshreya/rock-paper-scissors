import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import streamlit as st

model=tf.keras.models.load_model('/Users/shrutishreya/Downloads/my_model.keras')
class_labels = ['Rock', 'Paper', 'Scissors']

def load_and_preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((300, 300))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array

def predict_gesture(img):
    processed_img = load_and_preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)
    return class_labels[predicted_class[0]]

st.title("Rock-Paper-Scissors Predictor")
st.write("""Upload greyscale images of Rock, Paper, or Scissors to see 
which one wins.""")

uploaded_file1 = st.file_uploader("Choose the first image...", 
type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Choose the second image...", 
type=["jpg", "jpeg", "png"])

if uploaded_file1 and uploaded_file2:
    img1 = Image.open(uploaded_file1)
    img2 = Image.open(uploaded_file2)
    
    st.image([img1, img2], caption=["First Image", "Second Image"], 
    use_column_width=True)
    
    gesture1 = predict_gesture(img1)
    gesture2 = predict_gesture(img2)
    
    st.write(f"First Image is: {gesture1}")
    st.write(f"Second Image is: {gesture2}")
    
    if gesture1 == gesture2:
        st.write("It's a tie!")
    elif (gesture1 == 'Rock' and gesture2 == 'Scissors') or \
         (gesture1 == 'Scissors' and gesture2 == 'Paper') or \
         (gesture1 == 'Paper' and gesture2 == 'Rock'):
        st.write("First Image wins!")
    else:
        st.write("Second Image wins!")










