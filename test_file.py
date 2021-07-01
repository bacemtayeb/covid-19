%%writefile app.py

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()



st.sidebar.title("About")

st.sidebar.info(

"This is an implementation of deep learning in the medical field. The application identifies COVID-19 or CAP presence in CT scans. It was built using a Convolution Neural Network (CNN).")
# Designing the interface
st.title("COVID-19 CT scans Classification App")
# For newline
st.write('\n')
#st.sidebar.title("Upload Image")
st.sidebar.image("https://frapscentre.org/wp-content/uploads/2020/03/COVID19.jpg", use_column_width=True)
st.set_option('deprecation.showfileUploaderEncoding', False)

to_res = (224, 224)

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "my.h5"
    to_res = (224, 224)
    model = load_model(classifier_model)
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = cv2.cvtColor(test_image,cv2.COLOR_GRAY2RGB)
    #test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    d = {0:'Normal',1:'Covid',2:'Community Acquired Pneumonia'}
    predicted_class = d[int(np.argmax(prediction,axis=1))]
    confidence = prediction[0][int(np.argmax(prediction,axis=1))]
    result = f"{predicted_class} with a { (100 * np.round(confidence,4)) } % confidence." 
    return result
 

if __name__ == "__main__":
    to_res = (224, 224)
    main()

