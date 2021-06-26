import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
#importing the libraries
import joblib
#from skimage.transform import resize
import time


st.sidebar.title("About")

st.sidebar.info(

"This is an implementation of deep learning in the medical field. The application identifies COVID-19 or CAP presence in CT scans. It was built using a Convolution Neural Network (CNN).")



#loading the cat classifier model
#cat_clf=joblib.load("Cat_Clf_model.pkl")

#Loading Cat moew sound
#audio_file = open('Cat-meow.mp3', 'rb')
#audio_bytes = audio_file.read()



#Load model
to_res = (224, 224)
loaded_model = load_model('my.h5')

# Designing the interface
st.title("COVID-19 CT scans Classification App")
# For newline
st.write('\n')



st.sidebar.title("Upload Image")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

if uploaded_file is not None:
    
	test_image = image.load_img(uploaded_file, target_size = (224, 224)) 
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)

# For newline
st.sidebar.write('\n')
    
if st.sidebar.button("Click Here to Classify"):
    
    if uploaded_file is None:
        
        st.sidebar.write("Please upload an Image to Classify")
    
    else:
        
        with st.spinner('Classifying ...'):
            
        	prediction = loaded_model.predict(test_image)
        	d = {0:'Normal',1:'Covid',2:'Community Acquired Pneumonia'}
        	predicted_class = d[int(np.argmax(prediction,axis=1))]
	        confidence = prediction[0][int(np.argmax(prediction,axis=1))]
	        time.sleep(2)
    		#st.success('Done!')

            

        st.sidebar.header("Algorithm Predicts: ")
        
        #Formatted probability value to 3 decimal places
        
        # Classify cat being present in the picture if prediction > 0.5
        
            
        st.sidebar.write(predicted_class)
            
        st.sidebar.write('**Probability: **',confidence,'%')
    
    
