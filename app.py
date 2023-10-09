import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.utils import load_img,img_to_array
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

import pandas as pd
df = pd.read_csv('styles1.csv')

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1  # Success
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
        return 0  # Failure

def feature_extraction(img_path,model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    # indices = indices.split(' ')
    return indices
    
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file
    if save_uploaded_file(uploaded_file):  # Implement this function
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Recommendation
        indices = recommend(features, feature_list)  # Implement this function
        print(indices)
        # Display recommended images using columns
        columns = st.columns(5)

        with columns[0]:
            st.image(filenames[indices[0][0]])
        with columns[1]:
            st.image(filenames[indices[0][1]])
        with columns[2]:
            st.image(filenames[indices[0][2]])
        with columns[3]:
            st.image(filenames[indices[0][3]])
        with columns[4]:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occurred in file upload")