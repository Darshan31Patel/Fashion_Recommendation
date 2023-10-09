import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import pickle
import os

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Create a new model with the ResNet50 base model followed by GlobalMaxPooling2D
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to extract features for a batch of images
def extract_features_batch(image_paths, model):
    # store features of each image of batch in list
    images = []
    for image_path in image_paths:
        img = load_img(image_path, target_size=(224, 224, 3))
        img_array = img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_input = preprocess_input(expanded_img_array)
        images.append(preprocessed_input)
    images = np.vstack(images)
    # predict features of image from model which gives list of 2048 feature vector
    batch_features = model.predict(images)
    normalized_batch_features = batch_features / np.linalg.norm(batch_features, axis=1, keepdims=True)
    return normalized_batch_features

batch_size = 32 
image_directory = 'images'
filenames = []

for file in os.listdir(image_directory):
    filenames.append(os.path.join(image_directory, file))

total_images = len(filenames)
num_batches = (total_images + batch_size - 1) // batch_size

feature_list = []

# passing images in batch of 32 images at a time
for i in tqdm(range(num_batches)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_images)
    batch_filenames = filenames[start_idx:end_idx]
    batch_features = extract_features_batch(batch_filenames, model)
    feature_list.extend(batch_features)

# Save the features and filenames
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
