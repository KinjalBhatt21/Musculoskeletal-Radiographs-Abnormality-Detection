
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize, ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

# Loading all the image paths from the CSV file
def data_load(train_path, valid_path, body_part):
  train_images = pd.read_csv(str(train_path), names=['Image']) 
  valid_images = pd.read_csv(str(valid_path), names=['Image'])
  # Fetching path of Elbow X-ray
  train = [''.join('/content/drive/MyDrive/' + i) for i in train_images.values if str(body_part) in i[0]] 
  valid = [''.join('/content/drive/MyDrive/' + i) for i in valid_images.values if str(body_part) in i[0]]

  return train,valid

# Function to create labels
def labels(names): 
  label = []
  for i in names:
    if ('positive' in i):
      label.append('1')
    elif('negative' in i):
        label.append('0')
  return label

def load_images(location, labels):
  images = []
  for image_path in location:
    image  = keras.preprocessing.image.load_img(image_path, color_mode= 'rgb', target_size = (224, 224))
    input_arr = np.array(image)
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    images.append(input_arr)
  image_array = np.array(images)
  image_label = keras.utils.to_categorical(labels)
  return images, image_array, image_label