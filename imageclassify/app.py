from flask import jsonify, request, Flask, render_template
import base64
import io
import os
from PIL import Image
import keras.preprocessing.image as kimg
from keras.models import load_model
import tensorflow as tf


app=Flask(__name__)

def get_model():
    global model
    model = load_model('vgg16model.h5')
    print('Model loaded')

def preprocess_image(img):
    img=kimg.load_img(img,target_size=(224,224))
    img=kimg.img_to_array(img)
    img=img.reshape(1,224,224,3)
    img=img.astype('float32')
    img=img-[123.68,116.779,103.939]
    return img

get_model()

