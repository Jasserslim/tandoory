from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dense,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
import flask
from flask import Flask,request,Response
from flask_restful import Api, Resource, reqparse
import pickle
import base64
import numpy as np
import cv2
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from skimage.transform import resize

def parse_arg_from_requests(arg, **kwargs):
    parse = reqparse.RequestParser()
    parse.add_argument(arg, **kwargs)
    args = parse.parse_args()
    return args[arg]

def parse_params(param):
    return json.dumps([x.strip() for x in param.split(",")])

def model () :
    vgg_model = VGG16(include_top=False,
                                input_shape=(224, 224, 3))

    x4 = vgg_model.output  
    x4 = GlobalAveragePooling2D()(x4)  
    x4 = BatchNormalization()(x4)  
    x4 = Dropout(0.5)(x4)  
    x4 = Dense(512, activation ='relu')(x4) 
    x4 = BatchNormalization()(x4) 
    x4 = Dropout(0.5)(x4) 
     
    x4 = Dense(2, activation ='softmax')(x4)  
    vgg_model = Model(vgg_model.input, x4)


    vgg_model.load_weights('j.hdf5')

    return(vgg_model)

model = model()
app = Flask(__name__)
api = Api(app)

class predict(Resource) :
    def get(self):
        keys =  parse_arg_from_requests('key')
        keys='./'+keys
        print(keys)
        image = plt.imread(keys)
        #Resizing and reshaping to keep the ratio.
        resized = resize(image, (1,224,224,3))
        vect = np.asarray(resized, dtype="uint8")
        print('start')
        my_prediction = model.predict(vect)
        print('end')
        index = my_prediction[0]
        index = index.tolist()
        return Response(json.dumps(index),status=200,mimetype='application/json')

api.add_resource(predict, "/get", "/get/")

if __name__ == '__main__':

    app.run(debug=False)