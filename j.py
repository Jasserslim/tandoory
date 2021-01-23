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

def VGGmodel () :
    from keras.applications.vgg16 import VGG16
    from tensorflow.keras.applications.vgg16 import VGG16
    from keras.models import Model
    from keras.layers import MaxPooling2D
    from keras.layers import GlobalAveragePooling2D, Dense,BatchNormalization
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
    from keras.models import Sequential
    from keras import optimizers
    vgg_model = VGG16(weights='imagenet',
                                include_top=False,
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

    vgg_model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.RMSprop(lr=2e-5),
                        metrics=['accuracy'])

    vgg_model.load_weights("./weights.vgg_model.best.inc.male.hdf5")

    return(vgg_model)

app = Flask(__name__)
api = Api(app)

class predict(Resource) :
    def get(self):
        keys =  parse_arg_from_requests('key')
        keys='./'+keys
        print(keys)
        image = plt.imread(keys,0)
        #Resizing and reshaping to keep the ratio.
        my_image_resized = resize(image, (224,224,3))
        model = VGGmodel()
        print('start')
        probabilities = model.predict(np.array( [my_image_resized,] ))
        print('end')
        index = probabilities
        index = index.tolist()
        return Response(json.dumps(index),status=200,mimetype='application/json')

api.add_resource(predict, "/get", "/get/")

if __name__ == '__main__':

    app.run(debug=False)
