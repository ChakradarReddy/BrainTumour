from flask import Flask,render_template, request, redirect, send_from_directory
import os
import cv2
import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import cv2
import imutils
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            
    return new_image


def detect(img):
    model=load_model('cnn-parameters-improvement-23-0.91.model')
    image = cv2.imread(img)
    image = preprocess(image)
    image = cv2.resize(image, dsize=(240,240), interpolation=cv2.INTER_CUBIC)
    image = image / 255.
    X=[]
    X.append(image)
    X = np.array(X)
    y=model.predict(X)
    return round(y[0][0],2)
app = Flask(__name__)

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def get_images():
    #code fore deleting the given input images    
    dir = './static/' 
    for f in os.listdir(dir):
       os.remove(os.path.join(dir, f))
    given_img = request.files['given_img']
    image_path="./static/"+given_img.filename
    given_img.save(image_path)
    os.rename(os.path.join(os.getcwd(), image_path), "./static/"+'given_img.png')



    # image_1 = cv2.imread('./static/given_img.png')
    
    y=detect('./static/given_img.png') 

    if(y>0.9):
        ans = 'You have brain tumour with probability  '+ str(y)
    if(y<=0.9):
        ans = 'You dont have brain tumour with probability of '+ str(1-y)
    image_names = os.listdir('./static')
    return render_template('post_index.html',image_names=image_names, ans = ans)


if __name__ =="__main__":
     app.run(port=3000,debug=False)
