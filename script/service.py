##Import Flask
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS

#Import Keras
from keras.preprocessing import image

#Import python files
import numpy as np

import cv2
import requests
import json
import os
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename
from model_loader import cargarModelo

UPLOAD_FOLDER = '../images/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

port = int(os.getenv('PORT', 5000))
print ("Port recognized: ", port)

#Initialize the application service
app = Flask(__name__)
CORS(app)
global loaded_model, graph
loaded_model, graph = cargarModelo()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Define a route
@app.route('/')
def main_page():
	return '┬íServicio REST activo!'

@app.route('/model/covid19/', methods=['GET','POST'])
def default():
    data = {"success": False}
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            print('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #loading image
            filename = UPLOAD_FOLDER + '/' + filename
            print("\nfilename:",filename)
            

            original = cv2.imread("NLeo1.jpeg")
            contrast = cv2.imread(filename)
		
            WIDTH=64
            HEIGHT=64
		
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
            original= cv2.resize(original, (WIDTH,HEIGHT))
            contrast= cv2.resize(contrast, (WIDTH,HEIGHT))

            s=ssim(original, contrast)
            print("SSIM: %.2f" % (s))
		
            test_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            test_image = test_image/255.
            test_image1 = cv2.resize(test_image, (WIDTH,HEIGHT))
            test_image = test_image1.reshape(-1,WIDTH,HEIGHT,1)

            result = loaded_model.predict(test_image)
            # print(result)
            	
            # Resultados
            if(result[0][0]>result[0][1] and result[0][0]>result[0][2]): 
                prediction = 0

            if(result[0][1]>result[0][0] and result[0][1]>result[0][2]): 
                prediction = 1    
		
            if(result[0][2]>result[0][0] and result[0][2]>result[0][1]):
                prediction = 2

            if(s<0.1): 
                prediction = 3
                print("Esta imagen no coincide con una raiograf├şa de torax, por favor elija otra")
                
            CLASSES = ["NORMAL", "COVID-19", "Viral Pneumonia", "La imagen no es una radiograf├şa de Torax"]
            ClassPred = CLASSES[prediction]

            # print("Pedicci├│n:", ClassPred)
            # print("Prob:", "{0:.2f}".format(ClassProb))

            #Results as Json
            data["predictions"] = []
            if(prediction == 0):     
                r = {"label": ClassPred, "score": "{0:.2f}".format(result[0,0]*100)}
            if(prediction == 1):
                r = {"label": ClassPred, "score": "{0:.2f}".format(result[0,1]*100)}
            elif(prediction == 2):
                r = {"label": ClassPred, "score": "{0:.2f}".format(result[0,2]*100)}
            else:
                r = {"label": ClassPred}
                
            data["predictions"].append(r)

            #Success
            data["success"] = True

    return jsonify(data)

# Run de application
app.run(host='0.0.0.0',port=port, threaded=False)
