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
from werkzeug.utils import secure_filename
from model_loader import cargarModelo
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

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
	return '¡Servicio REST activo!'

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
            
            #image_to_predict = image.load_img(filename, target_size=(64, 64))
            #test_image = image.img_to_array(image_to_predict)
            #test_image = np.expand_dims(test_image, axis = 0)
            #test_image = test_image.astype('float32')
            #test_image /= 255
            WIDTH=64
            HEIGHT=64

            test_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            test_image = test_image/255.
            test_image = cv2.resize(test_image, (WIDTH,HEIGHT))
            test_image = test_image.reshape(-1,WIDTH,HEIGHT,1)
            
            
            def compare_images(imageA, imageB, title):
              # compute the mean squared error and structural similarity
              # index for the images
              s = ssim(imageA, imageB)
              # setup the figure
              fig = plt.figure(title)
              plt.suptitle("SSIM: %.2f" % (s))
              # show first image
              ax = fig.add_subplot(1, 2, 1)
              plt.imshow(imageA, cmap = plt.cm.gray)
              plt.axis("off")
              # show the second image
              ax = fig.add_subplot(1, 2, 2)
              plt.imshow(imageB, cmap = plt.cm.gray)
              plt.axis("off")
              # show the images
              plt.show() 

              if(s<0.75):
                print("La imagen no coincide con una radiografia de torax, incerte otra imagen")
              
            # load the images -- the original, the original + contrast,
            # and the original + photoshop
            original = cv2.imread("comparacion/abuelo.jpg")
            contrast = cv2.imread(test_image)
            #contrast = cv2.imread("comparacion/abuelo.jpg")

            # convert the images to grayscale and resize 64x64
            WIDTH=256
            HEIGHT=256

            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            original= cv2.resize(original, (WIDTH,HEIGHT))
            contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
            contrast= cv2.resize(contrast, (WIDTH,HEIGHT))


            # initialize the figure
            fig = plt.figure("Images")
            images = ("Original", original), ("Contrast", contrast)
            # loop over the images
            for (i, (name, image)) in enumerate(images):
              # show the image
              ax = fig.add_subplot(1, 3, i + 1)
              ax.set_title(name)
              plt.imshow(image, cmap = plt.cm.gray)
              plt.axis("off")
            # show the figure
            plt.show()
            # compare the images
            compare_images(original, contrast, "Original vs. Contrast")
            
         
	
              if(s>0.75):
                result = loaded_model.predict(test_image)
            	# print(result)
            	
                # Resultados
                if(result[0][0]>result[0][1] and result[0][0]>result[0][2]): 
                    prediction = 0

                elif(result[0][1]>result[0][0] and result[0][1]>result[0][2]): 
                    prediction = 1    

                else: prediction = 2

                CLASSES = ["NORMAL", "COVID-19", "Viral Pneumonia"]
                ClassPred = CLASSES[prediction]

                # print("Pedicción:", ClassPred)
                # print("Prob:", "{0:.2f}".format(ClassProb))

                #Results as Json
                data["predictions"] = []
                if(prediction == 0):

                    r = {"label": ClassPred, "score": "{0:.2f}".format(result[0,0]*100)}
                elif(prediction == 1):
                    r = {"label": ClassPred, "score": "{0:.2f}".format(result[0,1]*100)}
                else:
                    r = {"label": ClassPred, "score": "{0:.2f}".format(result[0,2]*100)}

                data["predictions"].append(r)

                #Success
                data["success"] = True

        return jsonify(data)

# Run de application
app.run(host='0.0.0.0',port=port, threaded=False)
