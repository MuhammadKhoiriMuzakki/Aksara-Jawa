# Aksara Jawa Backend

# Import Depedencies
# For Backend System
from json import load
from flask import Flask, render_template, request, url_for
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import cv2

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21
#Initializing the Default Graph (prevent errors)
graph = tensorflow.compat.v1.get_default_graph()

app = Flask(__name__)

dic = {0 : 'Ba', 1 : 'Ca' , 2 : 'Da', 3 : 'Dha', 4 : 'Ga', 5 : 'Ha', 6 : 'Ja', 7 : 'Ka', 8 : 'La', 9 : 'Ma', 10 : 'Na', 11 : 'Nga', 12 : 'Nya', 13 : 'Pa', 14 : 'Ra', 15 : 'Sa', 16 : 'Ta', 17 : 'Tha', 18 : 'Wa', 19 : 'Ya'}

model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(112,112), color_mode='grayscale')
    # i = image.img_to_array(i)/255.0
    i = np.asarray(i)
    i = np.expand_dims(i, axis=2)
    i = np.expand_dims(i, axis=0)
    i = i.reshape(-1, 112,112,1)
    p = model.predict_classes(i)
    return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("klasifikasi.html")

@app.route("/arsip_aksara_jawa")
def arsip_aksara_jawa():
    return render_template("arsip_aksara_jawa.html")

@app.route("/dokumentasi")
def dokumentasi():
    return render_template("dokumentasi.html")

@app.route("/canvas")
def canvas():
    return render_template("canvas.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename	
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("klasifikasi.html", prediction = p, img_path = img_path)

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    # global graph
    # with graph.as_default():
    if request.method == 'POST':
        final_pred = None
        #Preprocess the image : set the image to 28x28 shape
        #Access the image
        draw = request.form['url']
        #Removing the useless part of the url.
        draw = draw[init_Base64:]
        #Decoding
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        #Resizing and reshaping to keep the ratio.
        resized = cv2.resize(image, (112,112), interpolation = cv2.INTER_AREA)
        vect = np.asarray(resized, dtype="uint8")
        # vect = np.expand_dims(vect, axis=2)
        vect = np.expand_dims(vect, axis=0)
        vect = vect.reshape(-1, 112,112,1).astype('float32')
        
        #Launch prediction
        my_prediction = model.predict_classes(vect)
        #Getting the index of the maximum prediction
        index = np.argmax(my_prediction[0])
        #Associating the index and its value within the dictionnary
        final_pred = dic[index]

    return render_template('canvas.html', prediction =final_pred)

if __name__ =='__main__':
    #app.debug = True
    app.run(debug = True)