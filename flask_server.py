from flask import Flask, render_template, request, redirect, url_for

import prediction_script

import urllib
import cv2

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        result_string = ''
        math_exp = None
        result_list = [None, None]
        try:
            # Save Base64 coded image from frontend to file
            base64_image = request.form.get('base64_image')
            input_image = urllib.request.urlopen(base64_image)
            with open('image_input.jpg', 'wb') as f:
                f.write(input_image.file.read())
            # Load image from frontend to opencv image
            input_img = cv2.imread('image_input.jpg', 0)
            # Predict and return the result
            result_list = prediction_script.process_input_and_predict(input_img)
        except:
           result_string = "Error!"
        return render_template('home.html', result_string=result_list[0], math_exp = result_list[1])
    return render_template('home.html')

@app.route('/sendImage', methods=['POST'])
def send_image_to_backend():
    if request.method == 'POST':
        result_string = prediction_script.process_input_and_predict(0)
    return redirect(url_for('home'))

@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(requestResults(name)) + " </xmp> "

if __name__ == '__main__' :
    # Run on LAN
    #app.run(debug=True, host="192.168.0.26")
    # Run on localhost
    app.run(debug=True, host="127.0.0.1")
