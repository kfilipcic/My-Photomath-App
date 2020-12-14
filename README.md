# My-Photomath-App
My try in making an app which reads a math expression as an input and gives out the result.

## How it works?
More details about how the actual code works can be found in Jupyter Notebook files of this repository. Comments through code also explain certain functions and their significance.

## Requirements
1. CUDA 10.1
2. Python 3.8, along with libraries listed in `requirements.txt`.

## Using the Web Application

This project includes a Web Application with a simple UI where a user can their mobile phone to take a photo and send the image to the backend. The image is processed and the result is return from backend and shown on the site.

### Running the server

Run the following command in terminal `python flask_server.py`, assuming you are currently in the directory of the cloned repository. The command will run the server on localhost, if you wish to change the server address then change the `app.run()` arguments in `flask_server.py`.
You can then open the index page of the server which is the only site of this Web app. You simply just press the CALCULATE button to calculate the expression from camera, wait for a bit and the result should appear on the screen, along with the found math expression.

<p float="left">
  <img src="https://raw.githubusercontent.com/kfilipcic/My-Photomath-App/main/web_app1.png" width="300" height="500"/>
  <img src="https://raw.githubusercontent.com/kfilipcic/My-Photomath-App/main/web_app2.png" width="300" height="500"/>
</p>

## Training the model

If you wish to train the model, you need to have folders containing training data which are named exactly as the class names (which can be found in `training_script.py`). Then, you can run the command `python training_script.py model_name.h5`, where "model_name.h5" is the desired name of the output trained model.

## Making predictions and calculating expressions without the Web app

You can make predictions using the command `python prediction_script.py input_image.jpg` where input_image.jpg is the name of the desired input file image. The model used for prediction can be changed by changing the `h5_filename` string accordingly in the `prediction_script.py` file.
Assuming the terminal is currently in the cloned repository folder, you can run the following command: `python prediction_script.py ./image_examples/example3_with_all_chars.jpg`. The image_examples folder contains a few input images which work with the model located in the repository.
