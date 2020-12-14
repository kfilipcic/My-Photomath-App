import cv2
import numpy as np
import os
import pandas as pd
from sys import argv
import shutil
import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import re
import numpy as np

def process_input_and_predict(img):
    # Reading input image
    img_original = img

    h5_filename = 'h5_ccn_math_exp_detection_14121221.h5'
    img = cv2.bitwise_not(img)

    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    kernel = np.ones((5,5),np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)


    # Removing noise / preprocessing
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Using Morphological transformations for removing noise in the input image

    # Kernel for erosion and dilation
    kernel = np.ones((5,5),np.uint8)

    # cv2.morphologyEx function sometimes works perfectly,
    # but sometimes ruins the entire input
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)[1]
    # Dilation mostly helps
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    cv2.imwrite('input_image_processed.jpg', img)

    def sort_contours(contours):
        # Sort contours from left to right
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
            key=lambda b:b[1][0]))
        return (cnts, boundingBoxes)

    # Find contours
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours, boundingBoxes = sort_contours(contours)

    if os.path.isdir('test_image_unprepared'):
        shutil.rmtree('./test_image_unprepared')
    os.mkdir('test_image_unprepared')

    img_cnt= 0
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        # Take into consideration only contours bigger than 10x4,
        # to ignore noises
        if w>10 and h>4:
            # Save individual images

            if img_cnt >= 0:
                # Save unaltered cropped images
                cv2.imwrite('cropped_characters/' + str(img_cnt) + ".jpg", img_original[y:y+h,x:x+w])
                # Save modified cropped images for
                # further processing
                img_cropped = img[y:y+h,x:x+w]

                # If there are more than one contours found on one image
                # leave out only the biggest one and remove all smaller ones
                contours_cropped, hierarchy_cropped = cv2.findContours(img_cropped,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                # Index of current biggest contour
                biggest_contour_idx = 0

                # Find contour with biggest surface
                for cnt_crp_idx in range(len(contours_cropped)):
                    xc,yc,wc,hc = cv2.boundingRect(contours_cropped[cnt_crp_idx])
                    xcb,ycb,wcb,hcb = cv2.boundingRect(contours_cropped[biggest_contour_idx])
                    if wc*hc > wcb*hcb:
                        biggest_contour_idx = cnt_crp_idx

                # Fill all smaller contour with solid black color
                # In other words: remove all smaller contours
                for cnt_crp_idx in range(len(contours_cropped)):
                    if cnt_crp_idx != biggest_contour_idx:
                        xc,yc,wc,hc = cv2.boundingRect(contours_cropped[cnt_crp_idx])
                        img_shape = img_cropped[yc:yc+hc,xc:xc+wc].shape
                        # Fill area with solid black color
                        img_cropped[yc:yc+hc,xc:xc+wc] = np.zeros(img_shape)

                # Save only the biggest contour to image
                if contours_cropped:
                    xcb,ycb,wcb,hcb = cv2.boundingRect(contours_cropped[biggest_contour_idx])
                    #img_cropped[ycb:ycb+hcb, xcb:xcb+wcb] = cv2.erode(img_cropped[ycb:ycb+hcb, xcb:xcb+wcb], kernel, iterations=1)
                    cv2.imwrite('test_image_unprepared/' + str(img_cnt) + ".jpg", img_cropped[ycb:ycb+hcb, xcb:xcb+wcb])

                #cv2.imshow('img', img[y:y+h,x:x+w])
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            img_cnt += 1

    # Image preprocessing
    def resizeAndPad(img, size, padColor=0):

        h, w = img.shape[:2]
        sh, sw = size

        # interpolation method
        if h > sh or w > sw: # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

        # compute scaling and pad sizing
        if aspect > 1: # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # set pad color
        if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
            padColor = [padColor]*3

        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

        return scaled_img

    # Input images need to match
    # match MNIST dataset digits format (28x28)

    # Create folder which will contain
    # processed images ready for predicting
    if os.path.isdir('test_images_prepared'):
        shutil.rmtree('test_images_prepared')
    os.mkdir('test_images_prepared')
    os.mkdir('test_images_prepared/all_classes')

    for image_path in os.listdir('test_image_unprepared'):
        if '.jpg' not in image_path:
            continue
        # Read image in grayscale
        image = cv2.imread('test_image_unprepared/' + image_path, 0)

        # Invert black and white colors
        #image = cv2.bitwise_not(image)
        #cv2.imshow('img', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
        #cv2.imshow('img', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # Make image binary
        image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
        # Resize to 20x20 preserving aspect ratio
        image = resizeAndPad(image, (20, 20))
        # Pad the image so it ends up being 28x28
        # (just like MNIST dataset images are)
        image = cv2.copyMakeBorder(image, 4, 4, 4, 4, borderType=cv2.BORDER_CONSTANT)
        # Save the modified image
        #cv2.imshow('img', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        image_path_splitted = image_path.split('.')

        # Saving processed cropped characters ready to
        # be read by the generator
        cv2.imwrite('test_images_prepared/all_classes/' + image_path_splitted[0] + '_mnist' + '.' + image_path_splitted[1], image)

    train_dir = '.'

    train_datagen = ImageDataGenerator(#rescale=1./255,
        data_format='channels_first',
        validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=20,
    shuffle=True,
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '(', ')', 'div'],
    class_mode='categorical',
    subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_dir, # Same directory as training data
        target_size=(28, 28),
        color_mode='grayscale',
        batch_size=20,
        shuffle=True,
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '(', ')', 'div'],
        class_mode='categorical',
        subset='validation')

    # Load previously trained model
    model = load_model(h5_filename)

    # Create generator from cropped processed images
    test_generator = train_datagen.flow_from_directory(
        train_dir + '/test_images_prepared',
        target_size=(28, 28),
        color_mode='grayscale',
        batch_size=32,
        shuffle=False,
        class_mode=None,)
    test_generator.reset()


    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]

    # Sorting images in generator numerically
    # because contours were sorted and the input
    # cropped character images were numbered sequentially
    # (without this images are sorted alphabetically,
    # which is the wrong order)
    test_generator.filenames.sort(key=alphanum_key)
    test_generator.filepaths.sort(key=alphanum_key)

    # Get probabilities for each class
    predictions = model.predict(test_generator)
    # Get index of classes with highest probabilities
    predicted_class_indices = np.argmax(predictions,axis=1)
    # Convert those indices into actual class names
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    # Array containing predictions for each class or
    # number/symbol
    output_math_expression = predictions

    def convert_model_output_to_string(output_math_expression):
        for i in range(len(output_math_expression)):
            # NOTE: Class name wasn't '/' in the first place
            # because class names have to match folder names
            # which are being used for creating generators
            # But '/' can't be a valid folder name
            if output_math_expression[i] == 'div':
                output_math_expression[i] = '/'
            str_output = ''
        # Create string from list for math expression
        # parser/solver
        return str_output.join(output_math_expression)

    math_expression_string = convert_model_output_to_string(output_math_expression)

    # Parsing and calculating a given math
    # expression in a string format

    # Examples used for testing
    # All output correct results
    """
    math_expression_string = '174*36+(58-2)'
    math_expression_string = '80-(15-(20-8))*9'
    math_expression_string = '64/(24/(27/9))'
    math_expression_string = '((32/8)*(40-31))*200'
    math_expression_string = '90-(14-(61-5)/(48/6))*9'
    math_expression_string = '(720-220)*((12-7)*(62-54))'
    math_expression_string = '((160-40)/2)(9(3000/30))'
    math_expression_string = '(20-(36/(130-126)))*8'
    math_expression_string = '54-((42/(18-11))*((19+62)/9))'
    math_expression_string = '(((9-3)*(4+6))/3)*((12-8)*(33/(90/30)))'
    """

    # Parsing function takes list as input
    string_expr = math_expression_string
    math_expression_string = list(math_expression_string)

    # Convert numbers to integers
    for i in range(len(math_expression_string)):
        try:
            math_expression_string[i] = int(math_expression_string[i])
        except ValueError:
            pass

    op_priority = {'(': 2, '*': 2, '/': 2, '+': 1, '-': 1}

    def parse_output(idx):
        num = ''
        num_stack = []
        operator_stack = []

        def calc(num1, num2, op, same_priority=False):
            if not same_priority:
                operator_stack.pop()
            if op == '*':
                return num1*num2
            elif op == '/':
                return num1/num2
            elif op == '+':
                return num1+num2
            elif op == '-':
                return num1-num2
            return False

        def pop_stack_and_calc(op):
            stack_top_num = num_stack[-1]
            num_stack.pop()
            num_stack[-1] = calc(num_stack[-1], stack_top_num, op)

        # Calculates operations from left to right
        # (in case operations of same priority are in a series)
        def left_to_right_calc(steps_back):
            result = num_stack[steps_back-1]
            for i in range(steps_back, 0):
                result = calc(result, num_stack[i], operator_stack[i], True)
            i = -1
            while i >= steps_back:
                operator_stack.pop()
                num_stack.pop()
                i -= 1
            num_stack.pop()
            num_stack.append(result)

        def handle_operation(operation_string, ending_flag=False):
            if not operator_stack:
                operator_stack.append(operation_string)
            elif op_priority[operation_string] > op_priority[operator_stack[-1]] and not ending_flag:
                operator_stack.append(operation_string)
            else:
                steps_back = -1
                if len(operator_stack) > 1:
                    while steps_back > -len(operator_stack)+1 and op_priority[operator_stack[steps_back-1]] == op_priority[operator_stack[steps_back]]:
                        steps_back -= 1
                    left_to_right_calc(steps_back)
                else:
                    pop_stack_and_calc(operator_stack[-1])
                if not ending_flag:
                    operator_stack.append(operation_string)

        i = idx
        while i < len(math_expression_string):
            if isinstance(math_expression_string[i], int):
                num += str(math_expression_string[i])
                # If there is a number after closed brackets,
                # it means that we multiply the following number
                # with the result from inside brackets
                if i > 0 and math_expression_string[i-1] == ')':
                    operator_stack.append('*')
            if not isinstance(math_expression_string[i], int) or (isinstance(math_expression_string[i], int) and i == len(math_expression_string)-1):
                if i > 0 and math_expression_string[i-1] == ')' and isinstance(math_expression_string[i], int):
                    operator_stack.append('*')
                if len(num) > 0:
                    num_stack.append(int(num))
                    num = ''

                # Print state on stacks
                #print('num_stack:', num_stack)
                #print('operator_stack', operator_stack)

                if math_expression_string[i] == '(':
                    if i > 0 and (isinstance(math_expression_string[i-1], int) or math_expression_string[i-1] == ')'):
                        operator_stack.append('*')
                    if i+1 < len(math_expression_string):
                        return_dict = parse_output(i+1)
                        num_stack.append(return_dict['result'])
                        i = return_dict['idx']
                        if i == len(math_expression_string)-1:
                            i -= 1
                elif math_expression_string[i] == '*':
                    handle_operation('*')
                elif math_expression_string[i] == '/':
                    handle_operation('/')
                elif math_expression_string[i] == '+':
                    handle_operation('+')
                elif math_expression_string[i] == '-':
                    handle_operation('-')
                elif math_expression_string[i] == ')' or i == len(math_expression_string)-1:
                    while len(num_stack) > 1:
                        handle_operation(operator_stack[-1], True)
                    if len(num_stack) > 0:
                        return {'idx': i, 'result': float(num_stack[-1])}
                    else:
                        return {'idx': i, 'result': 0}
            i += 1

        if len(num_stack) > 0:
            return {'idx': i, 'result': float(num_stack[-1])}
        return 0

    print("Math expression: ", string_expr)
    result = parse_output(0)
    print("Result:", result['result'])
    r = result['result']
    return [r, string_expr]

if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1], 0)
    process_input_and_predict(img)
