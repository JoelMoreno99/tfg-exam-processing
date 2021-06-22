import pdf2image
import tempfile
import os
import imagehash
import time
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import imutils
import cv2
from scipy import ndimage
import math
import sys

def processAutoPages(pdf_path, images_path, poppler_path, model):
    """
    This function processes the PDF document to get images, detects the images forming part of the same exam resolution and
    saves them in a folder named as the NIU detected in the first image of the resolution.

    Parameters
    ----------
    pdf_path : (str)
        The path of the document.
    images_path : (str)
        The path to save the folders and the images.
    poppler_path : (str)
        The path of the C++ library called "Poppler".
    model : (tensorflow.sequential.model)
        The loaded model (neural network) used to classify the digits.
    exam_nius_list : (list)
        A list containing the correct NIU of every exam in the correct order.

    Returns
    -------
    nius_list : (list)
        A list containing the NIU detected on every exam resolution.
    """
    #Use temporary directory not to avoid getting a memory error
    with tempfile.TemporaryDirectory() as temp_path:
        #Convert document pages to images and save them in a list
        print("Reading the document... (it may take a while)")
        images_from_path = pdf2image.convert_from_path(pdf_path, output_folder=temp_path,
        poppler_path=poppler_path, dpi=300, fmt="jpeg", grayscale=True, use_pdftocairo=True)
        total_pages = len(images_from_path)
        
        #Rotate and save every image
        print("Processing and saving the images...")
        base_image = images_from_path[0].rotate(90, expand=True)
        resolution = 0
        page = 0
        nius_list = []

        for i in range(len(images_from_path)):
            rotated_image = images_from_path[i].rotate(90, expand=True)

            #Calculate the hash of the images and the difference between them to check if the image corresponds to a new resolution
            diff = imagehash.average_hash(base_image) - imagehash.average_hash(rotated_image)
            if(diff < 7):
                #The image corresponds to a new resolution so we detect student's NIU    
                croped_image = getROI(np.array(rotated_image))

                niu, rgb_image = detectNIU(croped_image, model)
                nius_list.append(niu)

                resolution += 1
                page = 0
                
            #Save the image in the corresponding folder and change the page counter
            if niu == 'unknown':
                aux_file_path = images_path + '/resolution' + str(resolution) + '- error/page'+ str(page) + '.jpg'
            else:
                aux_file_path = images_path + '/resolution' + str(resolution) + '-' + str(niu) + '/page'+ str(page) + '.jpg'
            os.makedirs(os.path.dirname(aux_file_path), exist_ok=True)
            rotated_image.save(aux_file_path, 'JPEG')
            page += 1
    
    cv2.destroyAllWindows()
    return total_pages, nius_list

def detectNIU(image, model):
    """
    This function processes the document by calculating the similarity between the new image and the base image (exam header).

    Parameters
    ----------
    image : (np.array)
        The image with the NIU to recognize.
    model : (tensorflow.sequential.model)
        The loaded model (neural network) used to classify the digits.

    Returns
    -------
    niu : (str)
        The NIU recognized in the input image.
    rgb_image : (np.array)
        The image with the bounding boxes and the predictions.
    """
    #Initialise the NIU
    niu = 'unknown'

    #Process the image
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)[1]
    image = cv2.GaussianBlur(image, (3, 3), 0)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    #Detect edges, find contours and sort the resulting contours
    edged = cv2.Canny(image, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sorted_cnts = sortContours(cnts)

    #List of bounding boxes and the associated characters to OCR
    chars = []
    valid_contours = []

    #Loop over contorus (first filter)
    for c in sorted_cnts:
        #Compute the bounding box of the contour
        (x, y, w, h) = c

        #Filter the bounding box to discard the ones too small or too large
        if (w >= 10 and w <= 120) and (h >= 20 and h <= 150):
            #Detect if the center of the contour (p) is inside (or near) a valid contour (correct size and not inside another contour)
            p = (int(x+w/2), int(y+h/2))
            validation_contours_list = []
            for contour in valid_contours:
                (cx, cy, cw, ch) = contour
                if p[0] > cx and p[0] < cx+cw and p[1] > cy and p[1] < cy+ch:
                    validation_contours_list.append(True) #Inside
                else:
                    validation_contours_list.append(False) #Outside

            #Continue with the detection if the contour is not inside another contour
            if not any(validation_contours_list):
                #Insert the contour into the list of valid contours
                valid_contours.append((x, y, w, h))

    #Loop over valid_contours (second filter)
    valid_contours_filtered = []
    for c in valid_contours:
        #Compute the bounding box of the contour
        (x, y, w, h) = c

        #Detect if the center of the contour (p) is inside (or near) a valid contour (correct size and not inside another contour)
        p = (int(x+w/2), int(y+h/2))
        validation_contours_list = []
        for contour in valid_contours:
            (cx, cy, cw, ch) = contour
            if x != cx and y != cy and p[0] > cx and p[0] < cx+cw and p[1] > cy and p[1] < cy+ch:
                validation_contours_list.append(True) #Inside
            else:
                validation_contours_list.append(False) #Outside

        #Continue with the detection if the contour is not inside another contour
        if not any(validation_contours_list):
            #Insert the contour into the list of valid contours
            valid_contours_filtered.append((x, y, w, h))
    
    #Loop over valid_contours_filtered (third filter)
    bounding_boxes = []
    combined_bounding_boxes = []
    for c in valid_contours_filtered:
        #Compute the bounding box of the contour
        (x, y, w, h) = c

        #Detect if the digit contour is split in a half (one above the other)
        validation_contours_list = []
        for contour in valid_contours_filtered:
            (cx, cy, cw, ch) = contour
            if (x, y, w, h) != (cx, cy, cw, ch) and (x, y, w, h) not in combined_bounding_boxes and (cx, cy, cw, ch) not in combined_bounding_boxes:
                if ((abs(x-cx) < 5 and abs((y+h)-cy) < 5) or (abs((x+w)-(cx+cw)) < 5 and abs((y+h)-cy) < 5)):
                    new_x = min([x, cx])
                    new_y = min([y, cy])
                    new_w = max([x+w, cx+cw]) - new_x
                    new_h = max([y+h, cy+ch]) - new_y
                    cv2.rectangle(rgb_image, (new_x, new_y), (new_x+new_w, new_y+new_h), (0, 0, 255), 4)
                    bounding_boxes.append((new_x, new_y, new_w, new_h))
                    combined_bounding_boxes.append((x, y, w, h))
                    combined_bounding_boxes.append((cx, cy, cw, ch))

        if (x, y, w, h) not in combined_bounding_boxes:
            bounding_boxes.append((x, y, w, h))

    #Loop over bounding_boxes (fourth filter)
    final_bounding_boxes = []
    (lx, ly, lw, lh) = bounding_boxes[-1]
    for c in bounding_boxes:
        #Compute the bounding box of the contour
        (x, y, w, h) = c
        if abs(y-ly) < 30:
            if w > h*1.5:
                first = (x, y, int(w/2), h)
                second = (x+int(w/2), y, int(w/2), h)
                final_bounding_boxes.append(first)
                final_bounding_boxes.append(second)
            else:
                final_bounding_boxes.append(c)

    #Loop over final_bounding_boxes (last loop)
    for c in final_bounding_boxes:
        #Compute the bounding box of the contour
        (x, y, w, h) = c

        #Detect if the contour contains more than one digit and extract the character (ROI)
        roi = image[y:y+h, x:x+w]
        
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        #Resize along height or width
        if tW > tH:
            thresh = imutils.resize(thresh, width=18)
        else:
            thresh = imutils.resize(thresh, height=18)

        #Grab the new dimensions and pad the resized image to make it 32x32 as the MNIST dataset images
        (tH, tW) = thresh.shape
        dX = int(max(0, 18 - tW) / 2.0)
        dY = int(max(0, 18 - tH) / 2.0)

        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded = np.pad(padded, ((7,7),(7,7)), 'constant')
        padded = cv2.resize(padded, (32, 32))

        #Prepare the padded image for the model
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)

        #Update the list of characters to OCR
        chars.append((padded, (x, y, w, h)))

    #Extract the bounding box and the padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    #Standardization
    mean_px = chars.mean().astype(np.float32)
    std_px = chars.std().astype(np.float32)
    chars = (chars - mean_px)/(std_px)

    #OCR the characters using the model
    preds = model.predict(chars)

    # define the list of label names
    labelNames = "0123456789"
    labelNames = [l for l in labelNames]

    #Loop over the predictions and bounding box locations together
    possible_niu = ''
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        #Find the index of the label with the largest corresponding probability
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        possible_niu += label

        #Draw the prediction on the image
        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(rgb_image, label, (int(x+w/2), int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    #Detect if we recongized at least 7 numbers (ONLY NUMBERS)
    if len(possible_niu) >= 7:
        possible_niu = possible_niu[-7:]
        if not any(c.isalpha() for c in possible_niu):
            niu = possible_niu
    
    return niu, rgb_image

def getAngle(image):
    """
    This function calculates the angle to correctly align the image horizontally

    Parameters
    ----------
    image : (np.array)
        The image with the NIU to recognize.

    Returns
    -------
    angle : (float)
        The angle to rotate the image in order to align the image horizontally.
    """
    #Get the edged image
    edges = cv2.Canny(image, 100, 100, apertureSize=3)
    
    #Get all the lines with the Hough transform method
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    #Calculate the angle to horizontally align every line
    angles = []
    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    #Return the mean of the angles
    return np.average(angles)

def sortContours(cnts):
    """
    This function sorts the list of detected contours in a reading way (top to bottom and left to right).

    Parameters
    ----------
    cnts : (np.array)
        A list of contours detected in the image.
    image : (np.array)
        The original image.

    Returns
    -------
    sorted_cnts : (np.array)
        The sorted contours list.
    """
    first_layer = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        first_layer.append((x, y, w, h))

    first_layer.sort(key=lambda x:x[0])

    return first_layer 

def getROI(image):
    """
    This function processes the image to get the area where the NIU is ubicated.

    Parameters
    ----------
    image : (np.array)
        The image to process.

    Returns
    -------
    return_image : (np.array)
        The region of the image where the NIU is ubicated.
    """
    #General cropping
    heigh, width = image.shape
    image = image[0:int(heigh/6), int(width/4):int(width/2)+20]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return_image = image

    #Process the image to get "stains"
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)[1]
    image = cv2.dilate(image, np.ones((5,5),np.uint8), iterations = 2)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=1)

    #Detect edges, find contours and sort the resulting contours
    edged = cv2.Canny(image, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    #Loop over contorus
    width_array = []
    for c in cnts:
        #Compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        if h < 80:
            width_array.append(0)
        else:
            width_array.append(w)
        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    #Get ROI with the handrwitten NIU
    roi = cnts[np.argmax(width_array)]
    (x, y, w, h) = cv2.boundingRect(roi)
    cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #Rotate the image
    return_image = return_image[y:y+h, x:x+w]
    angle = getAngle(return_image)
    return_image = ndimage.rotate(return_image, angle, cval=255, reshape=False)

    #Add precision to the ROI if it's possible
    heigh, width = return_image.shape
    if heigh > 80:
        #Find the horizontal black line below the NIU
        image = return_image
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)[1]
        row_sum_array = []
        for row in image[:]:
            row_sum_array.append(np.sum(row))
        horizontal_black_line = np.argmin(row_sum_array)-5

        #Find the the horizontal empty spot (white line) above NIU
        horizontal_white_line = 0
        cropped_row_sum_array = row_sum_array[10:horizontal_black_line-10]
        white_line_value = max(cropped_row_sum_array)
        
        for i, e in reversed(list(enumerate(cropped_row_sum_array))):
            if e > white_line_value - 30:
                horizontal_white_line = i
                break
        
        return_image = return_image[horizontal_white_line:horizontal_black_line, int(width/2):]

    return return_image

def main():
    start_time = time.time()

    model = load_model(os.getcwd() + '/auxiliar/trained_LeNet5_v4.h5')
    poppler_path = os.getcwd() + '/auxiliar/poppler-21.02.0/Library/bin'
    pdf_path = os.getcwd() + '/auxiliar/documents/' + pdf_name

    images_path = os.getcwd() + '/auxiliar/images/' + pdf_name[:-4]
    os.makedirs(os.path.dirname(images_path), exist_ok=True)

    total_pages, detected_nius = processAutoPages(pdf_path, images_path, poppler_path, model)

    #Open the TXT file to write the NIUs
    f = open(images_path + "\detectedNIUs.txt", "w+")
    resolution = 1
    for niu in detected_nius:
        f.write('Resolution ' + str(resolution) + '- ' + niu)
        f.write('\n')
        resolution += 1
    f.close()

    print('--------------------------------------------------------------')
    print('Pages processed: ', total_pages)
    print('Number of resolutions detected: ', len(detected_nius))
    print('Number of NIUs recognized: ', len(detected_nius) - detected_nius.count('unknown'))
    elapsed_time = time.time() - start_time
    print('Elapsed time: ', round(elapsed_time, 2), ' seconds.')
    print('--------------------------------------------------------------')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('--------------------------------------------------------------')
        print('Error: The name of the PDF file was not specified as an argument')
        print('--------------------------------------------------------------')
    else:
        pdf_name = str(sys.argv[1])
        main()