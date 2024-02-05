from django.shortcuts import render
from django.conf import settings
from .forms import Image_Form
from .models import Ancient_Image
from .models import Verify
import tensorflow as tf
import cv2
import os
import numpy as np
from scipy.ndimage import interpolation as inter # for image Processing
import imutils # For Character Segementation
from gtts import gTTS
import gtts
import os   
import time   

IMAGE_FILE_TYPES = ['png', 'jpg', 'jpeg']

def create_image_uploader(request): 
    
    form = Image_Form()
    if request.method == 'POST':
        form = Image_Form(request.POST, request.FILES)
        if form.is_valid():
            image_prop = form.save(commit=False)
            image_prop.picture = request.FILES['picture']
            print(type(image_prop.picture))
            file_type = image_prop.picture.url.split('.')[-1]
            file_type = file_type.lower()
            if file_type not in IMAGE_FILE_TYPES:
                return render(request, 'uploader/error.html')
            try:
                image_prop.save()
            except:
                print("assaf")
            
            cleared_image = image_processing(image_prop.picture) # Image Processing
            segmented = character_segmentation(cleared_image) # Characters as list (in sorted)  
            
            PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
            image_path = os.path.dirname(PROJECT_ROOT)+"\media"
            directory = str(image_prop.id)
            path = os.path.join(image_path, directory) 
            os.mkdir(path) 
            
            characters = []
            global predicted
            predicted = []
            predictedd = []
            i = 0
            for character in segmented:
                if i<3:
                    i = i + 1
                    continue
                if i>25:
                    break
                obj = Ancient_Image()
                cv2.imwrite(os.path.join(path , str(i)+".jpg"), character)
                obj.picture = str(image_prop.id)+"/"+str(i)+".jpg"
                characters.append(obj.picture)
                prediction = predict(obj.picture)
                predicted.append(prediction)
                i = i+1
                
            prediction = predict(image_prop.picture)
            image_path = str(image_prop.picture)
            image_prop.letter = prediction
            image_prop.save()
            for out in predicted:
                y=out.split("-")
                predictedd.append(y[1])
                tts = gtts.gTTS(text=y[1], lang='ta')

                tts.save("media/mp3/"+y[1]+".mp3")
            print(predictedd)

            path="media/mp3/"  
  
            zipped_data = zip(characters, predictedd)

            return render(request, 'uploader/details.html',{'image_path': image_path, 'image_prop':image_prop, 'disable_button':False, 'zipped_data':zipped_data,"path":path})
    context = {"form": form}
    
    return render(request, 'uploader/create.html', context)

def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def predict(image):
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.dirname(PROJECT_ROOT)+"\media"+"\\"+str(image)
    DATADIR = os.path.dirname(PROJECT_ROOT)+"\static\Labelled Dataset - Fig 51"

    CATEGORIES = []
    files = ['1 - Multipart','2 - Unknown']
    for directoryfile in os.listdir(DATADIR):
        if(directoryfile in files):
            continue
        CATEGORIES.append(directoryfile)
    modell = tf.keras.models.load_model("uploader/CNN .h5")
    image = prepare(image_path)
    image =  tf.cast(image, tf.float32)
    prediction = modell.predict([image])
    prediction = list(prediction[0])
    character = CATEGORIES[prediction.index(max(prediction))]
    return character
    

def verify_image(request,id):
    if request.method == 'POST':
        obj = Ancient_Image.objects.get(pk=id)
        r = Verify(image=obj, verify=True)
        r.save()

        image_path = str(obj.picture)
        context = {
        "image_prop": obj,
        "image_path": image_path,
        "disable_button" : True
        }
        return render(request, 'uploader/details.html', context)

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated

def image_processing(image):
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    image_path = os.path.dirname(PROJECT_ROOT)+"\media"+"\\"+str(image)

    image = cv2.imread(image_path)

    angle, rotated = correct_skew(image)
    gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(rotated, [c], -1, (255,255,255), 5)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(rotated, [c], -1, (255,255,255), 5)

    gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
    filter1 = cv2.medianBlur(gray,5)
    filter2 = cv2.GaussianBlur(filter1,(5,5),0)
    dst = cv2.fastNlMeansDenoising(filter2,None,17,9,17)
    th1 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    return th1

def character_segmentation(image):
    
    gray = cv2.GaussianBlur(image, (7, 7), 0)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilate = cv2.dilate(thresh1, None, iterations=2)
    cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    print(cnts)
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * image.shape[1] )

    images = []
    
    for cnt in sorted_ctrs:
        if(cv2.contourArea(cnt) < 200):
            continue
        
        x,y,w,h = cv2.boundingRect(cnt)
        roi = image[y:y+h, x:x+w]
        images.append(roi)
    
    return images

