from django.shortcuts import render
import os
import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model_path = os.path.join(os.path.dirname(__file__), 'model', 'plant_disease_model.h5')
model = load_model(model_path)
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def index(request):
    return render(request, 'index.html')

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os

def predict(request):
    if request.method == 'POST' and request.FILES.get('leaf'):
        image_file = request.FILES['leaf']

        # Save uploaded file temporarily
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        file_path = fs.path(filename)

        # Load and preprocess the image for prediction
        img = image.load_img(file_path, target_size=(224, 224))  # Match model's input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]

        # Delete uploaded image
        os.remove(file_path)

        # Return result on same page
        return render(request, 'index.html', {'result': predicted_class})

    return render(request, 'index.html')
