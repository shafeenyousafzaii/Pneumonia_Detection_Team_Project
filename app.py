from flask import Flask, render_template, request
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your Keras model
model = load_model('my_model3.h5')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
@app.route('/label', methods=['POST'])
def label():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file'
        
        if file:
            # Save the uploaded file to a folder in your server
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Load and preprocess the image
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize the image
            
            # Make prediction
            prediction = model.predict(img_array)
            accuracy = None  # Placeholder for accuracy
            
            threshold = 0.55
            pred_class = (prediction > threshold).astype(int)
            if pred_class[0][0] ==1:
                diagnosis = "Pneumonia"
                accuracy = prediction[0][0]  # Assuming prediction is the probability of pneumonia
            else:
                diagnosis = "Pneumonia Not Detected"
                accuracy =  prediction[0][0]
            
            # Render the appropriate template based on the diagnosis
            if diagnosis == "Pneumonia":
                return render_template('1.html', diagnosis=diagnosis, accuracy=accuracy)
            else:
                return render_template('0.html', diagnosis=diagnosis, accuracy=accuracy)
    
    return 'Method not allowed'

if __name__ == '__main__':
    app.run(debug=True)
