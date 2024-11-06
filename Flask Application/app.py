from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# Define the class names globally
class_names = ['Alaxan', 'Bactidol', 'Bioflu', 'Biogesic', 'DayZinc',
               'Decolgen', 'Fish Oil', 'Kremil S', 'Medicol', 'Neozep']

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = img.astype('float32')
    return img

def load_model_and_history(model_path, history_path):
    model = keras.models.load_model(model_path)
    history = np.load(history_path, allow_pickle='TRUE').item()
    return model, history

# Paths to saved models and their training history
model1_path = './Models/CNN_Model.keras'
model1_history_path = './Models/CNN_training_history.npy'
model1, history1 = load_model_and_history(model1_path, model1_history_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class_name = None  # Variable to hold predicted class name
    uploaded_image_url = None  # Variable to hold the uploaded image URL

    if request.method == 'POST':
        imagefile = request.files['imagefile']
        if imagefile:
            image_path = os.path.join("static", "uploads", imagefile.filename)  # Save in 'static/uploads'
            imagefile.save(image_path)
            uploaded_image_url = f"/static/uploads/{imagefile.filename}"  # URL to display the image
            testing_image = preprocess_image(image_path)

            # Prepare the image for prediction
            testing_image_input = np.expand_dims(testing_image, axis=0)
            predictions1 = model1.predict(testing_image_input)
            predicted_class_index = tf.argmax(predictions1, axis=1)[0].numpy()
            predicted_class_name = class_names[predicted_class_index]

    return render_template('index.html', predicted_class_name=predicted_class_name, uploaded_image_url=uploaded_image_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
