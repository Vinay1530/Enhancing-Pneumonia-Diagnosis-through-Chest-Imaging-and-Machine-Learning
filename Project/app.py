import os
import sys
from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from util import base64_to_pil


import cv2
import numpy as np


# Define constants
input_shape = (224, 224, 3)

# Load fine-tuned models
resnet_model = load_model('./models/resnet_fine_tuned.h5')
densenet_model = load_model('./models/densenet_fine_tuned.h5')
inception_model = load_model('./models/inception_fine_tuned.h5')
mobilenet_model = load_model('./models/mobilenet_fine_tuned.h5')
alexnet_model = load_model('./models/alexnet_fine_tuned.h5')

models = [resnet_model, densenet_model, inception_model, mobilenet_model]



def preprocess_image(img):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img




def ensemble_predict(img):
    image = preprocess_image(image_path)
    predictions = []

    for model in models:
        prediction = model.predict(image)
        predictions.append(prediction)

    # Use majority voting for binary classification
    ensemble_prediction = np.mean(predictions, axis=0)
    return ensemble_prediction







app = Flask(__name__)

MODEL_PATH = 'models/oldModel.h5'
model = load_model(MODEL_PATH)

print('Model loaded. Start serving...')

def model_predict(img, model):
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x = x / 255.0  # Normalize pixel values to be between 0 and 1
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="No image file provided")

    img_file = request.files['file']

    if img_file.filename == '':
        return jsonify(error="No selected file")

    img_path = os.path.join(os.path.dirname(__file__), 'uploads', 'image.jpg')
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(64, 64))
    preds = model_predict(img, model)

    result = preds[0, 0]

    # resultall = ensemble_predict(img)

    # result = (result+resultall)/2

    print(result)

    result1=result*100

    # if result > 0.5:
    #     return jsonify(result="PNEUMONIA")
    # else:
    #     return jsonify(result="NORMAL")



    label = "PNEUMONIA" if result > 0.5 else "NORMAL"

    return jsonify(result=label, raw_result=float(result1))




if __name__ == '__main__':
    app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
































# import os
# import sys
# from flask import Flask, request, render_template, jsonify
# from gevent.pywsgi import WSGIServer
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from util import base64_to_pil

# app = Flask(__name__)

# MODEL_PATHS = {
#     'AlexNet': 'models/oldModel.h5'
#     'ResNet': 'models/resnet_fine_tuned.h5',
#     'DenseNet': 'models/densenet_fine_tuned.h5',
#     'InceptionV3': 'models/inception_fine_tuned.h5',
#     'MobileNet': 'models/mobilenet_fine_tuned.h5'
# }

# models = {model_name: load_model(model_path) for model_name, model_path in MODEL_PATHS.items()}

# print('Models loaded. Start serving...')


# def model_predict(img, model):
#     x = image.img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#     x = x / 255.0  # Normalize pixel values to be between 0 and 1
#     preds = model.predict(x)
#     return preds


# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify(error="No image file provided")

#     img_file = request.files['file']

#     if img_file.filename == '':
#         return jsonify(error="No selected file")

#     img_path = os.path.join(os.path.dirname(__file__), 'uploads', 'image.jpg')
#     img_file.save(img_path)

#     img = image.load_img(img_path, target_size=(224, 224))  # Assuming images are resized to 224x224
#     results = {}

#     for model_name, model in models.items():
#         preds = model_predict(img, model)
#         results[model_name] = preds[0, 0]

#     print(results)

#     # Choose the model with the highest prediction
#     max_model = max(results, key=results.get)

#     if results[max_model] > 0.5:
#         return jsonify(result="PNEUMONIA", model=max_model)
#     else:
#         return jsonify(result="NORMAL", model=max_model)


# if __name__ == '__main__':
#     app.run(port=5002, threaded=False)

#     # Serve the app with gevent
#     http_server = WSGIServer(('0.0.0.0', 5000), app)
#     http_server.serve_forever()
