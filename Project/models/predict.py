import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Define constants
input_shape = (224, 224, 3)

# Load fine-tuned models
resnet_model = load_model('resnet_fine_tuned.h5')
densenet_model = load_model('densenet_fine_tuned.h5')
inception_model = load_model('inception_fine_tuned.h5')
mobilenet_model = load_model('mobilenet_fine_tuned.h5')
alexnet_model = load_model('alexnet_fine_tuned.h5')

models = [resnet_model, densenet_model, inception_model, mobilenet_model]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def ensemble_predict(image_path):
    image = preprocess_image(image_path)
    predictions = []

    for model in models:
        prediction = model.predict(image)
        predictions.append(prediction)

    # Use majority voting for binary classification
    ensemble_prediction = np.mean(predictions, axis=0)
    if ensemble_prediction >= 0.5:
        return "Pneumonia"
    else:
        return "Normal"

# Example usage
image_path = './input/image.jpeg'
result = ensemble_predict(image_path)
print(f"The ensemble prediction is: {result}")








# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Define constants
# input_shape = (224, 224, 3)

# # Load fine-tuned models
# resnet_model = load_model('resnet_fine_tuned.h5')
# densenet_model = load_model('densenet_fine_tuned.h5')
# inception_model = load_model('inception_fine_tuned.h5')
# mobilenet_model = load_model('mobilenet_fine_tuned.h5')

# models = [resnet_model, densenet_model, inception_model, mobilenet_model]

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (input_shape[1], input_shape[0]))
#     img = img / 255.0  # Normalize pixel values
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# def ensemble_predict(image_path):
#     image = preprocess_image(image_path)
#     predictions = []

#     for model in models:
#         prediction = model.predict(image)
#         predictions.append(prediction)

#     # Calculate average prediction
#     ensemble_prediction = np.mean(predictions)

#     # Scale prediction to percentage (0 to 100)
#     percentage = ensemble_prediction * 100
#     return percentage

# # Example usage
# image_path = './input/image.jpeg'
# percentage = ensemble_predict(image_path)
# print(f"The predicted percentage of pneumonia is: {percentage}%")
