import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, DenseNet121, InceptionV3, MobileNet
from tensorflow.keras.models import save_model
from tensorflow.keras import layers, models

# Define input shape
input_shape = (224, 224, 3)

# Create and save ResNet50 model
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
save_model(resnet_model, 'resnet_model.h5')

# Create and save DenseNet121 model
densenet_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
save_model(densenet_model, 'densenet_model.h5')

# Create and save InceptionV3 model
inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
save_model(inception_model, 'inception_model.h5')

# Create and save MobileNet model
mobilenet_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
save_model(mobilenet_model, 'mobilenet_model.h5')

# Simplified implementation of AlexNet
alexnet_model = models.Sequential([
    layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1000, activation='softmax')  # Adjust the number of output units based on your needs
])

save_model(alexnet_model, './models/alexnet_model.h5')

