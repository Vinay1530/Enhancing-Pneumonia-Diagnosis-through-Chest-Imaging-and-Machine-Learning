import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Define constants
input_shape = (224, 224, 3)
batch_size = 32
epochs = 10

# Define paths to your dataset
train_dir = './chest_xray/train'
val_dir = './chest_xray/val'
test_dir = './chest_xray/test'

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Load pre-trained models
# resnet_model = load_model('./models/resnet_model.h5')
vgg16_model = load_model('./models/vgg16_model.h5')
# densenet_model = load_model('./models/densenet_model.h5')
# inception_model = load_model('./models/inception_model.h5')
# mobilenet_model = load_model('./models/mobilenet_model.h5')

# Train and evaluate each model
# models = [resnet_model, vgg16_model, densenet_model, inception_model, mobilenet_model]
models = [vgg16_model]

for model in models:
    # Freeze convolutional layers
    for layer in model.layers:
        layer.trainable = False
    
    # Add new fully connected layers for binary classification
    x = model.layers[-1].output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Create new model
    new_model = tf.keras.Model(inputs=model.input, outputs=output)

    # Compile the model
    new_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    new_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )

    # Evaluate on the test set
    evaluation = new_model.evaluate(test_generator)
    print(f"Model Accuracy on Test Set: {evaluation[1]*100:.2f}%")

    # Save the fine-tuned model
    new_model.save(f'{model.name}_fine_tuned.h5')











# import os
# import cv2
# import numpy as np

# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.models import load_model

# # Define constants
# input_shape = (224, 224, 3)
# batch_size = 32
# epochs = 1

# # Define paths to your dataset
# train_dir = './chest_xray/train'
# val_dir = './chest_xray/val'
# test_dir = './chest_xray/test'

# # Create data generators
# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=True
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode='binary',
#     shuffle=False
# )

# # Load fine-tuned models
# # model_names = ['resnet', 'vgg16', 'densenet', 'inception', 'mobilenet']
# models = []

# def train_and_save_model(base_model, model_name):
#     # Freeze convolutional layers
#     for layer in base_model.layers:
#         layer.trainable = False
    
#     # Add new fully connected layers for binary classification
#     x = base_model.layers[-1].output
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Dense(256, activation='relu')(x)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

#     # Create new model
#     new_model = tf.keras.Model(inputs=base_model.input, outputs=output)

#     # Compile the model
#     new_model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

#     # Train the model
#     new_model.fit(
#         train_generator,
#         steps_per_epoch=train_generator.samples // batch_size,
#         epochs=epochs,
#         validation_data=val_generator,
#         validation_steps=val_generator.samples // batch_size
#     )

#     # Evaluate on the test set
#     evaluation = new_model.evaluate(test_generator)
#     print(f"{model_name.capitalize()} Model Accuracy on Test Set: {evaluation[1]*100:.2f}%")

#     # Save the fine-tuned model
#     new_model.save(f'{model_name}_fine_tuned.h5')
    
#     return new_model

# # Train and save each model
# # for model_name in model_names:

# model_name = "mobilenet"

# base_model = load_model(f'./models/{model_name}_model.h5')
# trained_model = train_and_save_model(base_model, model_name)
# models.append(trained_model)











import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
input_shape = (224, 224, 3)
batch_size = 32
epochs = 3

# Define paths to your dataset
train_dir = './chest_xray/train'
val_dir = './chest_xray/val'
test_dir = './chest_xray/test'

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

# Define the AlexNet-like model
model = models.Sequential([
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
    layers.Dense(1, activation='sigmoid')  # Binary classification, adjust for your task
])

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Evaluate on the test set
evaluation = model.evaluate(test_generator)
print(f"Model Accuracy on Test Set: {evaluation[1]*100:.2f}%")

# Save the trained model
model.save('alexnet_fine_tuned.h5')
