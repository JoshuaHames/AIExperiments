import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflowjs as tfjs
import os
import datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Disable GPU settings for CPU optimization
tf.config.set_visible_devices([], 'GPU')
num_threads = os.cpu_count()  # Use all available CPU threads
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

# Load the CSV file with filenames and binary labels
data = pd.read_csv('results.csv')

# Define a function to load and preprocess images
def load_image(file_path):
    img = tf.keras.utils.load_img(file_path, target_size=(294, 750))  # Resize image
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img

# Prepare the images and binary labels
images = []
labels = []

image_directory = 'Cropped/'  # Adjust path to match your setup

for index, row in data.iterrows():
    image_path = os.path.join(image_directory, row['filename'])
    if os.path.exists(image_path):
        images.append(load_image(image_path))
        labels.append(1 if row['rank'] else 0)  # Convert ranking to binary (True -> 1, False -> 0)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels, dtype=np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.25, random_state=42
)

# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True
)
datagen.fit(X_train)

# Define the model
image_input = Input(shape=(294, 750, 3), name="image_input")
cnn_branch = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
cnn_branch = layers.MaxPooling2D((2, 2))(cnn_branch)
cnn_branch = layers.Conv2D(64, (3, 3), activation='relu')(image_input)
cnn_branch = layers.MaxPooling2D((2, 2))(cnn_branch)
cnn_branch = layers.GlobalAveragePooling2D()(cnn_branch)
dense_layer = layers.Dense(128, activation='relu')(cnn_branch)
output = layers.Dense(1, activation='sigmoid', name="ranking_output")(dense_layer)

model = Model(inputs=image_input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Learning Rate Scheduler
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,  # Reduce the learning rate by half
    patience=3,  # Wait for 3 epochs with no improvement
    min_lr=1e-6  # Minimum learning rate
)

# Setup TensorBoard logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=8),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[tensorboard_callback, reduce_lr]
)

# Save the trained model
model.save('binary_image_ranking_model')
tfjs.converters.save_keras_model(model, 'tfjs_binary_model')

# Print completion message
print("Training complete. Model saved as 'binary_image_ranking_model'.")
