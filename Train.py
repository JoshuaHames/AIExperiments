import keras
import tensorflow as tf
import pandas as pd
import numpy as np
np.object = np.object_
from sklearn.model_selection import train_test_split
from keras import layers, models
import tensorflowjs as tfjs
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the CSV file with filenames and rankings
data = pd.read_csv('results.csv')

# Define a function to load and preprocess images
def load_image(file_path):
    # Load image, resize it to a consistent shape (128x128), and normalize pixel values
    img = tf.keras.utils.load_img(file_path, target_size=(423, 1080))
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img

# Prepare the images and their corresponding rankings
images = []
rankings = []

# Adjust the directory if needed to match where your images are stored
image_directory = 'Cropped/'

for index, row in data.iterrows():
    image_path = os.path.join(image_directory, row['filename'])
    images.append(load_image(image_path))
    rankings.append(row['ranking'])

# Convert to numpy arrays
images = np.array(images)
rankings = np.array(rankings)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, rankings, test_size=0.2, random_state=42)

# Build a simple CNN model for ranking
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(423, 1080, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # Single output for ranking
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=2
)

# Save the trained model
model.export('image_ranking_model')
tfjs.converters.save_keras_model(model, 'tfjs_model')



# Optionally, print a summary of the training process
print("Training complete. Model saved as 'image_ranking_model'.")