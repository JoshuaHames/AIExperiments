import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models, Model, Input
import tensorflowjs as tfjs
import os
from keras.callbacks import TensorBoard
import datetime

# Enable GPU memory growth
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
    # Load image, resize it to a consistent shape (423x1080), and normalize pixel values
    img = tf.keras.utils.load_img(file_path, target_size=(423, 1080))
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img

# Prepare the images, metadata, and corresponding rankings
images = []
metadata = []
rankings = []

# Adjust the directory if needed to match where your images are stored
image_directory = 'Cropped/'

for index, row in data.iterrows():
    image_path = os.path.join(image_directory, row['filename'])
    images.append(load_image(image_path))
    metadata.append([row['followCount'], row['followingCount'], row['numPosts']])
    rankings.append(row['ranking'])

# Convert to numpy arrays
images = np.array(images)
metadata = np.array(metadata, dtype=np.float32)
rankings = np.array(rankings, dtype=np.float32)

# Split the data into training and testing sets
X_train_images, X_test_images, X_train_metadata, X_test_metadata, y_train, y_test = train_test_split(
    images, metadata, rankings, test_size=0.25, random_state=42
)

# Define the CNN branch for processing images
image_input = Input(shape=(423, 1080, 3), name="image_input")
cnn_branch = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
cnn_branch = layers.MaxPooling2D((2, 2))(cnn_branch)
cnn_branch = layers.Conv2D(64, (3, 3), activation='relu')(cnn_branch)
cnn_branch = layers.Attention()([cnn_branch, cnn_branch]) 
cnn_branch = layers.MaxPooling2D((2, 2))(cnn_branch)
cnn_branch = layers.Conv2D(128, (3, 3), activation='relu')(cnn_branch)
cnn_branch = layers.MaxPooling2D((2, 2))(cnn_branch)
cnn_branch = layers.GlobalAveragePooling2D()(cnn_branch)

# Define the dense branch for processing metadata
metadata_input = Input(shape=(3,), name="metadata_input")
dense_branch = layers.Dense(64, activation='relu')(metadata_input)
dense_branch = layers.Dropout(0.1)(dense_branch)
dense_branch = layers.Dense(32, activation='relu')(dense_branch)


# Combine the two branches
combined = layers.Concatenate()([cnn_branch, dense_branch])
combined = layers.Dense(512, activation='relu')(combined)
combined = layers.Dropout(0.3)(combined)
combined = layers.Dense(256, activation='relu')(combined)
combined = layers.Concatenate()([combined, cnn_branch, dense_branch])
combined = layers.Dropout(0.3)(combined)
combined = layers.BatchNormalization()(combined)
combined = layers.Dense(128, activation='relu')(combined)
output = layers.Dense(1, activation='sigmoid', name="ranking_output")(combined)  # Single output for ranking

# Build the model
model = Model(inputs=[image_input, metadata_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
# Train the model
history = model.fit(
    {"image_input": X_train_images, "metadata_input": X_train_metadata},
    y_train,
    validation_data=(
        {"image_input": X_test_images, "metadata_input": X_test_metadata},
        y_test
    ),
    epochs=50,
    batch_size=8,
    callbacks=[tensorboard_callback],
    verbose=2,
)

# Save the trained model
model.save('image_ranking_model')
tfjs.converters.save_keras_model(model, 'tfjs_model')

# Optionally, print a summary of the training process
print("Training complete. Model saved as 'image_ranking_model'.")