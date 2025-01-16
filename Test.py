import os
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('binary_image_ranking_model')

# Define a function to load and preprocess images
def load_image(file_path):
    img = tf.keras.utils.load_img(file_path, target_size=(423, 1080))  # Resize image
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Directory containing test images
test_dir = 'testing/'

# Ensure the directory exists
if not os.path.exists(test_dir):
    print(f"Error: The directory '{test_dir}' does not exist.")
    exit()

# Process each image in the directory
for image_name in os.listdir(test_dir):
    image_path = os.path.join(test_dir, image_name)

    if not os.path.isfile(image_path):
        continue  # Skip non-file entries (e.g., directories)

    try:
        # Load and preprocess the image
        image = load_image(image_path)

        # Predict using the trained model
        prediction = model.predict(image)

        # Convert the prediction to a binary True/False value
        is_true = prediction[0][0] > 0.5

        # Output the result
        result = "True" if is_true else "False"
        print(f"Image: {image_name} | Prediction: {result} (Confidence: {prediction[0][0]:.2f})")

    except Exception as e:
        print(f"Error processing image '{image_name}': {e}")
