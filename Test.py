import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Function to preprocess the image
def preprocess_image(image_path):
    target_size = (423, 1080)  # Match the input shape of the model
    img = Image.open(image_path)
    img = img.resize(target_size)  # Resize image
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to load metadata for testing (dummy data or real metadata)
def get_test_metadata():
    # Replace this with actual metadata, or adjust for your case
    # Example: followers, following, numPosts
    return np.array([[1000, 500, 100]])  # Modify with realistic test values

def main():
    # Load the trained model
    model = load_model('image_ranking_model')  # Update with the correct model path

    # Define the testing folder
    testing_folder = "testing"

    if not os.path.exists(testing_folder):
        print(f"Testing folder '{testing_folder}' not found. Exiting.")
        return

    # Iterate over all image files in the testing folder
    for filename in os.listdir(testing_folder):
        file_path = os.path.join(testing_folder, filename)

        # Check if it's an image file
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"Skipping non-image file: {filename}")
            continue

        try:
            # Preprocess the image
            image_array = preprocess_image(file_path)
            metadata_array = get_test_metadata()

            # Make a prediction
            prediction = model.predict([image_array, metadata_array])
            predicted_rating = prediction[0][0]

            # Output the result
            print(f"Image: {filename}, Predicted Rating: {predicted_rating:.2f}")

        except Exception as e:
            print(f"Error processing file '{filename}': {e}")

if __name__ == "__main__":
    main()
