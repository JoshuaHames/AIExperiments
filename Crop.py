from PIL import Image
import os

def process_images_in_folder(folder_path, output_folder, target_height):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip non-image files
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            print(f"Skipping non-image file: {filename}")
            continue

        try:
            # Open the image
            with Image.open(file_path) as img:
                width, height = img.size

                if height >= target_height:
                    # Crop the image (keep the top part)
                    processed_img = img.crop((0, 0, width, target_height))
                else:
                    # Add black space to extend the bottom of the image
                    new_img = Image.new("RGB", (width, target_height), (0, 0, 0))
                    paste_position = (0, 0)
                    new_img.paste(img, paste_position)
                    processed_img = new_img

                # Save the processed image to the output folder
                output_path = os.path.join(output_folder, filename)
                processed_img.save(output_path)
                print(f"Processed and saved: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Specify the input folder, output folder, and target height
    input_folder = "raw/"
    output_folder = "cropped/"
    target_height = 2000  # Replace with your desired height

    process_images_in_folder(input_folder, output_folder, target_height)