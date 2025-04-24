from ImageProcessing import safe_convert_to_png
import os


def load_images_from_directory(directory_path, saving_directory_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.emf', '.wmf')
    image_data = []

    # Make sure the saving directory exists
    os.makedirs(saving_directory_path, exist_ok=True)

    for image_path in os.listdir(directory_path):
        if image_path.lower().endswith(valid_extensions):
            file_path = os.path.join(directory_path, image_path)
            with open(file_path, 'rb') as img_file:
                image_bytes = img_file.read()
                image_name, image_extension = os.path.splitext(image_path)
                print(f"Before the image_name: {image_name}, image_extension: {image_extension}")
                try:
                    # Convert to PNG
                    image_bytes, image_extension = safe_convert_to_png(image_bytes, image_extension)
                    # Save the PNG to the output directory
                    output_path = os.path.join(saving_directory_path, f"{image_name}.png")
                    with open(output_path, 'wb') as out_file:
                        out_file.write(image_bytes)
                except Exception as e:
                    print(f"Could not convert {image_name} with extension {image_extension} \n Exception: {e}")
                    continue  # Skip adding to image_data in case of failure
                image_data.append((image_bytes, image_name, image_extension.lower()))
    return image_data

inp_images_path = "./testing_figures" ; out_images_path = "./testing_figures/png_images"
image_data = load_images_from_directory(inp_images_path, out_images_path)
