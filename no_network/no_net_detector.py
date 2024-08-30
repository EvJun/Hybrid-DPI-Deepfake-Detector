import os
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import hashlib


#This version of the detector simulates the fragmenting and restructuring of image payloads.
#Then the image is recreated and classified.  
#When classified, the image is put into a fake or real image folder.

model = tf.keras.models.load_model(r'path/to/model/CNN_DF_model.keras')
input_folder = r'path/test_images'  # Folder containing input images
real_output_folder = r'path/real_images'  # Folder to save classified real images
fake_output_folder = r'path/fake_images'  # Folder to save classified fake images
signature_file = r'path/signatures.txt'
os.makedirs(real_output_folder, exist_ok=True)
os.makedirs(fake_output_folder, exist_ok=True)

#Load existing hashes from the signature file.
def load_hashes(signature_file):
    if not os.path.exists(signature_file):
        return set()
    with open(signature_file, 'r') as f:
        return set(line.strip() for line in f)


known_hashes = load_hashes(signature_file)


#Generate hash for an image.
def generate_hash(image_data):
    return hashlib.sha256(image_data).hexdigest()


def preprocess_image(image_data):
    """Preprocess image data to the format required by the model."""
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(image_data):
    """Use the loaded model to make a prediction on the image data."""
    img_array = preprocess_image(image_data)
    predictions = model.predict(img_array)
    return predictions

def split_and_reconstruct_image(image_path):
    """This is to simulate splitting and reconstructing image data 
	as if it were divided via network payloads."""
    with open(image_path, 'rb') as img_file:
        image_data = img_file.read()

    
    chunk_size = 1400  # Size of each payload.
    chunks = [image_data[i:i + chunk_size] for i in range(0, len(image_data), chunk_size)]

    # Reconstruct the image from the data chunks.
    reconstructed_data = b''.join(chunks)
    return reconstructed_data

def classify_and_save_images():
    """Classify images as real/fake and divide them into their according folders."""
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)

            reconstructed_data = split_and_reconstruct_image(image_path)
            image_hash = generate_hash(reconstructed_data)

            # Check if the image hash matches any known fake hashes
            if image_hash in known_hashes:
                print(f"Classified as fake: {filename} via Hash Signature - {image_hash}")
                output_path = os.path.join(fake_output_folder, filename)
                with open(output_path, 'wb') as f:
                    f.write(reconstructed_data)
                continue

            # Predict if the image is real or fake
            predictions = predict_image(reconstructed_data)
            print(f'Predictions for {filename}: {predictions}')

            if predictions[0][0] < 0.5:  #0 for real, 1 for fake.
                print(f"Classified as real: {filename}")
                output_path = os.path.join(real_output_folder, filename)
            else:
                print(f"Classified as fake: {filename}")
                output_path = os.path.join(fake_output_folder, filename)

            # Save the classified image.
            with open(output_path, 'wb') as f:
                f.write(reconstructed_data)

classify_and_save_images()
