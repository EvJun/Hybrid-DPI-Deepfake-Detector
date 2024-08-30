import tensorflow as tf
from PIL import Image
import numpy as np
from pydivert import WinDivert
import io

# Load the Keras model
model = tf.keras.models.load_model(r'path/CNN_DF_model.keras')
src_ip = "192.168.0.100"
src_port = 12345
dst_port = 80

def preprocess_image(image_data):
    """
    Preprocess image data to the format required by the model.
    """
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((256, 256))  # Adjust based on your model's expected input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(image_data):
    """
    Use the loaded model to make a prediction on the image data.
    """
    img_array = preprocess_image(image_data)
    predictions = model.predict(img_array)
    return predictions

def save_image(image_data, folder, count):
    """
    Save extracted image data to a file.
    """
    with open(f'{folder}/image_{count}.jpg', 'wb') as f:
        f.write(image_data)

def extract_images_from_data(data):
    """
    Extract JPEG image data from reassembled TCP data.
    """
    if b'Content-Type: image/jpeg' in data or b'Content-Type: image/jpg' in data:
        start = data.find(b'\xff\xd8')  # JPEG magic number
        end = data.find(b'\xff\xd9')  # JPEG end number
        if start != -1 and end != -1:
            return data[start:end + 2]  # Extract JPEG image
    return None

def main():
    packet_folder = r'path/collected_packets'
    image_folder = r'path/collected_images'
    image_count = 0  # Counter for saved images
    tcp_streams = {}  # Dictionary to store reassembled TCP streams by 4-tuple (src_ip, src_port, dst_ip, dst_port)

    # Corrected WinDivert filter string
    filter_str = f'tcp and ip.SrcAddr == {src_ip} and tcp.DstPort == {dst_port}'
    
    with WinDivert(filter_str) as w:
        for packet in w:
            key = (packet.src_addr, packet.src_port, packet.dst_addr, packet.dst_port)
            print(f"Captured packet from {key}")

            # Reassemble TCP streams
            if key not in tcp_streams:
                tcp_streams[key] = bytearray()
            tcp_streams[key].extend(packet.payload)
            print(f"Reassembled TCP stream from {key}, processing...")

            # Extract image data from the reassembled stream
            image_data = extract_images_from_data(tcp_streams[key])
            if image_data:
                print("Image found in reassembled stream, processing...")

                # Predict if the image is real or fake
                predictions = predict_image(image_data)
                print(f'Predictions: {predictions}')

                if predictions[0][0] < 0.5:  #Labels 0 for real, 1 for fake
                    print("Real image detected, saving...")
                    save_image(image_data, image_folder, image_count)
                    image_count += 1
                    w.send(packet)  # Forward the packet when flagged as real
                else:
                    print(f"Fake image detected, dropping packet #{packet.packet_id}.")
                    # Packet is not forwarded, effectively dropping it, its not reinjected into the stream

                # Clear the stream buffer after processing
                tcp_streams[key] = bytearray()

            else:
                print("No image found in this reassembled stream.")
                w.send(packet)  # Forward the packet if no image is found

if __name__ == "__main__":
    main()
