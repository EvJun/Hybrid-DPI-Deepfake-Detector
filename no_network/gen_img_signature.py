import hashlib

def generate_file_hash(file_path):
    """Generate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        #Read and update hash in chunks of 4096 bytes.
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def add_hash_to_signatures(file_path, signature_file):
    """Generate hash of the file and add it to the signatures.txt file."""
    file_hash = generate_file_hash(file_path)
    print(f"Hash for {file_path}: {file_hash}")

    #Append the hash to the signatures.txt file
    with open(signature_file, "a") as f:
        f.write(file_hash + "\n")
    print(f"Hash added to {signature_file}")

# Example usage
file_path = r'path/image.jpg'  # Path to the image file
signature_file = r'path/signatures.txt'  # Path to the signatures.txt file
add_hash_to_signatures(file_path, signature_file)
