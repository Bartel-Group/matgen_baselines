import gdown
import os

# Your file ID
file_id = "1tsi3HblmOIY0Qb5DmdfFD6HVhhmG0GJ9"
output_file = "pre-trained.zip"

# Download using gdown
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, output_file, quiet=False)

print(f"Downloaded file size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
