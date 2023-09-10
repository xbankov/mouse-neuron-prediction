import zipfile
from io import BytesIO

import requests
from pathlib import Path

from tqdm import tqdm


def download_dataset(root_dir: Path):
    url = "https://figshare.com/ndownloader/articles/6845348/versions/4"

    # Send a GET request to the URL to download the zip file
    response = requests.get(url, stream=True)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the total file size in bytes
        total_size = int(response.headers.get('content-length', 0))

        # Create a BytesIO object from the response content
        zip_data = BytesIO()

        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading', ascii=True) as pbar:
            for data in response.iter_content(chunk_size=1024):
                pbar.update(len(data))  # Update the progress bar
                zip_data.write(data)

        # Create a ZipFile object
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            # Specify the directory where you want to extract the contents
            extract_path = (root_dir / "dataset")
            extract_path.mkdir(exist_ok=True, parents=True)

            # Extract all files from the zip archive
            zip_ref.extractall(extract_path)

        print("Zip file downloaded and extracted successfully.")
    else:
        print("Failed to download the zip file. Status code:", response.status_code)


if __name__ == "__main__":
    download_dataset(Path("data"))
