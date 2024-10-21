import requests
import os
import zipfile

def download_data(url, path, name):
    if os.path.isfile(f'{path}/{name}.zip'):
        print('Archive already exists. Skipping download.')
    else:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(f'{path}/{name}.zip', 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    if os.path.isdir(path):
        print('Directory already exists. Skipping extraction.')
    else:
        with zipfile.ZipFile(f'{path}/{name}.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

def download_dataset(url_images, url_annotations, path, images_n, annotations_n):
    download_data(url_images, path, images_n)
    download_data(url_annotations, path, annotations_n)
