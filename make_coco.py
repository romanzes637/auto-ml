"""Make COCO dataset by COCO files

Make COCO dateset from 3 COCO files: train, val and test:
1. Make directory tree
2. Download images from "image" field of COCO files to img_dir

Example:
    python make_coco.py
    
Attributes:
    train_file (str): COCO file for train data
    val_file (str): COCO file for val data
    test_file (str): COCO file for test data

Dataset structure:
dataset_root
|__ train_dir
   |__ train_file
   |__ img_dir
      |__ image1
      |__ image2
      |__ image3
      |__ ...
|__ val_dir
   |__ val_file
   |__ img_dir
      |__ image1
      |__ image2
      |__ image3
      |__ ...
|__ test_dir
   |__ test_file
   |__ img_dir
      |__ image1
      |__ image2
      |__ image3
      |__ ...
"""

from pathlib import Path
import shutil
import json
import urllib.request
from tqdm import tqdm


# Files in COCO format (COCO files)
train_file = 'train_coco.json'
val_file = 'val_coco.json'
test_file = 'test_coco.json'

dataset_root = 'dataset_coco'  # Dataset dir
train_dir = 'train'  # Dir at dataset_root for train data (images and COCO file) 
val_dir = 'val'  # Dir at dataset_root for val data (images and COCO file) 
test_dir = 'test'  # Dir  at dataset_root for test data (images and COCO file)
img_dir = 'images'  # Dir at train_dir, val_dir and test_dir for images

url_key = 'coco_url'  # Key for image retrieving at 'images' field of COCO file
file_name_key = 'file_name'  # Key for image name at 'images' field of COCO file

# Make dataset dir
dataset_path = Path(dataset_root)
dataset_path.mkdir(parents=True, exist_ok=True)

# Make dirs for train, val and test data
train_path = dataset_path / train_dir
val_path = dataset_path / val_dir
test_path = dataset_path / test_dir
train_path.mkdir(parents=True, exist_ok=True)
val_path.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)

# Make dirs for images
train_img_path = train_path / img_dir
val_img_path = val_path / img_dir
test_img_path = test_path / img_dir
train_img_path.mkdir(parents=True, exist_ok=True)
val_img_path.mkdir(parents=True, exist_ok=True)
test_img_path.mkdir(parents=True, exist_ok=True)

# Copy COCO file to data dirs
shutil.copy(train_file, train_path)
shutil.copy(val_file, val_path)
shutil.copy(test_file, test_path)
train_file_path = train_path / train_file
val_file_path = val_path / val_file
test_file_path = test_path / test_file

# Get dataset images
def get_images(coco_file, img_dir, url_key, file_name_key):
  with open(coco_file) as f:
    data = json.load(f)
  img_dir.mkdir(parents=True, exist_ok=True)
  for image in tqdm(data['images']):
    urllib.request.urlretrieve(image[url_key], img_dir / image[file_name_key])

get_images(train_file_path, train_img_path, url_key, file_name_key)  # train
get_images(val_file_path, val_img_path, url_key, file_name_key)  # val
get_images(test_file_path, test_img_path, url_key, file_name_key)  # test
