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

Dataset structure::

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
import argparse
from pprint import pprint

from tqdm import tqdm


def get_images(coco_file, img_dir, url_key, file_name_key):
    """Get dataset images
    """
    with open(coco_file) as f:
        data = json.load(f)
    img_dir.mkdir(parents=True, exist_ok=True)
    for image in tqdm(data['images']):
        urllib.request.urlretrieve(image[url_key], img_dir / image[file_name_key])


if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='make_coco_input.json')
    parser.add_argument('-D', '--dataset_dir', help='dataset_coco')
    parser.add_argument('-t', '--train_file', help='train_coco.json')
    parser.add_argument('-v', '--val_file', help='val_coco.json')
    parser.add_argument('-s', '--test_file', help='test_coco.json')
    parser.add_argument('-T', '--train_dir', help='train')
    parser.add_argument('-V', '--val_dir', help='val')
    parser.add_argument('-S', '--test_dir', help='test')
    parser.add_argument('-I', '--img_dir', help='images')
    parser.add_argument('-u', '--url_key', help='coco_url')
    parser.add_argument('-f', '--file_name_key', help='file_name')
    print('Command line arguments')
    cmd_args = vars(parser.parse_args())  # convert from namespace to dict
    pprint(cmd_args)
    print('Config arguments')
    if cmd_args['input'] is not None:
        with open(cmd_args['input']) as f:
            cfg_args = json.load(f)
    else:
        cfg_args = {}
    pprint(cfg_args)
    print('Arguments')
    for k, v in cmd_args.items():  # Update cfg args by cmd args
        if v is not None or k not in cfg_args:
            cfg_args[k] = v
    pprint(cfg_args)
    dataset_dir = cfg_args['dataset_dir']
    train_file = cfg_args['train_file']
    val_file = cfg_args['val_file']
    test_file = cfg_args['test_file']
    train_dir = cfg_args['train_dir']
    val_dir = cfg_args['val_dir']
    test_dir = cfg_args['test_dir']
    img_dir = cfg_args['img_dir']
    url_key = cfg_args['url_key']
    file_name_key = cfg_args['file_name_key']

    # Make dataset dir
    dataset_path = Path(dataset_dir)
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

    # Get images
    print('Train images')
    get_images(train_file_path, train_img_path, url_key, file_name_key)
    print('Validation images')
    get_images(val_file_path, val_img_path, url_key, file_name_key)
    print('Test images')
    get_images(test_file_path, test_img_path, url_key, file_name_key)
