import shutil
import zipfile
import urllib.request
import os

if __name__ == '__main__':
    url = 'https://github.com/open-mmlab/mmdetection/archive/refs/heads/master.zip'
    file_name = 'mmdetection.zip'
    configs_dir = 'configs'
    urllib.request.urlretrieve(url, file_name)
    with zipfile.ZipFile(file_name, 'r') as f:
        f.extractall()
    shutil.copytree('mmdetection-master/configs', configs_dir)
    shutil.rmtree('mmdetection-master')
    os.remove(file_name)
