import os

os.system('git clone https://github.com/open-mmlab/mmdetection.git')
os.system('cp -r mmdetection/configs .')
os.system('rm -r mmdetection')
