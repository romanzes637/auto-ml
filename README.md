# auto-ml
Just another AutoML implementation...

# Installation
## Install python 3
System dependent, see https://www.python.org/downloads/

## Install python packages with built-in package manager pip
```
pip install -r requirements.txt
```

# Run
## Train
```
python train.py -i train_input.json
```
```"X"``` - samples to train with shape (n_samples, n_features)

```"y"``` - samples classes with shape (n_samples)

Script generates ```"output_path"``` from train_input.json

Script dumps model to ```"model_path"``` from train_input.json

## Predict
```
python predict.py -i predict_input.json
```
Script loads model from ```"model_path"``` from predict_input.json

Script generates ```"output_path"``` from predict_input.json

Field ```"y_pred"``` in "output_path" is predicted classes

# Docker

## Build 

docker build -t auto-ml-app .

## Train 

docker run -it \
-v "$(pwd)"/output:/usr/src/output \
-w /usr/src/output \
-v "$(pwd)"/train_input.json:/usr/src/input/train_input.json \
auto-ml-app python /usr/src/app/train.py -i /usr/src/input/train_input.json

## Predict 

docker run -it \
-v "$(pwd)"/output:/usr/src/output \
-w /usr/src/output \
-v "$(pwd)"/predict_input.json:/usr/src/input/predict_input.json \
-v "$(pwd)"/output/model.joblib:/usr/src/input/model.joblib \
auto-ml-app python /usr/src/app/predict.py -i /usr/src/input/predict_input.json

# MMDET
## Data structure
* `dataset` - set of images with annotations (labels, bboxes, polygons, etc)
* `model` - torch or caffe neural net model (tensorflow or keras in the future?)
* `config` - file with neural net structure, train params, dataset description 
* `checkpoint` - file with neural net weights
```
dataset(s)
|__ images
|__ annotation
|__ ... (depends on)

configs
|__ model1
   |__ config
   |__ config2
   |__ config3
   ...
|__ model2
|__ model3
   ...
   
checkpoints
|__ model1
   |__ submodel1
      |__ checkpoint1
      |__ checkpoint2
      |__ checkpoint3
      ...
   |__ submodel2
   |__ submodel3
   ...
|__ model2
|__ model3
   ...

work(s)
|__ log
|__ checkpoint1
|__ checkpoint2
|__ checkpoint3
   ...
|__ checkpoint_last
```

## Installation
### Prerequisites
* Linux or macOS (Windows is in experimental support)
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
* GCC 5+
* MMCV

See https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md

### Install python 3
System dependent, see https://www.python.org/downloads/

### Install python packages with built-in package manager pip
```
pip install -r requirements_mmdet.txt
```
### Get MMDET configs
```
python get_mmdet_configs.py
```

## Run
### Train
```
python make_coco.py -i make_coco_input.json (once per dataset)

python train_mmdet.py -i train_mmdet_input.json
```

### Inference
```
python inference_mmdet.py -i inference_mmdet_input.json
```

## Docker
See https://github.com/open-mmlab/mmdetection/tree/master/docker
