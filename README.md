# auto-ml
Just another AutoML implementation...

# MLP Classifier (MultiLayer Perceptron)
Based on [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

## Installation
### Install python 3
System dependent (see https://www.python.org/downloads/)

### Install python packages with built-in package manager pip
```
pip install -r requirements.txt
```

## Run
### Train
```
python train.py -i train_input.json
```
```"X"``` - samples to train with shape (n_samples, n_features)

```"y"``` - samples classes with shape (n_samples)

Script generates ```"output_path"``` from train_input.json

Script dumps model to ```"model_path"``` from train_input.json

### Predict
```
python predict.py -i predict_input.json
```
Script loads model from ```"model_path"``` from predict_input.json

Script generates ```"output_path"``` from predict_input.json

Field ```"y_pred"``` in "output_path" is predicted classes

## Docker

### Build 
```
docker build -t auto-ml-app .
```

### Train 
```bash
docker run -it \
           -v "$(pwd)"/output:/usr/src/output \
           -w /usr/src/output \
           -v "$(pwd)"/train_input.json:/usr/src/input/train_input.json \
           auto-ml-app python /usr/src/app/train.py -i /usr/src/input/train_input.json
```

### Predict 
```bash
docker run -it \
           -v "$(pwd)"/output:/usr/src/output \
           -w /usr/src/output \
           -v "$(pwd)"/predict_input.json:/usr/src/input/predict_input.json \
           -v "$(pwd)"/output/model.joblib:/usr/src/input/model.joblib \
           auto-ml-app python /usr/src/app/predict.py -i /usr/src/input/predict_input.json
```

# Computer Vision
Based on

* [MMCV](https://github.com/open-mmlab/mmcv)
* [MMDetection](https://github.com/open-mmlab/mmdetection)

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
System dependent (see https://www.python.org/downloads/)

### Install python packages with built-in package manager pip
```
pip install -r requirements_mmdet.txt
```

## Train
### Prepare data in `data` directory (Example)
```
python get_mmdet_configs.py 
python make_coco.py -i make_coco_input.json
```

### Run
```
python train_mmdet.py -i train_mmdet_input.json
```
Results will be in `data/works` directory

## Inference
### Prepare data in `data` directory (Example, also requires config and checkpoint file from trained model)
```
cp test.jpg data/test.jpg
```

### Run
```
python inference_mmdet.py -i inference_mmdet_input.json
```
Results will be in `data` directory (`result.jpg` by default)

## Docker
Based on [MMDetection docker](https://github.com/open-mmlab/mmdetection/tree/master/docker)

### Prerequisites
* [CUDA Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) or [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (CUDA Driver included)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

[What is the difference between CUDA, CUDNN, CUDA Driver, CUDA Toolkit, and NCVV?](https://www.programmersought.com/article/57794836777/)

### Build
#### Change Dockerfile_mmdet ARGs according to host lib versions (see [Dockerfile_mmdet](https://github.com/romanzes637/auto-ml/blob/main/Dockerfile_mmdet))
```dockerfile
ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"
ARG CUDA_MMCV="111"
ARG MMCV="1.3.5"
ARG MMDET="2.13.0"
```
#### Build image
```bash
docker build -t auto-ml-mmdet -f Dockerfile_mmdet . 
```

### Train
#### Prepare data in `data` directory (Example)
```
python get_mmdet_configs.py 
python make_coco.py -i make_coco_input.json
```
#### Data structure
* `dataset` - set of images with annotations (labels, bboxes, polygons, etc)
* `model` - torch or caffe neural net model (tensorflow or keras in the future?)
* `config` - text file with neural net structure, train params, dataset description 
* `checkpoint` - binary or text file with neural net weights
* `work` - directory with train logs, metrics and intermediate checkpoints
```
data
|__ datasets
    |__ dataset1
        |__ images
        |__ annotation
        |__ ... (depends on)
    |__ dataset2
    |__ dataset3
    ...
|__ configs
    |__ model1
        |__ config1
        |__ config2
        |__ config3
        ...
    |__ model2
    |__ model3
    ...   
|__ checkpoints
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
|__ works
    |__ work1
        |__ log
        |__ checkpoint1
        |__ checkpoint2
        |__ checkpoint3
        ...
        |__ checkpoint_latest
    |__ work2
    |__ work3
    ...
```

#### Run
```bash
docker run --gpus all --shm-size 8G \
           -v "$(pwd)"/data:/auto-ml/data \
           -v "$(pwd)"/train_mmdet_input.json:/auto-ml/train_mmdet_input.json \
           auto-ml-mmdet python train_mmdet.py
```
* `--gpus all` - GPU devices to add to the container ('all' to pass all GPUs)
* `--shm-size 8G` - Size of /dev/shm. The format is <number><unit>. number must be greater than 0. Unit is optional and can be b (bytes), k (kilobytes), m (megabytes), or g (gigabytes). If you omit the unit, the system uses bytes. If you omit the size entirely, the system uses 64m.
* `-v "$(pwd)"/data:/auto-ml/data` - mount host data dir `"$(pwd)"/data` to container data dir `/auto-ml/data`
* `-v "$(pwd)"/train_mmdet_input.json:/auto-ml/train_mmdet_input.json` - mount host train config `"$(pwd)"/train_mmdet_input.json` to container train config `/auto-ml/train_mmdet_input.json`
* `python train_mmdet.py` - run train script
   
Results will be in `data/works` directory

### Inference
#### Prepare data in `data` directory (Example, also requires config and checkpoint file from trained model)
```
cp test.jpg data/test.jpg
```

#### Run
```bash
docker run --gpus all --shm-size 8G \
           -v "$(pwd)"/data:/auto-ml/data \
           -v "$(pwd)"/inference_mmdet_input.json:/auto-ml/inference_mmdet_input.json \
           auto-ml-mmdet python inference_mmdet.py
```
* `--gpus all` - GPU devices to add to the container ('all' to pass all GPUs)
* `--shm-size 8G` - Size of /dev/shm. The format is <number><unit>. number must be greater than 0. Unit is optional and can be b (bytes), k (kilobytes), m (megabytes), or g (gigabytes). If you omit the unit, the system uses bytes. If you omit the size entirely, the system uses 64m.
* `-v "$(pwd)"/data:/auto-ml/data` - mount host data dir `"$(pwd)"/data` to container data dir `/auto-ml/data`
* `-v "$(pwd)"/inference_mmdet_input.json:/auto-ml/inference_mmdet_input.json` - mount host inference config `"$(pwd)"/inference_mmdet_input.json` to container inference config `/auto-ml/inference_mmdet_input.json auto-ml-mmdet`
* `python inference_mmdet.py` - run inference script
   
Results will be in `data` directory (`result.jpg` by default)
