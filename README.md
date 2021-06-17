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



