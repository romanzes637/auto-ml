from pathlib import Path
import urllib.request
from mmcv import Config
from mmdet.apis import set_random_seed

# Args
dataset_type = 'CocoDataset'
dataset_root = 'dataset_coco'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'
train_file = 'train_coco.json'
val_file = 'val_coco.json'
test_file = 'test_coco.json'
img_dir = 'images'
root = '.'
checkpoints_dir = 'checkpoints'
configs_dir = 'configs'
is_pretrained = True
checkpoint_file = 'cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
checkpoint_url = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
config_file = 'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
new_config_file = f'cascade_rcnn/new.py'
work_dir = 'work_dir'
seed = 42

# Init dataset
dataset_path = Path(dataset_root)
train_path = dataset_path / train_dir
val_path = dataset_path / val_dir
test_path = dataset_path / test_dir
train_img_path = train_path / img_dir
val_img_path = val_path / img_dir
test_img_path = test_path / img_dir
train_file_path = train_path / train_file
val_file_path = val_path / val_file
test_file_path = test_path / test_file
assert dataset_path.exists()
assert train_path.exists()
assert val_path.exists()
assert test_path.exists()
assert train_img_path.exists()
assert val_img_path.exists()
assert test_file_path.exists()
assert train_file_path.exists()
assert val_file_path.exists()
assert test_file_path.exists()

# Get classes
def get_classes(coco_file):
  with open(coco_file) as f:
    data = json.load(f)
  classes = [x['name'] for x in data['categories']]
  return classes
train_classes = get_classes(train_file_path)
val_clases = get_classes(val_file_path)
test_classes = get_classes(test_file_path)
assert set(train_classes) == set(train_classes) == set(test_classes)
classes = train_classes
print(classes)

# Config, checkpoint, work_dir
root_path = Path(root)
root_path.mkdir(parents=True, exist_ok=True)
checkpoints_path = root_path / checkpoints_dir
checkpoints_path.mkdir(parents=True, exist_ok=True)
configs_path = root_path / configs_dir
configs_path.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoints_path / checkpoint_file
config_path = configs_path / config_file
assert config_path.exists()
new_config_path = configs_path / new_config_file
work_path = root_path / work_dir
work_path.mkdir(parents=True, exist_ok=True)
if is_pretrained and not checkpoint_path.exists():
  urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
 
# Cascade Mask R-CNN https://paperswithcode.com/method/cascade-mask-r-cnn
cfg = Config.fromfile(str(config_path.resolve()))
print(f'Config:\n{cfg.pretty_text}')  

# Update config
# https://mmdetection.readthedocs.io/en/v2.0.0/config.html
# 1. dataset settings
cfg.dataset_type = dataset_type
cfg.data_root = str(dataset_path.resolve())
cfg.data.train.type = dataset_type
cfg.data.train.classes = classes
cfg.data.train.data_root = str(train_path.resolve())
cfg.data.train.ann_file = str(train_file_path.resolve())
cfg.data.train.img_prefix = img_dir
cfg.data.val.type = dataset_type
cfg.data.val.classes = classes
cfg.data.val.data_root = str(val_path.resolve())
cfg.data.val.ann_file = str(val_file_path.resolve())
cfg.data.val.img_prefix = img_dir
cfg.data.test.type = dataset_type
cfg.data.test.classes = classes
cfg.data.test.data_root = str(test_path.resolve())
cfg.data.test.ann_file = str(test_file_path.resolve())
cfg.data.test.img_prefix = img_dir
# 2. model settings
cfg.model.roi_head.bbox_head[0].num_classes = len(classes)
cfg.model.roi_head.bbox_head[1].num_classes = len(classes)
cfg.model.roi_head.bbox_head[2].num_classes = len(classes)
cfg.model.roi_head.mask_head.num_classes = len(classes)
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
if is_pretrained:
  cfg.load_from = str(checkpoint_path.resolve())
# # Set up working dir to save files and logs.
cfg.work_dir = str(work_path.resolve())
# https://mmdetection.readthedocs.io/en/latest/tutorials/customize_runtime.html
# # The original learning rate (LR) is set for 8-GPU training.
# # We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.policy = 'cyclic'  # described in https://arxiv.org/pdf/1506.01186.pdf
cfg.lr_config.warmup = None
cfg.log_config.interval = 10
# # Change the evaluation metric since we use customized dataset.
# cfg.evaluation.metric = 'mAP'
# # We can set the evaluation interval to reduce the evaluation times
# cfg.evaluation.interval = 12
# # We can set the checkpoint saving interval to reduce the storage cost
# cfg.checkpoint_config.interval = 12
# # Set seed thus the results are more reproducible
cfg.seed = seed
set_random_seed(seed, deterministic=False)
cfg.gpu_ids = range(1)

# Dump new config
cfg.dump(new_config_path.resolve())
cfg = Config.fromfile(str(new_config_path.resolve()))  # Check
print(f'Config:\n{cfg.pretty_text}')

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = classes

train_detector(model, datasets, cfg, distributed=False, validate=True)
