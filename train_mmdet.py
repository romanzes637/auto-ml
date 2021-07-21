from pathlib import Path
import urllib.request
import argparse
from pprint import pprint
import json

from mmcv import Config
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

# Get classes
def get_classes(coco_file):
    with open(coco_file) as f:
        data = json.load(f)
    classes = [x['name'] for x in data['categories']]
    return classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='train_mmdet_input.json')
    parser.add_argument('-D', '--dataset_dir', help='dataset_coco')
    parser.add_argument('-y', '--dataset_type', help='CocoDataset')
    parser.add_argument('-t', '--train_file', help='train_coco.json')
    parser.add_argument('-v', '--val_file', help='val_coco.json')
    parser.add_argument('-s', '--test_file', help='test_coco.json')
    parser.add_argument('-T', '--train_dir', help='train')
    parser.add_argument('-V', '--val_dir', help='val')
    parser.add_argument('-S', '--test_dir', help='test')
    parser.add_argument('-I', '--img_dir', help='images')
    parser.add_argument('-W', '--work_dir', help='work')
    parser.add_argument('-C', '--configs_dir', help='configs')
    parser.add_argument('-K', '--checkpoints_dir', help='checkpoints')
    parser.add_argument('-p', '--is_pretrained', type=bool, default=None)
    parser.add_argument('-c', '--config_file',
                        help='cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py')
    parser.add_argument('-n', '--new_config_file',
                        help='cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco_new.py')
    parser.add_argument('-k', '--checkpoint_file',
                        help='cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py')
    parser.add_argument('-u', '--checkpoint_url',
                        help='https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth')
    parser.add_argument('-e', '--seed', type=int, help='42')
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
    dataset_type = cfg_args['dataset_type']
    dataset_dir = cfg_args['dataset_dir']
    train_file = cfg_args['train_file']
    val_file = cfg_args['val_file']
    test_file = cfg_args['test_file']
    train_dir = cfg_args['train_dir']
    val_dir = cfg_args['val_dir']
    test_dir = cfg_args['test_dir']
    img_dir = cfg_args['img_dir']
    configs_dir = cfg_args['configs_dir']
    checkpoints_dir = cfg_args['checkpoints_dir']
    work_dir = cfg_args['work_dir']
    config_file = cfg_args['config_file']
    new_config_file = cfg_args['new_config_file']
    is_pretrained = cfg_args['is_pretrained']
    checkpoint_file = cfg_args['checkpoint_file']
    checkpoint_url = cfg_args['checkpoint_url']
    seed = cfg_args['seed']

    # Check dataset
    dataset_dir = Path(dataset_dir)
    train_path = dataset_dir / train_dir
    val_path = dataset_dir / val_dir
    test_path = dataset_dir / test_dir
    train_img_path = train_path / img_dir
    val_img_path = val_path / img_dir
    test_img_path = test_path / img_dir
    train_file_path = train_path / train_file
    val_file_path = val_path / val_file
    test_file_path = test_path / test_file
    assert dataset_dir.exists()
    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()
    assert train_img_path.exists()
    assert val_img_path.exists()
    assert test_file_path.exists()
    assert train_file_path.exists()
    assert val_file_path.exists()
    assert test_file_path.exists()

    # Get dataset classes
    train_classes = get_classes(train_file_path)
    val_classes = get_classes(val_file_path)
    test_classes = get_classes(test_file_path)
    assert set(train_classes) == set(val_classes) == set(test_classes)
    classes = train_classes
    print('Dataset classes')
    print(classes)

    # Prepare environment
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    checkpoints_path = Path(checkpoints_dir)
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    configs_path = Path(configs_dir)
    configs_path.mkdir(parents=True, exist_ok=True)
    config_path = configs_path / config_file
    assert config_path.exists()
    new_config_path = configs_path / new_config_file
    new_config_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_path / checkpoint_file
    # Check checkpoint (if pretrained)
    if is_pretrained and not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading checkpoint from {checkpoint_url} to {checkpoint_path.resolve()}')
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)

    # Get base config
    # Cascade Mask R-CNN https://paperswithcode.com/method/cascade-mask-r-cnn
    cfg = Config.fromfile(str(config_path.resolve()))
    print(f'Config:\n{cfg.pretty_text}')

    # Update config
    # https://mmdetection.readthedocs.io/en/v2.0.0/config.html
    # 1. dataset settings
    cfg.dataset_type = dataset_type
    cfg.data_root = str(dataset_dir.resolve())
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
    cfg.lr_config.policy = 'step'  # described in https://arxiv.org/pdf/1506.01186.pdf
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

    # Save new config
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
