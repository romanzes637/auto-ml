import json
import argparse
from pprint import pprint

from mmdet.apis import inference_detector, init_detector


if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='inference_mmdet_input.json')
    parser.add_argument('-c', '--config', help='configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco_new.py')
    parser.add_argument('-k', '--checkpoint', help='work/latest.pth')
    parser.add_argument('-m', '--img', help='test.jpg')
    parser.add_argument('-s', '--score_thr', type=float, help='0.5')
    parser.add_argument('-o', '--output', help='result.jpg')
    parser.add_argument('-O', '--output_json', help='result.json')
    parser.add_argument('-d', '--device', help='cpu or cuda:0')
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
    checkpoint = cfg_args['checkpoint']
    config = cfg_args['config']
    img = cfg_args['img']
    score_thr = cfg_args['score_thr']
    output = cfg_args['output']
    output_json = cfg_args['output_json']
    device = cfg_args['device']

    print('Initialization')
    model = init_detector(config, checkpoint, device=device)
    print('Inference')
    result = inference_detector(model, img)
    print(f'Writing result to {output}')
    model.show_result(img, result, score_thr=score_thr, out_file=output)
    
    print(f'Writing result to output json {output_json}')
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: x.tolist())
