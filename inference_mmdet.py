from mmdet.apis import inference_detector, init_detector

checkpoint = 'work_dir/latest.pth'
config = 'configs/cascade_rcnn/new.py'
img = 'test.jpg'
score_thr = 0.5
out_file = 'result.jpg'
device = 'cuda:0'
# device = 'cpu

model = init_detector(config, checkpoint, device=device)

result = inference_detector(model, img)

model.show_result(img, result, score_thr=score_thr, out_file=out_file)
