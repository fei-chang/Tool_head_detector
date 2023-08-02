import os
from pathlib import Path

import torch

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync

# Adopted from YOLOv3 🚀 by Ultralytics, GPL-3.0 license
# Modified for head detection usage in gaze following

def load_model(pretrained_weights: str,
               imgsz=(640,640),  # inference size (pixels)
               device='cuda:0',
               ):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(pretrained_weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # warmup
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  
    
    return model, device, imgsz

def run_single_folder(model, 
        device, 
        dataset,
        txt_name,  # save results to an txt file in the output folder
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=3,  # maximum detections per image
        print_every=500 # print info
        ):
    
    t3 = time_sync()
    counter = 0
    total = len(dataset)
    for path, im, im0s, vid_cap, _ in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = model(im, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            txt_path = str(txt_name) # im.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xy_normalized = tuple((torch.tensor(xyxy).view(1, 4) / gn).view(-1))  # normalized xywh
                    with open(txt_path, 'a') as f:
                        f.write('%s'%p.stem) # number of frame
                        line = (cls, *xy_normalized)
                        f.write((', %g' * len(line))% line + '\n')
        counter+=1
        if counter%print_every==0:
            LOGGER.info(f'Finished Processing {counter:d}/{total:d} ({time_sync()-t3:.3f}s)')



if __name__ == "__main__":
    # Load model
    model_weights = '/home/changfei/Tool_head_detector/best.pt'
    model, device, imgsz = load_model(model_weights)

    ################################################################################
    # Modify code here for different folder stucture
    # ################################################################################
    input_img_folder = '/home/changfei/X_Nas/ShanghaiASD/20230531/frames/小鸟/DD96-RJA-小鸟'
    output_txt_file = '/home/changfei/X_Nas/ShanghaiASD/20230531/frames/小鸟/DD96-RJA-小鸟/annotations/raw_detections.txt'

    dataset = LoadImages(input_img_folder, img_size=imgsz, stride=model.stride, auto=model.pt and not model.jit)
    run_single_folder(model, device, dataset, output_txt_file)
    LOGGER.info('Done: %s'%input_img_folder)