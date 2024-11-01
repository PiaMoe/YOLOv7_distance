# Script to run Inference with Pytorch framework without relying on YOLOv7-dataloader

import cv2
import torch
import os
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.datasets import create_dataloader
import numpy as np

# Set paths
test_images_dir = "/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/test/images"
output_dir = "/home/marten/Uni/Semester_4/src/YOLOv7-DL23/testrun_pytorch"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load YOLOv7 model
model = attempt_load('/home/marten/Uni/Semester_4/src/YOLOv7-DL23/best.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Define image size (same as training, e.g., 640x640)
img_size = 1024     # for 1024 res, letterbox uses 608x1056 resolution

# Loop through each image in the test set
for image_name in os.listdir(test_images_dir):
    # Load image
    image_path = os.path.join(test_images_dir, image_name)
    img = cv2.imread(image_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        print(f"Could not read {image_path}")
        continue

    # Preprocess image for YOLO
    img_resized = letterbox(img, new_shape=(1024, 1024), auto=False, scaleup=False, stride=32)[0]  # Resize with padding
    img_plotting = img_resized
    img_resized = img_resized[:,:,::-1].transpose(2, 0, 1).copy()  # BGR to RGB, shape to [3, height, width]
    img_resized = torch.from_numpy(img_resized).float() / 255.0
    img_resized = img_resized.unsqueeze(0).to('cuda')  # Add batch dimension
    # Run inference
    with torch.no_grad():
        pred, train_out = model(img_resized)  # Model predictions

    # Apply non-max suppression to filter detections
    conf_thresh = 0.001
    iou_thresh = 0.65
    detections = non_max_suppression(pred, conf_thresh, iou_thresh)

    # Process detections and save image with bounding boxes
    for det in detections:  # Detections per image
        if det is not None and len(det):
            # Rescale boxes to original image
            #det[:, :4] = scale_coords(img_resized.shape[2:], det[:, :4], img.shape).round()
            # Draw bounding boxes on the image
            for *xyxy, conf, cls, dist in det:
                label = f'{conf:.2f}, {int(dist)}'
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                # only display BB if conv > 0.25 -> As in Yolov7 testscript implementation
                if conf > 0.25:
                    cv2.rectangle(img_plotting, c1, c2, (255, 0, 0), 2)
                    cv2.putText(img_plotting, label, (c1[0], c1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the frame to the output directory
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, img_plotting)
    print(f"Saved {output_path}")



# import argparse
# import json
# import os
# from pathlib import Path
# from threading import Thread

# import numpy as np
# import torch
# import yaml
# from tqdm import tqdm
# from collections import defaultdict
# from models.experimental import attempt_load
# from utils.datasets import create_dataloader
# from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
#     box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
# from utils.metrics import ap_per_class, ConfusionMatrix
# from utils.plots import plot_images, output_to_target, plot_study_txt
# from utils.torch_utils import select_device, time_synchronized, TracedModel

# def test(data,
#          weights=None,
#          batch_size=32,
#          imgsz=640,
#          conf_thres=0.001,
#          iou_thres=0.6,  # for NMS
#          save_json=False,
#          single_cls=False,
#          augment=False,
#          verbose=False,
#          model=None,
#          dataloader=None,
#          save_dir=Path(''),  # for saving images
#          save_txt=False,  # for auto-labelling
#          save_hybrid=False,  # for hybrid auto-labelling
#          save_conf=False,  # save auto-label confidences
#          plots=True,
#          wandb_logger=None,
#          compute_loss=None,
#          half_precision=True,
#          trace=False,
#          is_coco=False,
#          v5_metric=False,
#          hyp=None):
#     # Initialize/load model and set device
#     device = select_device(opt.device, batch_size=batch_size)

#     # Directories
#     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#     # Load model
#     model = attempt_load(weights, map_location=device)  # load FP32 model

#     # Configure
#     model.eval()
#     if isinstance(data, str):
#         with open(data) as f:
#             data = yaml.load(f, Loader=yaml.SafeLoader)
#     check_dataset(data)  # check
#     nc = 1 if single_cls else int(data['nc'])  # number of classes
#     iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
#     niou = iouv.numel()

#     names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

#     # Dataloader
#     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#     task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
#     print("OPT", opt)
#     dataloader = create_dataloader(data[task], imgsz, batch_size, 32, opt, pad=0.5, rect=True,
#                                     prefix=task, traintestval='test')[0]


#     s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
#     for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
#         img = img.to(device, non_blocking=True)
#         img = img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         targets = targets.to(device)
#         nb, _, height, width = img.shape  # batch size, channels, height, width


#     #--------------------
#     # test_images_dir = "/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/test/images"
#     # import cv2
#     # from utils.datasets import letterbox
#     # for batch_i, image_name in enumerate(os.listdir(test_images_dir)):
#     #     # Load image
#     #     image_path = os.path.join(test_images_dir, image_name)
#     #     img = cv2.imread(image_path)
#     #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #     if img is None:
#     #         print(f"Could not read {image_path}")
#     #         continue

#     #     # Preprocess image for YOLO
#     #     img_resized = letterbox(img, new_shape=(608, 1056), scaleup=False)[0]  # Resize with padding
#     #     img_resized = img_resized.transpose(2, 0, 1)  # BGR to RGB, shape to [3, height, width]
#     #     img_resized = torch.from_numpy(img_resized).float() / 255.0
#     #     img = img_resized.unsqueeze(0).to('cuda')  # Add batch dimension
#         #-----------

#         with torch.no_grad():
#             # Run model
#             out, train_out = model(img)  # inference and training outputs

#             # Run NMS
#             out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres)

#         # Plot images
#         if plots: #and batch_i < 20:
#             f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
#             Thread(target=plot_images, args=(img, output_to_target(out), None, f, names), daemon=True).start()

#     # Return results
#     model.float()  # for training
#     print(f"Results saved to {save_dir}{s}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(prog='test.py')
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
#     parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
#     parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
#     parser.add_argument('--task', default='val', help='train, val, test, speed or study')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--verbose', action='store_true', help='report mAP by class')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
#     parser.add_argument('--project', default='runs/test', help='save to project/name')
#     parser.add_argument('--name', default='exp', help='save to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
#     parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
#     parser.add_argument('--hyp', type=str, default='', help='hyperparameters path')
#     opt = parser.parse_args()
#     opt.save_json |= opt.data.endswith('coco.yaml')
#     opt.data = "/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/data.yaml"
#     opt.task = 'test'
#     opt.weights = "/home/marten/Uni/Semester_4/src/YOLOv7-DL23/best.pt"
#     opt.img_size = 1024
#     opt.batch_size = 1
#     opt.data = check_file(opt.data)  # check file

#     if opt.task in ('train', 'val', 'test'):  # run normally
#         test(opt.data,
#              opt.weights,
#              opt.batch_size,
#              opt.img_size,
#              opt.conf_thres,
#              opt.iou_thres,
#              opt.save_json,
#              opt.single_cls,
#              opt.augment,
#              opt.verbose,
#              save_txt=opt.save_txt | opt.save_hybrid,
#              save_hybrid=opt.save_hybrid,
#              save_conf=opt.save_conf,
#              trace=not opt.no_trace,
#              v5_metric=opt.v5_metric,
#              hyp=None
#              )