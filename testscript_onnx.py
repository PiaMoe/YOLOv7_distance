# Script to evalueate ONNX Model with YOLOv7 Object Detection & Distance Metrics
# Similar to test.py script that uses pytorch framework

import argparse
import json
import os
from pathlib import Path
from threading import Thread
import onnxruntime as ort
import onnx
from onnx_opcounter import calculate_params
import numpy as np
import torch
import yaml
from tqdm import tqdm
from collections import defaultdict
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel


def create_distance_bins(max_distance, number_bins):
    # Calculate the width of each bin
    bin_width = max_distance / number_bins

    # Create the bins
    distance_bins = [(i * bin_width, (i + 1) * bin_width) for i in range(number_bins)]

    return distance_bins

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False, 
         v5_metric=False,
         hyp=None,
         max_params=50e6):
    # Initialize/load model and set device

    set_logging()
    device = select_device(opt.device, batch_size=batch_size)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load ONNX model
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(weights, providers=providers)
    gs = 32  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    # compute number of parameters for onnx model and compare to max threshold
    model = onnx.load_model(weights)
    try:
        params = calculate_params(model)
    except:
        raise ValueError("Could not compute number of parameters for onnx model.")
    print(f"Amount of model params: {params}")
    assert params <= max_params, f"Amount of model parameters ({params}) exceeds maximum threshold of {max_params}"

    # get data
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes

    # statistics
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images

    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=task, traintestval='test', onnx=True)[0]
                                    # prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    coco91class = coco80_to_coco91_class()
    names = {0: "boat"}
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    distance_errors = []    
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # convert image to numpy array for ONNX
        img_onnx = img.cpu().numpy()

        # Run model
        inputs = {session.get_inputs()[0].name: img_onnx}
        outname = [i.name for i in session.get_outputs()]

        t = time_synchronized()
        out = session.run(outname, inputs)[0]  # Get predictions for batch or single image
        out = torch.from_numpy(out)
        t0 += time_synchronized() - t

        # Run NMS
        targets[:, 2:-1] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_synchronized()
        out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
        t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            tdist = labels[:, -1].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    # Append statistics (correct, conf, pcls, tcls, pdist, tdist)
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls, torch.Tensor(), tdist))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls, dist in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf, dist) if save_conf else (cls, *xywh, dist)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "distance": dist,
                                 "domain": "pixel"} for *xyxy, conf, cls, dist in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'distance': p[-1],
                                  'score': round(p[4], 5)},)

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                tdist_tensor = labels[:, -1]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn[:,:-1], torch.cat((labels[:, 0:1], tbox), 1))
                distance_errors_per_cat = {}
                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                    distance_errors_per_cat[int(cls)] = []
                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                #distances
                                pred_dist = pred[pi[j], -1]
                                target_dist = labels[d, -1]
                                pred_conf = pred[pi[j], 4]
                                distance_error = abs(pred_dist - target_dist) # relative dist error
                                distance_conf_and_error_and_gt = [float(pred_conf), float(distance_error), float(target_dist), float(pred_dist)]
                                distance_errors_per_cat[int(cls)].append(distance_conf_and_error_and_gt)
                                if len(detected) == nl:  # all targets already located in image
                                    break
                distance_errors.append(distance_errors_per_cat)
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls, pred[:,-1].cpu(), tdist))

        # Plot images
        if plots and batch_i < 10:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            plot_images(img, targets, paths, f, names)
            #Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            plot_images(img, output_to_target(out), paths, f, names)
            #Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    if hyp is not None:
        distance_bins = create_distance_bins(hyp["max_distance"], 5)
    else:
        distance_bins = create_distance_bins(1000, 5)   # use default dist of 1000 m if no hyperparameters passed
    # print(distance_bins)
    # distance_bins = [(0, 50), (50, 100), (100, 150), (200, 250), (250, 300), (300, 500), (500, 700), (700, 1000)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class

        #do it for individual distances as well:
        # stats_for_bins = {}
        # for bin_min, bin_max in distance_bins:
        #     #stats[-1] should be target distances
        #     boolean_array = (stats[-1] >= bin_min) & (stats[-1] < bin_max)
        #     bin_key = (bin_min, bin_max)
        #     for k, nparray in enumerate(stats):
        #         print(k, nparray)
        #         p_d, r_d, ap_d, f1_d, ap_class_d = ap_per_class(nparray[boolean_array])
        #     stats_for_bins[bin_key] = [p_d, r_d, ap_d, f1_d, ap_class_d]
        #     print(bin_key, stats_for_bins[bin_key])

    else:
        nt = torch.zeros(1)

    # print(distance_errors)



    # Initialize dictionaries to store accumulated weighted errors and total confidences
    mean_dist_err_boat_bins = defaultdict(float)
    abs_dist_err_boat_bins = defaultdict(float)
    total_conf_boat_bins = defaultdict(float)
    samples_per_bin = defaultdict(int)

    # Initialize variables to store total accumulated weighted errors and confidences
    total_mean_dist_err_boat= 0 # weighted with conf & relative
    abs_dist_err_boat = 0 # absolute dist error without conf weights
    total_conf_boat = 0
    samples = 0
    # print(distance_errors)
    for distance_err in distance_errors:
        if 0 in distance_err.keys():
            for obj_disst_pair in distance_err[0]:
                dconf, derror, gt, pred = obj_disst_pair
                total_mean_dist_err_boat += dconf * derror / gt
                abs_dist_err_boat += derror
                total_conf_boat += dconf
                samples += 1
                for bin_min, bin_max in distance_bins:
                    if bin_min <= gt < bin_max:
                        bin_key = (bin_min, bin_max)
                        mean_dist_err_boat_bins[bin_key] += dconf * derror / gt
                        abs_dist_err_boat_bins[bin_key] += derror
                        total_conf_boat_bins[bin_key] += dconf
                        samples_per_bin[bin_key] +=1
                        break

    # Calculate the weighted mean distance error for each bin
    weighted_mean_dist_err_boat_bins = {
        bin_key: mean_dist_err_boat_bins[bin_key] / total_conf_boat_bins[bin_key]
        if total_conf_boat_bins[bin_key] > 0 else -1
        for bin_key in distance_bins
    }

    # Compute mean of absolute dist error bins
    mean_abs_dist_err_boat_bins = {
        bin_key: abs_dist_err_boat_bins[bin_key] / samples_per_bin[bin_key]
        if samples_per_bin[bin_key] > 0 else -1
        for bin_key in distance_bins
    }
    
    mean_abs_dist_err_boat = abs_dist_err_boat / samples


    # Calculate the overall weighted mean distance error
    overall_weighted_mean_dist_err_boat = total_mean_dist_err_boat / total_conf_boat if total_conf_boat > 0 else -1
    metrics_bin_distances = {}

    # compute combined metric between mAP@0.5:0.95 and err_weighted_dist_rel+
    combined_metric = map * (1 - min(overall_weighted_mean_dist_err_boat, 1))

    # Print the results for each bin
    for bin_key in distance_bins:
        print(f"Distance bin {bin_key}:")
        print("  samples: ", samples_per_bin[bin_key])
        print("  weighted_reL_dist_err_boat =", weighted_mean_dist_err_boat_bins[bin_key])
        print("  abs_mean_dist_err_boat =", mean_abs_dist_err_boat_bins[bin_key])
        metrics_bin_distances["metrics/distancebins/weighted_rel_dist_err_boat_"+str(bin_key)] = weighted_mean_dist_err_boat_bins[bin_key]
        metrics_bin_distances["metrics/distancebins/abs_mean_dist_err_boat_"+str(bin_key)] = mean_abs_dist_err_boat_bins[bin_key]
    if not wandb_logger is None:
        wandb_logger.log(metrics_bin_distances)
    # Print the overall results
    print("Total Samples: ", samples)
    print("Overall weighted_rel_dist_err_boat =", overall_weighted_mean_dist_err_boat)
    print("Overall abs_mean_dist_err_boat =", mean_abs_dist_err_boat)
    print("Combined Metric = ", combined_metric)


    metrics_overall_distance = {}
    metrics_overall_distance["metrics/weighted_rel_dist_err_boat"] = overall_weighted_mean_dist_err_boat
    metrics_overall_distance["metrics/abs_mean_dist_err_boat"] = mean_abs_dist_err_boat
    metrics_overall_distance["metrics/combined_metric"] = combined_metric 
    if not wandb_logger is None:
        wandb_logger.log(metrics_overall_distance)


    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or nc < 50) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='test', help='train, val, test')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path')
    parser.add_argument('--max-params', type=int, default=50e6, help='maximum amount of allowed parameters')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')

    ### for debugging
    #opt.data = "/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/data.yaml"
    #opt.weights = "/home/marten/Uni/Semester_4/src/YOLOv7-DL23/best.onnx"
    #opt.save_json = False
    #print(opt)

    opt.data = check_file(opt.data)  # check file
 

    # load hyperparameters (for distance statistics)
    hyp = None
    if opt.hyp != '':   # if hyp param file was passed
        try:
            with open(opt.hyp) as f:
                hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        except FileNotFoundError:
            print("Path to hyperparameter file " + str(opt.hyp) + " does not exist.")
            raise FileNotFoundError
    else:
        print("No Hyperparameter file passed to test script")
        print("Using default max dist of 1000 to compute dist bins")
        print("Use the --hyp argument to provide path to the file")

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             hyp=hyp,
             max_params=int(opt.max_params)
             )
