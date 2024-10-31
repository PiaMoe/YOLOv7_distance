# Inference for ONNX model
import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision

# Load the ONNX model
onnx_model_path = '/home/marten/Uni/Semester_4/src/YOLOv7-DL23/best.onnx'  # Replace with your ONNX model path
session = ort.InferenceSession(onnx_model_path)

# Define the input image size and the model input shape
input_size = (1024, 1024)  # Adjust this based on your model's expected input size

# Function to preprocess the image
def preprocess_image(image_path, input_size):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    img = cv2.resize(img, input_size)
    img = img.transpose(2, 0, 1)  # Change to CHW format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return img, (h, w)  # Return original height and width for scaling later

# Function to post-process the outputs
def postprocess_predictions(preds, original_shape, conf_threshold=0.35, iou_threshold = 0.45):
    h, w = original_shape

    output_tensor = torch.from_numpy(preds)
    # scale BB preds to img size  
    output_tensor[..., 0] = output_tensor[..., 0] / input_size[0] * w     # new x
    output_tensor[..., 2] = output_tensor[..., 2] / input_size[0] * w     # new width
    output_tensor[..., 1] = output_tensor[..., 1] / input_size[1] * h     # new y
    output_tensor[..., 3] = output_tensor[..., 3] / input_size[1] * h     # new height
    output_tensor = output_tensor.unsqueeze(0)

    nms_results = non_max_suppression(output_tensor, conf_thres=conf_threshold, iou_thres=iou_threshold)

    filtered_outputs = [output.squeeze(0).numpy() for output in nms_results if output.size(0) > 0] # only return anchor tensors that still remain

    return filtered_outputs  # list of BB predictions including dist for each image (n,7) - [xyxy, conf, cls, dist]

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,7) tensor per image [xyxy, conf, cls, dist]
    """

    nc = 1  # number of classes -> only one
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 7), device=prediction.device)] * prediction.shape[0]  # output tensor has shape [n, [xyxy,conf,cls,dist]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # use confidence mask to extract anchors that exceed conf threshold

        # Cat apriori labels if autolabelling -> we dont pass labels
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 6), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:-1] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate -> set conf = objectness
        else:
            x[:, 5:-1] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls, distance)
        if multi_label:
            i, j = (x[:, 5:-1] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), x[i, -1, None]), 1)
        else:  # best class only
            conf, j = x[:, 5:-1].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), x[:,-1].unsqueeze(1)), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

# Run inference on an image
def run_inference(image_path):
    img, original_shape = preprocess_image(image_path, input_size)

    # Prepare the input for the ONNX model
    inputs = {session.get_inputs()[0].name: img}
    

    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    # Run inference
    batch_preds = session.run(outname, inputs)[0]  # Get predictions for batch or single image

    # Post-process predictions
    results = []
    for i, pred in enumerate(batch_preds):  # i is image / batch index, pred is prediction tensor for each image
        nms_results = postprocess_predictions(pred, original_shape, conf_threshold=0.25, iou_threshold=0.45)
        results.append(nms_results)    # create result entry for each image i

    return results

# Example usage
if __name__ == "__main__":
    image_path = '/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/val/images/004781.png'  # Replace with your image path
    visualize = True

    # run ingerence on images with onnx model
    results = run_inference(image_path)
    print(results)
    # Optionally, visualize the results
    if visualize:
        for i, result in enumerate(results):  # iterate over images
            img = cv2.imread(image_path)
            #img = cv2.resize(img, input_size)
            for BB in result:
                cv2.rectangle(img, (int(BB[0]), int(BB[1])), (int(BB[2]), int(BB[3])), (255, 0, 0), 2)
                cv2.putText(img, f"Class: nav_buoy, Score: {BB[4]:.2f}, Dist: {BB[-1]:.3f}", (int(BB[0]), int(BB[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Predictions", img)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()