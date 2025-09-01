# YOLOv7 Object Detection with second-stage network for Distance and Heading Estimation

The following instructions are for training the second-stage network located in `/secondStageModel`. This model takes cropped images of detected objects as input and predicts their distance and heading angle.

Prerequisite:
Train a YOLOv7 model for object detection (e.g., on boats) as described in the [main repository](https://github.com/PiaMoe/YOLOv7_distance/tree/main).

## Dataset
set the path to the dataset manually in `/secondStageModel/train.py`

For each image in the dataset, there should be a corresponding label file located in a parallel labels/ directory. Each label file must follow this format:
```
class x y w h distance cos sin
```

Here:
- `class` is the class index (e.g., 0 for 'boat', 1 for 'buoy'),
- `x, y, w, h` are the **normalized** coordinates of the bounding box (center x/y, width, height),
- `distance` is the estimated distance to the object **in meters**.
- `cos` is the cosine of the estimated heading angle of the object.
- `sin` is the sine of the estimated heading angle of the object.


## Training of Crop Regressor

Run `/secondStageModel/train.py`. First, the target image crops will be generated from the dataset. Then the Custom ResNet18 or MobileNetV2 will be trained depending on which model is selected in `train()`.


## Testing
> [!NOTE]
> :pushpin: Pretrained Weights are available [here](https://drive.google.com/drive/folders/1GujICE9Ev-ppfH4PUX19UjywqgFn-5Zf?hl=de) 

For testing and inference, a pretrained YOLOv7 model for object detection is needed. Its outputs are the inputs of the "Crop Regressor". You can evaluate the 2-stage-model performance w.r.t. object detection, distance estimation and heading estimation:

``` shell
python3 YOLOv7_distance/test.py --data 'path/to/data.yaml' --img 1024 --batch 4 --conf 0.001 --iou 0.65 --device 0 --weightsYOLO 'YOLOv7_distance/weights/bestDet.pt' --weightsRegressor 'YOLOv7_distance/secondStageModel/outputs/ResNet18/best.pth' --task 'test' --hyp 'YOLOv7_distance/data/hyp.scratch.p5.yaml'
```
Note that 2 weight files are needed, one for object detection and one for distance & heading estimation.
Make sure that the data.yaml file contains a test or val entity depending on the task argument.


Sample output for a model with pretrained weights on testsplit:

```
Distance bin (0.0, 200.0):
  samples:  877
  weighted_reL_dist_err_boat = 0.19412737838630373
  abs_mean_dist_err_boat = 25.058669469783354
Distance bin (200.0, 400.0):
  samples:  812
  weighted_reL_dist_err_boat = 0.2057969997856522
  abs_mean_dist_err_boat = 64.8434421182266
Distance bin (400.0, 600.0):
  samples:  790
  weighted_reL_dist_err_boat = 0.1542782489203172
  abs_mean_dist_err_boat = 70.57278481012658
Distance bin (600.0, 800.0):
  samples:  350
  weighted_reL_dist_err_boat = 0.13248381672101447
  abs_mean_dist_err_boat = 107.10892857142858
Distance bin (800.0, 1000.0):
  samples:  69
  weighted_reL_dist_err_boat = 0.1258767028866825
  abs_mean_dist_err_boat = 110.01449275362319
Total Samples:  2907
Overall weighted_rel_dist_err_boat = 0.18248750203805492
Overall abs_mean_dist_err_boat = 60.99138394392845
Mean heading error = 52.6 degrees
Combined Metric (MAP & distance) =  0.2853667810023252
Combined_metric (MAP, distance & heading) =  0.2019852521421183


                Class      Images      Labels        P           R         mAP@.5    mAP@.5:.95:
                 all        2268        3439       0.773       0.639       0.669       0.272
```
The Distance Error is computed for 5 distance bins. The interval size of a bin depends on the max dist hyperparameter passed to the testscript in hyp.scratch-p5.yaml.

Furthermore the default YOLOv7 statistics for Object Detection are displayed.

The weighted relative distance error is defined as follows:

$$\epsilon_{Dist} = \sum_{i=1}^{n} \frac{c_i}{\sum_{j=1}^{n} c_j} \frac{|d_i - \hat{d}_i|}{d_i}$$

where $i$ is the index of the test sample, $n$ is the cardinality of the test set, $c_i$ the confidence of the prediction 
(objectness * class probability $\rightarrow$ since we only have one class, this is equal to objectness), $d_i$ the ground 
truth distance and $\hat{d}_i$ the predicted distance.
Since predictions for distant objects naturally have higher deviations, we employ a relative measure to also penalize smaller absolute errors for close objects. 

The heading error is defined as follows:

$$\epsilon_{Head} = \min(|(\hat{h} - h)|, 360 - |(\hat{h} - h)|)$$

To produce a final score which takes the object detection performance, the distance error and the heading error into account, the combined metric is specified as:

$$\text{Combined Metric} = \text{mAP@[0.5:0.95]} \cdot (1 - \min(\epsilon_{Dist}, 1)) \cdot (1 - \min(\epsilon_{Head}, 1))$$

## Inference

Use the detect script to run inference on video.
The first number is the confidence value, the second number the metric distance estimate in meters and the third number the heading angle estimate in degrees which is visualized by the arrow.
