# YOLOv7 with Distance Estimation

This forked and adjusted repo has scripts and methods for training a [YOLOv7](https://github.com/WongKinYiu/yolov7) 
detection model including distance predictions. For each
anchor, it additionally predicts the normalized metric distance of that
object and concatenates this with the default YOLOv7 Detection Vector containing
class predictions, objectness and boundingbox dimensions.
During inference, the normalized distance is rescaled according to the defined maximum distance.
Objects are only lateral marks, they can come in 
different shapes or forms, e.g. see [Dataset](#dataset). 

The train set may contain ambiguous object appearances in that
some marks are not visibly distinguishable from other types
of navigational or other maritime marks. Feel free to blacken 
these yourself. The test set is largely free of these ambiguities
and mostly contains scenes without "distractor" marks.

Currently, it only supports the "normal" YOLOv7 model, not
bigger ones. If you deviate from any of the below settings,
please make sure the network layers (the head) is correct.

You can download the dataset and pretrained weights
as per the [challenge webpages](https://macvi.org/workshop/macvi25/challenges/usv_dist). 

The following instructions guide you through training the adapted model and running inference on images and video.

## Training
Single GPU training
``` shell
python YOLOv7-DL23/train.py --workers 8 --device 0 --batch-size 4 --data 'path/to/data.yaml' --img 1024 1024 --cfg YOLOv7-DL23/cfg/training/yolov7_custom.yaml --weights 'YOLOv7-DL23/init_weights.pt' --name yolov7_dist_v1 --hyp YOLOv7-DL23/data/hyp.scratch.p5.yaml
```
Replace 'path/to/data.yaml' with the path to the yaml file contained in the dataset folder from the challenge website.
Note that a customised hyperparameter file is used where distance scaling method and max distance are defined.

Multi GPU training
``` shell
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 YOLOv7-DL23/train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 16 --data 'path/to/data.yaml' --img 1024 1024 --cfg YOLOv7-DL23/cfg/training/yolov7_custom.yaml --weights 'YOLOv7-DL23/init_weights.pt' --name yolov7_dist_v1 --hyp YOLOv7-DL23/data/hyp.scratch.p5.yaml
```
You may have to replace the --local-rank argument in the train.py script with --local_rank depending on your CUDA version.

<br/>

The *results.txt* file has the following format:
```
Epoch  GPUMem  train_box  train_obj  train_cls  train_dist  total  labels  img_size  precision  recall  map0.5  map.5:.95  val_box  val_obj  val_cls  val_dist
```
The generated *results.png* file displays box, objectness and distance loss on the train and validatin set.

## Testing
> [!NOTE]
> :pushpin: Pretrained Weights are available [here](https://drive.google.com/drive/folders/1GujICE9Ev-ppfH4PUX19UjywqgFn-5Zf?hl=de) 

Using the pretrained model, you can evaluate its performance w.r.t. object detection and distance estimation:

``` shell
python YOLOv7-DL23/test.py --data 'path/to/data.yaml' --img 1024 --batch 4 --conf 0.001 --iou 0.65 --device 0 --weights 'YOLOv7-DL23/init_weights.pt' --name yolov7_DistV1_test --task 'test' --hyp 'YOLOv7-DL23/data/hyp.scratch.p5.yaml'
```
Make sure that the data.yaml file contains a test or val entity depending on the task argument.


Sample output for a model with pretrained weights on testsplit:

```
Distance bin (0.0, 200.0):
  samples:  877
  weighted_reL_dist_err_buoy = 0.19412737838630373
  abs_mean_dist_err_buoy = 25.058669469783354
Distance bin (200.0, 400.0):
  samples:  812
  weighted_reL_dist_err_buoy = 0.2057969997856522
  abs_mean_dist_err_buoy = 64.8434421182266
Distance bin (400.0, 600.0):
  samples:  790
  weighted_reL_dist_err_buoy = 0.1542782489203172
  abs_mean_dist_err_buoy = 70.57278481012658
Distance bin (600.0, 800.0):
  samples:  350
  weighted_reL_dist_err_buoy = 0.13248381672101447
  abs_mean_dist_err_buoy = 107.10892857142858
Distance bin (800.0, 1000.0):
  samples:  69
  weighted_reL_dist_err_buoy = 0.1258767028866825
  abs_mean_dist_err_buoy = 110.01449275362319
Total Samples:  2907
Overall weighted_rel_dist_err_buoy = 0.18248750203805492
Overall abs_mean_dist_err_buoy = 60.99138394392845
Combined Metric =  0.22202799959332511

                Class      Images      Labels        P           R         mAP@.5    mAP@.5:.95:
                 all        2268        3439       0.773       0.639       0.669       0.272
```
The Distance Error is computed for 5 distance bins. The interval size of a bin depends on the max dist hyperparameter passed to the testscript in hyp.scratch-p5.yaml.

Furthermore the default YOLOv7 statistict for Object Detection are displayed.

## Inference

Use the detect script to run inference on video:
``` shell
python YOLOv7-DL23/detect.py --weights 'YOLOv7-DL23/init_weights.pt' --conf 0.25 --img-size 1024 --source '/path/to/video.avi'
```

![](https://github.com/Ben93kie/YOLOv7-DL23/blob/distance_network/assets/detect.gif)

The first number is the confidence value, the second number the metric distance estimate in meters.

## Export

You need to export your model to ONNX to be evaluated on the
server for getting displayed on the [leaderboard](https://macvi.org/leaderboard/surface/distEstimation/distance-estimation).

To export the model provided in this repo, run:
``` shell
python YOLOv7-DL23/export_yoloV7_withdistances.py --weights 'YOLOv7-DL23/init_weights.pt' --size 1024 1024
```
This will create an .onnx file and a labels.txt file. The exported model in ONNX format can be uploaded to the [testserver](https://macvi.org/upload?track=Distance+Estimation).
Be aware that you might need to make adjustments to the export script depending on the adaptions you make to the model architecture.

> [!NOTE]
> Model submission requirements:
> - Model Parameters not exceeding 50 million
> - Exported for image size of 1024x1024
> - NMS not included
> - Output tensor Format: [batch_sz, num_anchors, [xyhw,objectness,class,dist]]

To test whether your exported ONNX model meets the specifications you can run:
``` shell
python YOLOv7-DL23/testscript_onnx.py --weights 'YOLOv7-DL23/model.onnx' --data 'path/to/data.yaml'
```
The script is similar to the test.py file but uses the ONNX model instead of relying on the pytorch framework. An almost identical script is used on the testserver.
## Dataset
The [Dataset](https://drive.google.com/drive/folders/1M-K03ELa1Lf8Ob-sVJFEFMBrpQMS0210?hl=de) contains around 3000 images of maritime navigational aids (mostly red/green buoy markers). You are only provided with a training set. 
The testset is withheld to create a benchmark for all submitted models during the competition.

We provide a reliable distance ground truth value by computing the haversine distance between the cameras GPS position for each frame and mapped navigational buoys.
As you might notice when inspecting some of the images we decided to also include samples where the distance to the object is significantly large and the object only 
consist of a few pixels in the video feed. This might push the boundaries of the object detector, hence decreasing the mAP metric. Since the evaluation metric of the challenge also includes
object detection critera (e.g. mAP) you can feel free to remove samples from the training data where this might be the case.

Examples containing the most common buoy types:
<p float="left">
  <img src="https://github.com/Ben93kie/YOLOv7-DL23/blob/distance_network/assets/Figure_1.png" width="400" />
  <img src="https://github.com/Ben93kie/YOLOv7-DL23/blob/distance_network/assets/Figure_2.png" width="400" /> 
</p>


The dataset follows the YOLO format convention, where images and labels are located in separate folders and each image is linked to a corresponding labels (.txt) file.
Each line in the textfile represents a bounding box:
```text
class-id  center-X  center-Y  width  height  distance
```
The Bounding Box coordinates and dimensions are normalized. The distance on the other hand is provided as a metric value in meters!

## Evaluation
The submitted models are evaluated on the test split of the dataset. The test set is not publically available.

Given that the challenge seeks to address both monocular distance estimation and object detection, two performance metrics are utilized. 
The quality of object detection task for the submitted models is assessed using the mAP@[.5:.95] metric.
The distance error is defined as follows:

$$\epsilon_{Dist} = \sum_{i=1}^{n} \frac{c_i}{\sum_{j=1}^{n} c_j} \frac{|d_i - \hat{d}_i|}{d_i}$$

where $i$ is the index of the test sample, $n$ is the cardinality of the test set, $c_i$ the confidence of the prediction 
(objectness * class probability $\rightarrow$ since we only have one class, this is equal to objectness), $d_i$ the ground 
truth distance and $\hat{d_i}$ the predicted distance.
Since predictions for distant objects naturally have higher deviations, we employ a relative measure to also penalize smaller absolute errors for close objects. 
