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

You can download the dataset (and potentially pretrained weights)
as per the challenge webpages. 

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

## Testing

You can test on your trained model, or download a 
pre-trained model here:

TODO

Using the pretrained model, you can compute its accuracy:

``` shell
python YOLOv7-DL23/test.py --data 'path/to/data.yaml' --img 1024 --batch 4 --conf 0.001 --iou 0.65 --device 0 --weights 'YOLOv7-DL23/init_weights.pt' --name yolov7_DistV1_test --task 'test' --hyp 'YOLOv7-DL23/data/hyp.scratch.p5.yaml'
```
Make sure that the data.yaml file contains a train entity.

Sample output for a model with pretrained weights:

```
Distance bin (0.0, 200.0):
  mean_dist_err_boat = 11.461130875618755
  mean_dist_err_other = -1
Distance bin (200.0, 400.0):
  mean_dist_err_boat = 19.127938123078223
  mean_dist_err_other = -1
Distance bin (400.0, 600.0):
  mean_dist_err_boat = 22.477531653126366
  mean_dist_err_other = -1
Distance bin (600.0, 800.0):
  mean_dist_err_boat = 28.41937782634907
  mean_dist_err_other = -1
Distance bin (800.0, 1000.0):
  mean_dist_err_boat = 52.7596413584453
  mean_dist_err_other = -1
Overall mean_dist_err_boat = 21.641303812311364
Overall mean_dist_err_other = -1

               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95
                 all         522         785       0.936       0.926       0.944       0.499
```
The Distance Error is computed for 5 distance bins. The interval size of a bin depends on the max dist hyperparameter passed to the testscript in hyp.scratch-p5.yaml.

Furthermore the default YOLOv7 statistict for Object Detection are displayed.

## Inference

On video:
``` shell
python YOLOv7-DL23/detect.py --weights 'YOLOv7-DL23/init_weights.pt' --conf 0.25 --img-size 1024 --source '/path/to/video.avi'
```

## Export

You need to export your model to ONNX to be evaluated on the
server for getting displayed on the leaderboard.


Instructions coming soon

## Dataset
The Dataset contains around 4000 images of maritime navigational aids (mostly red/green buoy markers). You are provided with a training and validaton set. The testset is
withheld to create a benchmark for all submitted models during the competition.

We provide a reliable distance ground truth value by computing the haversine distance between the cameras GPS position for each frame and mapped navigational buoys.
As you might notice when inspecting some of the images we decided to also include samples where the distance to the object is significantly large and the object only 
consist of a few pixels in the video feed. This might push the boundaries of the object detector, hence decreasing the mAP metric. Since the evaluation metric of the challenge also includes
object detection critera (e.g. mAP) you can feel free to remove samples from the training data where this might be the case.

Examples containing the most common buoy types:
<p float="left">
  <img src="https://github.com/Ben93kie/YOLOv7-DL23/blob/distance_network/assets/Figure_1.png" width="400" />
  <img src="https://github.com/Ben93kie/YOLOv7-DL23/blob/distance_network/assets/Figure_2.png" width="400" /> 
</p>


The dataset follows the YOLO format convention, where images and labels are located in seperate folders and each image is linked to a corresponding labels (.txt) file.
Each line in the textfile represents a bounding box:
```text
class-id  center-X  center-Y  width  height  distance
```
The Bounding Box coordinates and dimensions are normalized. The distance on the other hand is provided as a metric value in meters!

## Evaluation
The submitted models are evaluated on the test split of the training dataset. The test set is not publically available.

Given that the challenge seeks to address both monocular distance estimation and object detection, two performance metrics are utilized. 
The quality of object detection task for the submitted models is assessed using the mAP[.5:.95] metric.
The distance error is defined as follows:

$$\epsilon_{Dist} = \frac{1}{n}\sum_{i}^{n} c_i \frac{|d_i-\hat{d_i}|}{d_i}$$

where $i$ is the index of the test sample, $n$ is the cardinality of the test set, $c_i$ the confidence of the prediction 
(objectness * class probability $\rightarrow$ since we only have one class, this is equal to objectness), $d_i$ the ground 
truth distance and $\hat{d_i} the predicted distance.
Since predictions for distant objects naturally have higher deviations, we employ a relative measure to also penalize smaller absolute errors for close objects. 
