# YOLOv7 with Distance and Heading Estimation

This forked and adjusted repo has scripts and methods for training a [YOLOv7](https://github.com/WongKinYiu/yolov7) 
detection model including distance predictions. For each
anchor, it additionally predicts the normalized metric distance and a normalized heading direction (viewpoint) of that
object. Both values are concatenated with the default YOLOv7 Detection Vector containing
class predictions, objectness and boundingbox dimensions.
During inference, the normalized distance is rescaled according to the defined maximum distance and the heading value is rescaled to [0°, 360°]. 


## Dataset
The `data/` directory should contain a `.yaml` file with the following structure:

```yaml
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images

nc: 1
names: ['boat']
```

For each image in the dataset, there should be a corresponding label file located in a parallel labels/ directory. Each label file must follow this format:
```
class x y w h distance heading
```

Here:
- `class` is the class index (e.g., 0 for 'boat', 1 for 'buoy'),
- `x, y, w, h` are the **normalized** coordinates of the bounding box (center x/y, width, height),
- `distance` is the estimated distance to the object **in meters**.
- `heading` is the estimated heading angle **in degrees**.

*Bild einfügen von Kreis mit Winkeln*

## Training
Single GPU training
``` shell
python3 YOLOv7_distance/train.py --workers 8 --device 0 --batch-size 4 --data 'path/to/data.yaml' --img 1024 1024 --cfg YOLOv7_distance/cfg/training/yolov7_custom.yaml --weights 'YOLOv7_distance/init_weights.pt' --name yolov7_dist_v1 --hyp YOLOv7_distance/data/hyp.scratch.p5.yaml
```
Replace 'path/to/data.yaml' with the path to the data yaml file.
Note that a customised hyperparameter file is used where distance scaling method and max distance are defined.

Multi GPU training
``` shell
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 YOLOv7_distance/train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 16 --data 'path/to/data.yaml' --img 1024 1024 --cfg YOLOv7_distance/cfg/training/yolov7_custom.yaml --weights 'YOLOv7_distance/init_weights.pt' --name yolov7_dist_v1 --hyp YOLOv7_distance/data/hyp.scratch.p5.yaml
```
You may have to replace the --local-rank argument in the train.py script with --local_rank depending on your CUDA version.

<br/>

The *results.txt* file has the following format:
```
Epoch  GPUMem  train_box  train_obj  train_cls  train_dist  train_head total  labels  img_size  precision  recall  map0.5  map.5:.95  dist_err combined_metric val_box  val_obj  val_cls  val_dist val_head
```
The generated *results.png* file displays box, objectness, distance and heading loss on the train and validatin set.

## Testing
> [!NOTE]
> :pushpin: Pretrained Weights are available [here](https://drive.google.com/drive/folders/1GujICE9Ev-ppfH4PUX19UjywqgFn-5Zf?hl=de) 

Using the pretrained model, you can evaluate its performance w.r.t. object detection, distance estimation and heading estimation:

``` shell
python YOLOv7_distance/test.py --data 'path/to/data.yaml' --img 1024 --batch 4 --conf 0.001 --iou 0.65 --device 0 --weights 'YOLOv7_distance/init_weights.pt' --name yolov7_DistV1_test --task 'test' --hyp 'YOLOv7_distance/data/hyp.scratch.p5.yaml'
```
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

Use the detect script to run inference on video:
``` shell
python YOLOv7_distance/detect.py --weights 'YOLOv7_distance/init_weights.pt' --conf 0.25 --img-size 1024 --source '/path/to/video.avi'
```
*Inference Sequenz einfügen*

The first number is the confidence value, the second number the metric distance estimate in meters and the third number the heading angle estimate in degrees which is visualized by the arrow.
