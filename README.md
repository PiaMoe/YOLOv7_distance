# YOLOv7 with Distance Estimation

This forked and adjusted repo has scripts and methods for training a YOLOv7 
detection model including distance predictions. For each
anchor, it additionally predicts the normalized metric distance of that
object. Objects are only lateral marks, they can come in 
different shapes or forms, e.g. see TODO. 

The train set may contain ambiguous object appearances in that
some marks are not visibly distinguishable from other types
of navigational or other maritime marks. Feel free to blacken 
these yourself. The test set is largely free of these ambiguities
and mostly contains scenes without "distractor" marks.

Currently, it only supports the "normal" YOLOv7 model, not
bigger ones. If you deviate from any of the below settings,
please make sure the network layers (the head) is correct.

Download the dataset (and potentially pretrained weights)
as per the challenge webpages and put them in the correct path
according to data/CharlestonWithDistance.yaml.

## Training

``` shell
python train.py ---data data/CharlestonWithDistance.yaml
```

## Testing

You can test on your trained model, or download a 
pre-trained model here:

TODO

Using the pretrained model, you can compute its accuracy:

``` shell
python test.py --weights "best.pt" --data data/CharlestonWithDistance.yaml
```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.51206
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.69730
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.55521
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.35247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.55937
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66693
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.63765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.68772
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.53766
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.73549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.83868
```

## Inference

On video:
``` shell
python --weights "C:\Users\ben93\My Drive\Weights\distance\best.pt" --source "G:\Shared drives\TempRecordings\BGfirst\75.avi" --img-size 1280```
```



## Export

You need to export your model to ONNX to be evaluated on the
server for getting displayed on the leaderboard.


Instructions coming soon


