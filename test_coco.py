from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# GT laden
cocoGt = COCO("instances_val.json")   # Ground truth

# Predictions laden
cocoDt = cocoGt.loadRes("predictions.json")

# Evaluation vorbereiten
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
