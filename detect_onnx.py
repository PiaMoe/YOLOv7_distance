import cv2
import numpy as np
import onnxruntime as ort
import random
import math


w = "../runs/train/finalDataset/B3_singleHead/weights/B3_singleHead_test.onnx"

names = ['boat', 'other']
colors = {name: [random.randint(0, 255) for _ in range(3)] for name in names}

# start session
session = ort.InferenceSession(w, providers=['CPUExecutionProvider'])
inname = [i.name for i in session.get_inputs()]
outname = [i.name for i in session.get_outputs()]


def letterbox(im, new_shape=(608, 1088), color=(114,114,114), stride=32):
    shape = im.shape[:2]  # h, w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


cap = cv2.VideoCapture("../../../data/new_downloads/BoatApproachingPort/Video Of Boat Approaching Port.mp4")

def truncate(num, decimals):
    factor = 10.0 ** decimals
    return int(num * factor) / factor

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img, ratio, dwdh = letterbox(frame)
    img = img.transpose((2,0,1))[None] / 255.0
    img = img.astype(np.float32)

    outputs = session.run(outname, {inname[0]: img})[0]

    # outputs shape: (1, N, 6)
    outputs = session.run(outname, {inname[0]: img})[0]
    pred = outputs[0]  # (N, 6)

    boxes = pred[:, :4]  # [x0, y0, x1, y1]
    scores = pred[:, 4]  # confidence
    cls_ids = pred[:, 5].astype(int)

    mask = scores > 0.65
    boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]

    # OpenCV NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.3,
        nms_threshold=0.45
    )

    for i in indices:
        i = int(i)
        x0, y0, x1, y1 = boxes[i]
        cls_id = cls_ids[i]
        score = scores[i]
        conf = truncate(score, 2)

        # decode distance and heading from confidence
        rest = score - conf
        distance = int(rest * 1e5)
        heading = int(truncate((rest * 1e8) % 1000, 0))

        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(int).tolist()

        name = f"{names[cls_id]} {conf} {distance}m {heading}deg"
        color = colors[names[cls_id]]

        cv2.rectangle(frame, box[:2], box[2:], color, 2)
        cv2.putText(frame, name, (box[0], box[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if heading:
            # Plots arrow pointing in heading direction
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            rad = math.radians(heading)
            box_w = abs(box[2] - box[0])
            box_h = abs(box[3] - box[1])
            box_diag = math.sqrt(box_w ** 2 + box_h ** 2)
            img_diag = math.sqrt(frame.shape[0] ** 2 + frame.shape[1] ** 2)
            arrow_length = max(0.03 * img_diag, min(0.15 * img_diag, box_diag * 0.5))
            end_x = int(center_x + arrow_length * math.sin(rad))
            end_y = int(center_y + arrow_length * math.cos(rad))
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), color, thickness=2, line_type=cv2.LINE_AA,
                            tipLength=0.2)

    cv2.imshow("YOLOv7 ONNX Inference", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to end
        break

cap.release()
cv2.destroyAllWindows()
