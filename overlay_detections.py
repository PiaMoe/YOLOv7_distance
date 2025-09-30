from unicodedata import category

import cv2
import csv
import os
import sys
from utils.plots import plot_one_box

def get_color_based_on_distance(distance):
    if distance <= 50:
        return (0, 0, 255)  # Red in BGR
    elif 50 < distance <= 150:
        return (0, 165, 255)  # Orange in BGR
    elif 150 < distance <= 300:
        return (0, 250, 250)  # Yellow in BGR
    else:
        return (255, 0, 0)  # Blue in BGR


def load_detections(det_file):
    detections = {}
    with open(det_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 11:
                continue

            frame = int(row[0])
            frame = frame + 10
            category = row[1]
            obj_id = row[2]
            x, y, w, h = map(float, row[3:7])
            distance = row[7]
            heading = row[8]
            conf = row[10]
            if frame not in detections:
                detections[frame] = []
            detections[frame].append({
                "category": category,
                "obj_id": obj_id,
                "bbox": (x, y, w, h),
                "distance": distance,
                "heading": heading,
                "conf": conf
            })
    return detections

def overlay_video(video_file, detections, out_file="output_overlay.mp4"):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("❌ Could not open video:", video_file)
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_file, fourcc, fps, (frame_w, frame_h))

    # Skalierungsfaktoren relativ zu 1920x1080
    scale_x = frame_w / 1920.0
    scale_y = frame_h / 1080.0

    frame_idx = 0
    while True:
        print(frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in detections:
            for det in detections[frame_idx]:
                # BB aus CSV
                x, y, bw, bh = map(float, det["bbox"])
                # Skalieren auf aktuelle Auflösung
                x = int(x * scale_x)
                y = int(y * scale_y)
                bw = int(bw * scale_x)
                bh = int(bh * scale_y)
                xyxy = (x, y, x + bw, y + bh)

                category = det["category"]
                distance = int(det["distance"])
                heading = int(det["heading"])
                conf = float(det["conf"])

                print("heading:", heading, "distance:", distance)
                if category != 'boat' or distance > 300:  # only predict heading for close boats
                    heading = None

                label = f'{category} {conf:.2f} {distance} {heading}' if heading else f'{category} {conf:.2f} {distance}'
                color = get_color_based_on_distance(distance)
                txtcolor = [255, 255, 255]
                if color == (0, 250, 250):  # Yellow
                    txtcolor = [0, 0, 0]
                plot_one_box(xyxy, frame, label=label, color=color, line_thickness=1, textcolor=txtcolor, heading=heading)

        out.write(frame)
        cv2.imshow("Overlay", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Overlay saved to", out_file)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python overlay_detections.py <video_file> <detections_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    det_file = sys.argv[2]
    detections = load_detections(det_file)
    out_file = video_file.replace(".mp4", "_overlay.mp4")
    overlay_video(video_file, detections, out_file)

