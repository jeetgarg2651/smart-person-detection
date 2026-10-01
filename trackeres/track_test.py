from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.track(
    source="input.mp4",
    tracker="bytetrack.yaml",
    show=True
)