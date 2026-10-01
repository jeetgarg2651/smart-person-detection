from ultralytics import YOLO
import winsound
import cv2


model = YOLO("yolov8m.pt") # model load


cap = cv2.VideoCapture("input.mp4") # video open hogi read hoti hai




if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

print("Video Loaded Successfully")

# Read first frame
success, first_frame = cap.read()

if not success:
    print("Cannot read first frame")
    exit()

display_frame = cv2.resize(first_frame, (1280, 720))

# Select restricted area
roi = cv2.selectROI("Select Restricted Area", display_frame)

cv2.destroyWindow("Select Restricted Area")

RX1, RY1, W, H = roi
RX2 = RX1 + W
RY2 = RY1 + H

# Restart video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)




unique_persons = set()

alerted_ids = set()
while True:
    
    success, frame = cap.read() # read one frame

    if not success: # video end then break
        break

    cv2.rectangle(
        frame,
        (RX1, RY1),
        (RX2, RY2),
        (255, 0, 0),
        3
    )



    results = model.track(
        frame,
        persist=True,
        conf=0.6,
        verbose=False,
        tracker="bytetrack.yaml"
    ) # current frame ko yolo ke pass bhejta hai


    person_count = 0


    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0]) # coco dataset me 0 = person ki id


            if cls != 0: # sirf person detect karo
                continue


            if box.id is not None: # track id
                track_id = int(box.id[0])
                unique_persons.add(track_id)
            else:
                track_id = -1

            if track_id == -1:
                continue


            person_count += 1


            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Person ka center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Center point draw (optional)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Check agar person restricted area ke bahar hai
            if not (RX1 <= cx <= RX2 and RY1 <= cy <= RY2):
                cv2.putText(
                    frame,
                    f"ALERT! ID: {track_id}",
                    (50,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
)
                
                if track_id not in alerted_ids:
                    print(f"Alert! Person ID {track_id} is outside the restricted area.")
                    winsound.Beep(1000, 500)
                    alerted_ids.add(track_id)
            else:
                if track_id in alerted_ids:
                    alerted_ids.remove(track_id)
            
            cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
            )


            cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
            )


    cv2.putText(
    frame,
    f"Persons: {person_count}",
    (30, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    2,
    (0,0,255),
    4
)
    

    # Video display size fix (zoom issue solve)
    h, w = frame.shape[:2]

    scale = 800 / w

    new_w = 800
    new_h = int(h * scale)

    display_frame = cv2.resize(
        frame,
        (new_w, new_h)
    )


    cv2.imshow(
        "Input Video",
        display_frame
    )


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

print("=" * 30)
print(f"Total Unique Persons: {len(unique_persons)}")
print("=" * 30)


cap.release()
cv2.destroyAllWindows()