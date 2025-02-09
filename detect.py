from ultralytics import YOLO
import cv2

model = YOLO(r"D:\Fruit Ninja\runs\detect\orange_classifier_rtx40508\weights\best.pt").to('cuda')

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open cameras.")
    exit()

def detect_quality(frame, model):
    results = model(frame, conf=0.25)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            return label
    return None

def conclude_quality(quality_cam1, quality_cam2):
    if quality_cam1 == "fresh" and quality_cam2 == "fresh":
        return "fresh"
    else:
        return "rotten"

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Failed to capture frames from one or both cameras")
        break

    quality_cam1 = detect_quality(frame1, model)
    quality_cam2 = detect_quality(frame2, model)

    if quality_cam1 is not None and quality_cam2 is not None:
        final_quality = conclude_quality(quality_cam1, quality_cam2)
        print(f"Final quality of the orange: {final_quality}")

    results1 = model(frame1, conf=0.25)
    results2 = model(frame2, conf=0.25)
    annotated_frame1 = results1[0].plot()
    annotated_frame2 = results2[0].plot()

    cv2.imshow("Camera 1 - YOLOv8 Orange Defect Detection", annotated_frame1)
    cv2.imshow("Camera 2 - YOLOv8 Orange Defect Detection", annotated_frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
