from ultralytics import YOLO
import cv2

model = YOLO('../yoloWeights/yolov8l.pt')
results = model("Images/circulation.jpg",show=True)
cv2.waitKey(0)
 