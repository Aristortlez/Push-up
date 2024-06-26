from cvzone import FPS
from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0)
detector = HandDetector()
fpsReader = FPS()
while True:
    ret, frame = cap.read()
    hands, frame = detector.findHands(frame)
    print("Hands:", frame)
    fps, frame = fpsReader.update(frame)
    print(fps)

    cv2.imshow("Image", frame)
    if ord('q') == 0xFF & cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()