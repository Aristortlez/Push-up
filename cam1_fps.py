from cvzone import FPS
import cv2

cap = cv2.VideoCapture(0)
fpsReader = FPS()
while True:
    ret, frame = cap.read()
    fps, frame = fpsReader.update(frame)
    print(fps)

    cv2.imshow("Image", frame)
    if ord('q') == 0xFF & cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()