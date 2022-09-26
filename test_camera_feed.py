import cv2
import keyboard

# initiallize camera
capture = cv2.VideoCapture(0)

while(True):
    # look at the latest video frame
    ret, frame = capture.read()
    # display latest frame
    cv2.imshow("frame", frame)

    res = cv2.waitKey(1)

    # Stop if the user presses "q"
    if res & 0xFF == ord('q'):
        break

# When everything is done, release the capture
capture.release()
cv2.destroyAllWindows()