import cv2 as cv
import numpy as np


'''class handDetection():
    def __init__(self)-> None:
       pass
    def frame():
'''

# set capture at video index
capture = cv.VideoCapture(0)

# check if camera opens (allow mac permissions possibly)
if not capture.isOpened():
    print("Cannot open camera")
    print("Please check your access to camera")
    exit()

while True:
    # read frames from video capture
    ret, frame = capture.read()

    if not ret:
        print("Error recieving frame. ")
        break

    # convert image to gray scale
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    # show video
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'): # press 'q' to quit
        break

capture.release()
cv.destroyAllWindows()