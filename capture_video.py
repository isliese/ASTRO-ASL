import cv2 as cv
import numpy as np


'''class handDetection():
    def __init__(self)-> None:
       pass
    def frame():
'''

# Set capture at video index
capture = cv.VideoCapture(0)

# Check if camera opens
if not capture.isOpened():
    print("Cannot open camera")
    print("Please check your access to camera")
    exit()

while True:
    # Read frames from video capture
    ret, frame = capture.read()

    if not ret:
        print("Error recieving frame. ")
        break

    # Convert image to gray scale
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    # Show video
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'): # Press 'q' to quit
        break

capture.release()
cv.destroyAllWindows()