import cv2
import time
import numpy as np

backSub = cv2.createBackgroundSubtractorMOG2()

capture = cv2.VideoCapture(1)
frame_rate = capture.get(cv2.CAP_PROP_FPS)
ret = capture.set(cv2.CAP_PROP_FPS, 1)

print(ret)

THRESHOLD = 1000

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    frame = cv2.resize(frame, (100, 100))

    frame = cv2.blur(frame, (10, 10))

    fgMask = backSub.apply(frame)

    ret, fgMask = cv2.threshold(fgMask, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

    if np.sum(fgMask / 255) > THRESHOLD:
        print('Motion detected at {}'.format(time.strftime('%H%M%S')))

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    # keyboard = cv2.waitKey(30)
    # if keyboard == 'q' or keyboard == 27:
    #     break

    # time.sleep(1)