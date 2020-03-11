import cv2
import time
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='[{asctime}][{name}][{levelname}] {message}', style='{')
_logger = logging.getLogger(__name__)

backSub = cv2.createBackgroundSubtractorMOG2()

capture = cv2.VideoCapture(1)

INTERVAL_s = 1

_logger.debug('Original frame rate: {} fps'.format(capture.get(cv2.CAP_PROP_FPS)))

ret_max_fps = capture.set(cv2.CAP_PROP_FPS, np.inf)
_logger.debug('Try to set the frame rate to +inf. Success? {}'.format(ret_max_fps))
_logger.debug('Maximum frame rate: {} fps'.format(capture.get(cv2.CAP_PROP_FPS)))

ret_min_fps = capture.set(cv2.CAP_PROP_FPS, 0.1)
_logger.debug('Try to set the frame rate to 0.1. Success? {}'.format(ret_min_fps))
_logger.debug('Minimum frame rate: {} fps'.format(capture.get(cv2.CAP_PROP_FPS)))

min_frame_rate = capture.get(cv2.CAP_PROP_FPS)

THRESHOLD = 1000


prev_time = time.time()

while True:
    # ret, frame = capture.read()
    # if frame is None:
    #     break

    grab_success = capture.grab()
    if time.time() - prev_time < INTERVAL_s:
        continue

    if not grab_success:
        _logger.error('Failed to grab a frame')
        continue



    retrieve_success, frame = capture.retrieve()
    prev_time = time.time()

    # frame = cv2.resize(frame, (100, 100))
    #
    # frame = cv2.blur(frame, (10, 10))
    #
    # fgMask = backSub.apply(frame)
    #
    # ret, fgMask = cv2.threshold(fgMask, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    #
    # if np.sum(fgMask / 255) > THRESHOLD:
    #     print('Motion detected at {}'.format(time.strftime('%H%M%S')))

    cv2.imshow('Frame', frame)
    _logger.info('hi')
    # cv2.imshow('FG Mask', fgMask)

    keyboard = cv2.waitKey(1)
    if keyboard == ord('q') or keyboard == 27:
        break

    # time.sleep(1)

cv2.destroyAllWindows()