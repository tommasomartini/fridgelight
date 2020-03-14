import logging
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='[{asctime}][{name}][{levelname}] {message}',
                    style='{')
_logger = logging.getLogger(__name__)


_THRESHOLD = 1000
_VIEW = True
_CAMERA_ID = 0
_FPS_hz = 10


def _initialize_camera():
    cap = cv2.VideoCapture(_CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera {}'.format(_CAMERA_ID))

    _logger.debug('Camera {}: opened'.format(_CAMERA_ID))

    ret_fps = cap.set(cv2.CAP_PROP_FPS, _FPS_hz)
    if ret_fps is False:
        raise RuntimeError('Camera {}: failed to set '
                           'FPS to {} '.format(_CAMERA_ID, _FPS_hz))

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps != _FPS_hz:
        _logger.warning('Camera {}: tried to set FPS to {}, '
                        'but actual FPS is {}'
                        .format(_CAMERA_ID, _FPS_hz, actual_fps))
    else:
        _logger.debug('Camera {}: set FPS to {}'.format(_CAMERA_ID, _FPS_hz))

    return cap


def _process_frame(frame):
    resized_frame = cv2.resize(frame, (100, 100))
    blurred_frame = cv2.blur(resized_frame, (10, 10))
    return blurred_frame


def _main():
    cap = _initialize_camera()
    bgnd_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret_val, frame = cap.read()
        if ret_val is False:
            continue

        proc_frame = _process_frame(frame)
        fgnd_mask = bgnd_subtractor.apply(proc_frame)
        ret_thr, bw_fgnd_mask = cv2.threshold(fgnd_mask,
                                              thresh=1,
                                              maxval=255,
                                              type=cv2.THRESH_BINARY)

        if np.sum(bw_fgnd_mask / 255) > _THRESHOLD:
            _logger.debug('Motion detected at {}'
                          .format(time.strftime('%H%M%S')))

        if _VIEW:
            cv2.imshow('Frame', frame)
            cv2.imshow('Processed frame', proc_frame)
            cv2.imshow('FG Mask', bw_fgnd_mask)

            keyboard = cv2.waitKey(1)
            if keyboard == ord('q') or keyboard == 27:
                break

    if _VIEW:
        cv2.destroyAllWindows()

    cap.release()

    _logger.debug('Goodbye!')


if __name__ == '__main__':
    _main()
