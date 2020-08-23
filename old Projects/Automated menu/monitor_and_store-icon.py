"""
We are going to use:
    - PyAutoGUI -> To actually do the playing.
    - mss -> To capture video data fast.
    - OpenCV -> For computer vision.

"""

import numpy as np
import cv2
from mss import mss
from PIL import Image


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        img_blank = np.zeros((height, width, 3), np.uint8)
        hor = [img_blank] * rows
        hor_con = [img_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def get_contours(img, img_contour, screen_object):  # Input image and define
    # Define which image and retrieval method. Retrieval method retrieves extreme outer contours. Define approximation.
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)  # For each contour, find the area.
        # print(area)  # Area of shapes detected.
        if 10 < area < 100:  # Check for minimum area and give a minimum threshold for it.
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)  # Image. Contours. Index (-1 to write). Thickness
            perimeter = cv2.arcLength(cnt, True)  # Find arc length of contours. True = contours are closed.
            # print(perimeter)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter,
                                      True)  # Contour. Define resolution. True = contours are closed.
            # print(len(approx))  # Print the numbers of corners of each shape.
            objectCorner = len(approx)
            x, y, w, h = cv2.boundingRect(approx)  # Create bounding box around detected object.

            if screen_object == "Container_notify":
                object_type = "Container_notify"

            cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(objectCorner)


# Track bar stuff
def empty(i):
    pass

def init_trackbar():
    cv2.namedWindow('TrackBars')
    cv2.resizeWindow('TrackBars', 640, 240)
    cv2.createTrackbar('Hue Min', 'TrackBars', 0, 179, empty)
    cv2.createTrackbar('Hue Max', 'TrackBars', 0, 179, empty)
    cv2.createTrackbar('Sat Min', 'TrackBars', 140, 255, empty)
    cv2.createTrackbar('Sat Max', 'TrackBars', 240, 255, empty)
    cv2.createTrackbar('Val Min', 'TrackBars', 153, 255, empty)
    cv2.createTrackbar('Val Max', 'TrackBars', 255, 255, empty)
init_trackbar()

# Define data
mon_width = 1920
mon_height = 1080

full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}
store_icon = {'top': 0, 'left': (mon_width - 78), 'width': 78, 'height': 64}

sct_full_mon = mss()
sct_store_icon = mss()

container_notified = [0, 179, 0, 250, 0, 255]


while 1:
    sct_full_mon.get_pixels(full_mon)
    sct_store_icon.get_pixels(store_icon)

    # Full monitor
    frameRGB_mon = np.array(Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
    frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)
    frameGray_mon = cv2.cvtColor(frameBGR_mon, cv2.COLOR_BGR2GRAY)

    # Store icon
    frameBGR_si = frameBGR_mon[0:64, (mon_width - 78):mon_width]
    frameCanny_si = cv2.Canny(frameBGR_si, 50, 50)
    si_contour = frameBGR_si.copy()

    # Track bar stuff
    img = frameBGR_mon

    frameHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')  # Get the track bar. Which window to change.
    h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')
    sat_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')
    sat_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')
    v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')
    v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')
    lower = np.array([h_min, sat_min, v_min])  # NumPy array
    upper = np.array([h_max, sat_max, v_max])  # NumPy array
    mask = cv2.inRange(frameHSV, lower, upper)  # Create a mask within the range of these colors.

    frameResult = cv2.bitwise_and(img, img, mask=mask)


    # After config
    frameStack = stack_images(0.75, [frameResult, img, mask])

    # get_contours(frameCanny_si, si_contour, "Container_notify")

    cv2.imshow('test', frameStack)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
