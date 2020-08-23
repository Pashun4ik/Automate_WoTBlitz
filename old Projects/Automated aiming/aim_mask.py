"""
The green aiming circle is detected with:
h_min: 46
h_max: 53
s_min: 177
s_max: 255
v_min: 0
v_max: 255

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


def detect_aim(src_img, img_contour):  # Input image and define
    # Define which image and retrieval method. Retrieval method retrieves extreme outer contours. Define approximation.
    detected = False
    contours, hierarchy = cv2.findContours(src_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)  # For each contour, find the area.
        if area > 500:  # Check for minimum area and give a minimum threshold for it.
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)  # Image. Contours. Index (-1 to write). Thickness
            perimeter = cv2.arcLength(cnt, True)  # Find arc length of contours. True = contours are closed.
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter,
                                      True)  # Contour. Define resolution. True = contours are closed.
            print(len(approx))  # Print the numbers of corners of each shape.
            if len(approx) >= 8:
                detected = True
            else:
                detected = False
    return detected


# Define data
mon_width = 1920
mon_height = 1080

full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}

sct_full_mon = mss()
kernel = np.ones((5, 5), np.uint8)

while 1:
    sct_full_mon.get_pixels(full_mon)

    # Full monitor
    frameRGB_mon = np.array(Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
    frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)
    frameGray_mon = cv2.cvtColor(frameBGR_mon, cv2.COLOR_BGR2GRAY)

    # Cursor
    frameBGR_cursor = cv2.resize(frameBGR_mon[500:580, 930:990], (200, 300))
    frameCanny_cursor = cv2.Canny(frameBGR_cursor, 50, 50)
    cursor_contour = frameBGR_cursor.copy()

    # Track bar stuff
    img = frameBGR_cursor

    frameHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([42, 110, 153])  # NumPy array
    upper = np.array([61, 240, 255])  # NumPy array
    mask = cv2.inRange(frameHSV, lower, upper)  # Create a mask within the range of these colors.
    maskDilate = cv2.dilate(mask, kernel, iterations=1)
    frameResult = cv2.bitwise_and(img, img, mask=mask)

    maskContours = maskDilate.copy()

    aimed = detect_aim(maskDilate, maskContours)
    if aimed:
        print("Fire!")

    # After config
    frameStack = stack_images(0.75, [frameResult, img, maskContours])
    cv2.imshow('test', frameStack)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
