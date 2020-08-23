"""
TODO: 1. Scan News (X) and close it ✓
TODO: 2. Scan Store/container -> notification. Open containers -> Return to garage menu -> Repeat
TODO: 3. Scan missions and pick appropriate tanks
TODO: 4. Auto-battle

News(x) area:
(15, 15)
(50, 50)

News(x) colors:
    h_min: 0
    h_max: 179
    s_min: 0
    s_max: 255
    v_min: 0
    v_max: 50

Store notification colors:
    h_min: 0
    h_max: 0
    s_min: 255
    s_max: 255
    v_min: 255
    v_max: 255
"""
import pyautogui
import cv2
from mss import mss
import numpy as np
from PIL import Image


class CropAndResize:
    def __init__(self, from_frame_bgr, y_start, y_end, x_start, x_end, x_size, y_size):
        self.from_frame_bgr = from_frame_bgr
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.x_size = x_size
        self.y_size = y_size
        self.new_frame_bgr = cv2.resize(from_frame_bgr[
                                        self.y_start: self.y_end,
                                        self.x_start: self.x_end],
                                        (self.x_size, self.y_size))


def detect_object(src_frame, contour_frame, area_threshold, corners_threshold):  # Input image and define
    detected = False
    contours, hierarchy = cv2.findContours(src_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            cv2.drawContours(contour_frame, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            print(len(approx))
            if len(approx) >= corners_threshold:
                detected = True
            else:
                detected = False
    return detected


class ContourDilation:
    def __init__(self, bgr2hsv, h_min, h_max, s_min, s_max, v_min, v_max, kernel):
        self.bgr2hsv = bgr2hsv
        self.h_min = h_min
        self.h_max = h_max
        self.s_min = s_min
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max
        self.kernel = kernel

        frame_hsv = cv2.cvtColor(bgr2hsv, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])

        # Use these attributes
        self.mask = cv2.inRange(frame_hsv, lower, upper)
        self.dilated_mask = cv2.dilate(self.mask, self.kernel, iterations=1)
        self.dilated_contours = self.dilated_mask.copy()
        self.frame_result = cv2.bitwise_and(self.bgr2hsv, self.bgr2hsv, mask=self.mask)


# def empty(i):
#     pass
#
# cap = cv2.imread('news(x).png')
# cv2.namedWindow('TrackBars')
# cv2.resizeWindow('TrackBars', 640, 240)
# cv2.createTrackbar('Hue Min', 'TrackBars', 46, 179, empty)
# cv2.createTrackbar('Hue Max', 'TrackBars', 53, 179, empty)
# cv2.createTrackbar('Sat Min', 'TrackBars', 177, 255, empty)
# cv2.createTrackbar('Sat Max', 'TrackBars', 255, 255, empty)
# cv2.createTrackbar('Val Min', 'TrackBars', 0, 255, empty)
# cv2.createTrackbar('Val Max', 'TrackBars', 255, 255, empty)


# Define data
mon_width = 1600
mon_height = 900

full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}
sct_full_mon = mss()

kernel_x = np.ones((10, 10), np.uint8)
look_garage = True
look_store = False

while 1:
    sct_full_mon.get_pixels(full_mon)

    # Full monitor frames
    frameRGB_mon = np.array(Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
    frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)
    frameGray_mon = cv2.cvtColor(frameBGR_mon, cv2.COLOR_BGR2GRAY)

    # h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')
    # h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')
    # sat_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')
    # sat_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')
    # v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')
    # v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')

    while look_garage:
        # News (X) scanner
        bgr_newsX = CropAndResize(frameBGR_mon, 15, 50, 15, 50, 200, 200).new_frame_bgr
        modify_newsX = ContourDilation(bgr_newsX, 0, 0, 0, 255, 0, 50, kernel_x)
        detect_x = detect_object(modify_newsX.dilated_mask, modify_newsX.dilated_contours, 0, 11)
        if detect_x:
            pyautogui.leftClick(32, 32)
            print("News closed")

        # Store notification scanner
        bgr_notify = CropAndResize(frameBGR_mon, 0, 65, (mon_width - 78), mon_width, 200, 200).new_frame_bgr
        modify_notify = ContourDilation(bgr_notify, 0, 0, 255, 255, 255, 255, kernel_x)
        detect_notification = detect_object(modify_notify.dilated_mask, modify_newsX.dilated_contours, 0, 8)
        if detect_notification:
            pyautogui.leftClick((mon_width - 50), 20)
            look_garage = False
            look_store = True
            cv2.destroyAllWindows()
        else:
            cv2.imshow('News (X) scanner', bgr_newsX)
            cv2.imshow('Notification scanner', modify_notify.dilated_mask)

    while look_store:
        # Container scanner in sub menu list
        bgr_submenu = CropAndResize(frameBGR_mon, 70, 100, 0, mon_width, 200, 200).new_frame_bgr
        modify_submenu = ContourDilation(bgr_submenu, 0, 0, 255, 255, 255, 255, kernel_x)
        detect_container = detect_object(modify_submenu.dilated_mask, modify_submenu.dilated_contours, 0, 11)
        if detect_container:
            # pyautogui.leftClick(32, 32)
            # print("News closed")
            pass
        else:
            cv2.imshow('Sub menu scanner', bgr_submenu)



    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
