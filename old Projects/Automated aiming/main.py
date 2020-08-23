"""
There are 2 kinds of aiming:
    - No-scope -> Aims mouse towards an enemy tank and scopes
    - Scoped -> While scoped-in, the program moves the gun to where there're no tomatoes inside the profile.

No-scoped object colors:
    h_min: 0
    h_max: 1
    s_min: 110
    s_max: 240
    v_min: 153
    v_max: 255

TODO: 1. No-scope aiming ✓
TODO: 2. Scoped aiming
TODO: 3. (Optional) Use auto-aim feature.

"""

import numpy as np
import cv2
from mss import mss
from PIL import Image
import pyautogui
import time
import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

MAPVK_VK_TO_VSC = 0

# msdn.microsoft.com/en-us/library/dd375731
VK_TAB = 0x09
VK_MENU = 0x12
W = 0x57
A = 0x41
S = 0x53
D = 0x44
mouse_left = 0x01
mouse_mid = 0x04
shift = 0x10

# C struct definitions

wintypes.ULONG_PTR = wintypes.WPARAM


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        # some programs use the scan code even if KEYEVENTF_SCANCODE
        # isn't set in dwFflags, so attempt to map the correct code.
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))

    _anonymous_ = ("_input",)
    _fields_ = (("type", wintypes.DWORD),
                ("_input", _INPUT))


LPINPUT = ctypes.POINTER(INPUT)


def _check_count(result, func, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args


user32.SendInput.errcheck = _check_count
user32.SendInput.argtypes = (wintypes.UINT,  # nInputs
                             LPINPUT,  # pInputs
                             ctypes.c_int)  # cbSize


# Functions

def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


if __name__ == "__main__":
    pass


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
        # hor_con = [img_blank] * rows
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


def detect_tomato(src_frame, contour_frame, display_frame, area_threshold, corners_threshold, mon_w, mon_h):
    x = mon_w / 2
    y = mon_h / 2
    contours, hierarchy = cv2.findContours(src_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            cv2.drawContours(contour_frame, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)  # Put a rectangle on the tank
            if len(approx) >= corners_threshold:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(
                    display_frame,
                    "Armor",
                    (x + (w // 2), y + (h // 2)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)
            else:
                pass
    return x, y


def detect_tank(src_frame, contour_frame, display_frame, area_threshold, corners_threshold, mon_w, mon_h):
    x = mon_w / 2
    y = mon_h / 2
    contours, hierarchy = cv2.findContours(src_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            cv2.drawContours(contour_frame, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)  # Put a rectangle on the tank
            # if len(approx) >= corners_threshold:
            #     cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     cv2.putText(
            #         display_frame,
            #         "Enemy tank",
            #         (x + (w // 2) - 10, y + (h // 2) - 10),
            #         cv2.FONT_HERSHEY_COMPLEX,
            #         0.7,
            #         (0, 255, 255),
            #         2)
            # else:
            #     pass
    if x == mon_w / 2 and y == mon_h / 2:
        cv2.putText(display_frame, "No enemy tanks detected", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),
                    2)
    else:
        cv2.putText(display_frame, "Tank spotted! Tracking...", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 255, 0), 2)

    return x, y


def detect_object(src_frame, contour_frame, area_threshold, corners_threshold):  # Input image and define
    detected = False
    contours, hierarchy = cv2.findContours(src_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            cv2.drawContours(contour_frame, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            # print(len(approx))
            if len(approx) >= corners_threshold:
                detected = True
            else:
                detected = False
    return detected


# Track bar stuff
# def empty(i):
#     pass
#
#
# cv2.namedWindow('TrackBars')
# cv2.resizeWindow('TrackBars', 640, 240)
# cv2.createTrackbar('Hue Min', 'TrackBars', 0, 179, empty)
# cv2.createTrackbar('Hue Max', 'TrackBars', 1, 179, empty)
# cv2.createTrackbar('Sat Min', 'TrackBars', 110, 255, empty)
# cv2.createTrackbar('Sat Max', 'TrackBars', 240, 255, empty)
# cv2.createTrackbar('Val Min', 'TrackBars', 153, 255, empty)
# cv2.createTrackbar('Val Max', 'TrackBars', 255, 255, empty)

# Define data
mon_width = 1920
mon_height = 1080

full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}
sct_full_mon = mss()

kernel_x = np.ones((10, 10), np.uint8)
kernel_reticule = np.ones((10, 10), np.uint8)

while 1:
    sct_full_mon.get_pixels(full_mon)

    # Full monitor
    frameRGB_mon = np.array(Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
    frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)
    frameGray_mon = cv2.cvtColor(frameBGR_mon, cv2.COLOR_BGR2GRAY)

    # Track bar stuff
    # h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')
    # h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')
    # sat_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')
    # sat_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')
    # v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')
    # v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')

    # Computer view-port
    frameBGR_CV_view = frameBGR_mon[100:mon_height - 247, 100:(mon_width - 100)]
    modify_CV_view = ContourDilation(frameBGR_CV_view, 0, 0, 110, 240, 153, 255, kernel_x)
    d_x, d_y = detect_tank(modify_CV_view.dilated_mask,
                           modify_CV_view.dilated_contours,
                           frameBGR_CV_view,
                           1100,
                           10, mon_width, mon_height)

    # Aiming reticule scanner
    frameBGR_reticule = CropAndResize(frameBGR_mon,
                                      int(mon_height * 0.40),
                                      int(mon_height * 0.60),
                                      int(mon_width * 0.40),
                                      int(mon_width * 0.60),
                                      200,
                                      150).new_frame_bgr
    modify_reticule = ContourDilation(frameBGR_reticule, 46, 53, 177, 255, 0, 255, kernel_reticule)
    aimed = detect_object(modify_reticule.dilated_mask, modify_reticule.dilated_contours, 0, 8)
    if aimed:
        PressKey(shift)
    if not aimed:
        ReleaseKey(shift)

    if d_x != mon_width / 2 and d_y != mon_height / 2:
        print("Tracking tank...")
        offset_x = 140
        offset_y = 240
        pyautogui.moveTo(d_x + offset_x, d_y + offset_y)
        # VK(0x04, True)

    cv2.imshow('test', frameBGR_CV_view)
    cv2.imshow('reticule', modify_reticule.frame_result)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
