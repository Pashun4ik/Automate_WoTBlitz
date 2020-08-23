"""
TODO: 1. Environmental-based movement
TODO: 2. Mini-map-based movement

Possible environmental Canny thresholds:
    A = 200
    B = 50
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


def detect_env(src_frame, contour_frame, display_frame, area_min, area_max, corners_threshold, m_width):
    x = m_width / 2
    y = 1
    w = 1
    h = 1
    contours, hierarchy = cv2.findContours(src_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min < area < area_max:
            cv2.drawContours(contour_frame, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
            if len(approx) >= corners_threshold:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.line(display_frame, (0, y), (int(x + w / 2), int(y + h)), (0, 255, 255), 2)
    return x, y, w, h


# Track bar stuff
def empty(i):
    pass

cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars', 640, 240)
cv2.createTrackbar('A', 'TrackBars', 255, 255, empty)
cv2.createTrackbar('B', 'TrackBars', 0, 255, empty)

# Define data
mon_width = 1920
mon_height = 1080

full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}

sct_full_mon = mss()
kernel_env = np.ones((2, 2), np.uint8)

index = 0
index_threshold = 5
go_left_duration = 0
go_right_duration = 0
go_reverse_duration = 0

obs_x, obs_y, obs_w, obs_h = mon_width / 2, 1, 0, 1

def go_left():
    PressKey(A)
def go_right():
    PressKey(D)
def go_forwards():
    PressKey(W)
def go_backwards():
    PressKey(S)

while 1:
    sct_full_mon.get_pixels(full_mon)

    # Track bar stuff
    a = cv2.getTrackbarPos('A', 'TrackBars')
    b = cv2.getTrackbarPos('B', 'TrackBars')

    # Full monitor
    frameRGB_mon = np.array(Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
    frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)
    frameGray_mon = cv2.cvtColor(frameBGR_mon, cv2.COLOR_BGR2GRAY)
    frameCanny_mon = cv2.Canny(frameGray_mon, a, b)

    # CV view-port stuff
    port_offset_from_y = int(100)
    port_offset_to_y = int(mon_height - 247)
    port_offset_from_x = int(100)
    port_offset_to_x = int(mon_width - 100)
    frameBGR_CV_view = frameBGR_mon[port_offset_from_y: port_offset_to_y, port_offset_from_x: port_offset_to_x]
    frameCanny_CV_view = frameCanny_mon[port_offset_from_y: port_offset_to_y, port_offset_from_x: port_offset_to_x]
    dilate_canny_CV_view = cv2.dilate(frameCanny_CV_view, kernel_env, iterations=1)
    copy = dilate_canny_CV_view.copy()

    # Environment detection:
    CV_view_width = int(mon_width - (port_offset_from_x + port_offset_to_x))
    CV_view_width = int(mon_width - (port_offset_from_x + port_offset_to_x))

        # Detect on the x-axis
    if index == 0:
        obs_x, obs_y, obs_w, obs_h = detect_env(dilate_canny_CV_view,
                                                copy,
                                                frameBGR_CV_view,
                                                3000,
                                                int((mon_width * mon_height * 0.25)),
                                                4,
                                                mon_width)
        if (obs_x + obs_w / 2) < (mon_width * 0.25) or (obs_x + obs_w / 2) > (mon_width * 0.75):
            if (obs_x + obs_w / 2) < (mon_width / 2):
                # go_right_duration = index_threshold / 2
                pass
            else:
                # go_left_duration = index_threshold / 2
                pass
        elif (mon_width * 0.25) < (obs_x + obs_w / 2) < (mon_width * 0.5) or (mon_width * 0.5) < (obs_x + obs_w / 2) < (
                mon_width * 0.75):
            if (obs_x + obs_w / 2) < (mon_width / 2):
                go_right_duration = index_threshold
            else:
                go_left_duration = index_threshold
            # Detect on the y-axis
        if (obs_y + obs_h) < (mon_height / 2):
            go_forwards()
        else:
            go_reverse_duration = index_threshold
            # Detect durations -> Press keys
        if go_left_duration != 0:
            go_left()
            go_left_duration -= 1
        elif go_right_duration != 0:
            go_right()
            go_right_duration -= 1
        elif go_reverse_duration != 0:
            go_backwards()
            go_reverse_duration -= 1
        # Detect durations -> _Release keys
    if go_left_duration == 0:
        ReleaseKey(A)
    elif go_right_duration == 0:
        ReleaseKey(D)
    elif go_reverse_duration == 0:
        ReleaseKey(S)

    # stack = stack_images(1, [frameCanny_CV_view, dilate_canny_CV_view])
    # cv2.imshow('Environment', dilate_canny_CV_view)
    cv2.imshow('Environment 2', frameBGR_CV_view)

    index += 1
    if index == index_threshold:
        index = 0
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
