"""
TODO: 1. Track player
TODO: 2. Define mini-map contours
TODO: 3. Make player avoid mini-map contours
TODO: 4. Make player be able of fully autonomous navigation

Data for detecting map shape:
    0, 179, 0, 60, 70, 140
"""

import numpy as np
import cv2
from mss import mss
from PIL import Image
import pyautogui
import time
import ctypes
from ctypes import wintypes
import math
import random

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


def detect_player(src_frame, contour_frame, display_frame, area_min, corners_threshold, m_width, m_height):
    x = m_width / 2
    y = m_height / 2
    w = 100
    h = 100
    maxArea = area_min
    biggest = np.array([])
    contours, hierarchy = cv2.findContours(src_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min < area:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
            if area > maxArea and len(approx) >= corners_threshold:
                biggest = approx
                maxArea = area
    cv2.drawContours(contour_frame, biggest, -1, (255, 0, 0), 3)
    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 4)
    return x, y, w, h


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


# Define data
mon_width = 1920
mon_height = 1080

full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}

sct_full_mon = mss()
kernel_icon = np.ones((2, 2), np.uint8)
index = 0
index_threshold = 30
DexA, DexB, DexC, DexD = 0, 0, 0, 0
while 1:
    print(index)
    sct_full_mon.get_pixels(full_mon)  # Setup full monitor

    '''Frames set up'''

    # Full monitor
    frameRGB_mon = np.array(Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
    frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)
    frameGray_mon = cv2.cvtColor(frameBGR_mon, cv2.COLOR_BGR2GRAY)

    # Mini-map view -> View the mini-map at the lower left corner
    map_offset_from_y = int(mon_height - mon_height * 0.2722222222)  # Crop from full monitor
    map_offset_to_y = int(mon_height)
    map_offset_from_x = int(mon_width - mon_width)
    map_offset_to_x = int(mon_width * 0.15)
    map_window_x, map_window_y = 700, 700
    frameBGR_map_view = CropAndResize(frameBGR_mon,
                                      map_offset_from_y,
                                      map_offset_to_y,
                                      map_offset_from_x,
                                      map_offset_to_x,
                                      map_window_x, map_window_y).new_frame_bgr

    # Track player icon on map stuff
    frameBGR_player_icon = CropAndResize(frameBGR_mon,
                                         map_offset_from_y,
                                         map_offset_to_y,
                                         map_offset_from_x,
                                         map_offset_to_x,
                                         map_window_x, map_window_y).new_frame_bgr
    modify_player_icon = ContourDilation(frameBGR_player_icon, 80, 90, 10, 140, 200, 255,
                                         kernel_icon)  # Emphasize player icon
    pi_x, pi_y, pi_w, pi_h = detect_player(modify_player_icon.dilated_mask,
                                           # Return X, Y, W, H, and ratio of player icon
                                           modify_player_icon.dilated_contours,
                                           frameBGR_player_icon,
                                           100, 4, mon_width, mon_height)

    ''' Set values and decisions'''
    # Draw line from (0, 0) to player icon location
    player_icon_x = int(pi_x + pi_w / 2)
    player_icon_y = int(pi_y + pi_h / 2)
    coords_area = player_icon_x * player_icon_y
    area_min_threshold = coords_area - 400
    area_max_threshold = coords_area + 400

    # Detect if the tank is not moving
    PressKey(W)
    if index == int(index_threshold * 0.25):
        DexA = coords_area
        # print("DexA: " + str(coords_area))
    elif index == int(index_threshold * 0.5):
        DexB = coords_area
        # print("DexB: " + str(coords_area))
    elif index == int(index_threshold * 0.75):
        DexC = coords_area
        # print("DexC: " + str(coords_area))
    elif index == int(index_threshold * 1):
        DexD = coords_area
        # print("DexD: " + str(coords_area))
    average_area = (DexA + DexB + DexC + DexD) / 4
    if area_min_threshold < average_area < area_max_threshold:
        print("The tank had stopped... Reversing")
        ReleaseKey(W)
        PressKey(S)
        time.sleep(4)
        random_decision = int(random.uniform(0, 2))
        print(random_decision)
        if random_decision == 0:
            PressKey(A)
        else:
            PressKey(D)
        ReleaseKey(S)
        time.sleep(2)
        ReleaseKey(D)
        ReleaseKey(A)
        PressKey(W)
        time.sleep(3)

    # Show GUIs
    stack = stack_images(0.45, ([frameBGR_player_icon],
                                [modify_player_icon.mask]))
    cv2.imshow('Map views', stack)

    index += 1
    if index > index_threshold:
        index = 0
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
