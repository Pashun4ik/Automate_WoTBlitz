"""
TODO: 1.) Detect reload ✓
TODO: 2.) Detect tomatoes ✓
TODO: 3.) Detect aiming reticule ✓

With respect to mon_width and mon_height:
    - Reload counter area :
        (0.44, 0.5)
        (0.4688, 0.52)
    - Aiming reticule area:
        (0.4271, 0.3704)
        (0.5729, 0.6389)


"""

import numpy as np
import pyautogui
import cv2
from mss import mss
from PIL import Image
import pytesseract
import ctypes
from ctypes import wintypes


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


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


# Define data
mon_width = 1920
mon_height = 1080
full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}
sct_full_mon = mss()

kernel_reload = np.ones((5, 5), np.uint8)
kernel_reticule = np.ones((10, 10), np.uint8)
kernel_tomato = np.ones((5, 5), np.uint8)

reloaded = False
aimed = False

# while loop config
index = 0
index_threshold = 4
DexA, DexB, DexC, DexD = 1, 2, 3, 4
while 1:
    index = index + 1
    # print(index)

    sct_full_mon.get_pixels(full_mon)

    # Full monitor frame
    frameRGB_mon = np.array(Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
    frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)
    frameGray_mon = cv2.cvtColor(frameBGR_mon, cv2.COLOR_BGR2GRAY)

    # Reload counter frame
    frameBGR_reload = CropAndResize(frameBGR_mon,
                                    int(mon_height * 0.5),
                                    int(mon_height * 0.52),
                                    int(mon_width * 0.44),
                                    int(mon_width * 0.4688),
                                    300,
                                    170).new_frame_bgr

    frameGray_reload = cv2.cvtColor(frameBGR_reload, cv2.COLOR_BGR2GRAY, 10, 8)
    frameDilate_reload = cv2.dilate(frameGray_reload, kernel_reload, iterations=1)
    frameEroded_reload = cv2.erode(frameDilate_reload, kernel_reload, iterations=1)
    reload_sec = pytesseract.image_to_string(frameEroded_reload)
    # print(reload_sec)

    # Detect the reload counter
    if not reloaded:
        if index == 1:
            DexA = reload_sec
            print("DexA: " + str(reload_sec))
        elif index == 2:
            DexB = reload_sec
            print("DexB: " + str(reload_sec))
        elif index == 3:
            DexC = reload_sec
            print("DexC: " + str(reload_sec))
        elif index == 4:
            DexD = reload_sec
            print("DexD: " + str(reload_sec))
        # elif index == 5:
        #     DexE = reload_sec
        #     print("DexE: " + str(reload_sec))
        else:
            DexA, DexB, DexC, DexD = 1, 2, 3, 4
        if DexA == DexB == DexC == DexD:
            reloaded = True
            print("Reloaded: " + str(reloaded))

    # Bright tomatoes scanner
    frameBGR_bright = CropAndResize(frameBGR_mon,
                                    int(mon_height * 0.45),
                                    int(mon_height * 0.55),
                                    int(mon_width * 0.45),
                                    int(mon_width * 0.55),
                                    200,
                                    150).new_frame_bgr
    modify_bright = ContourDilation(frameBGR_bright, 140, 179, 0, 255, 0, 255, kernel_tomato)
    detect_bright = detect_object(modify_bright.dilated_mask, modify_bright.dilated_contours, 100, 8)
    if detect_bright:
        print("Tomato! Don't shoot")
    else:
        print("Weak-point")

    # Aiming reticule scanner
    frameBGR_reticule = CropAndResize(frameBGR_mon,
                                      int(mon_height * 0.40),
                                      int(mon_height * 0.60),
                                      int(mon_width * 0.40),
                                      int(mon_width * 0.60),
                                      200,
                                      150).new_frame_bgr
    modify_reticule = ContourDilation(frameBGR_reticule, 46, 53, 177, 255, 0, 255, kernel_reticule)
    aimed = detect_object(modify_reticule.dilated_mask, modify_reticule.dilated_contours, 1000, 8)
    if not aimed:
        if aimed:
            print("Aimed: " + str(aimed))

    # Fire control system
    if reloaded and not detect_bright and aimed:
        print("Fire!")
        pyautogui.leftClick()
        # ReleaseKey(shift)
        reloaded = False
        aimed = False

    cv2.imshow('Reload counter', frameEroded_reload)
    cv2.imshow('Aiming reticule', modify_reticule.frame_result)
    cv2.imshow('Tomato', modify_bright.mask)

    if index == index_threshold:
        index = 0
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
