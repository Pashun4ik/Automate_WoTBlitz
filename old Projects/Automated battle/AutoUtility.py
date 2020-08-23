import cv2 as cv
import numpy as np
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
T = 0x54
R = 0x52
E = 0x45
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


class CropAndResize:
    def __init__(self, from_frame_bgr, y_start, y_end, x_start, x_end, x_size, y_size):
        self.from_frame_bgr = from_frame_bgr
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.x_size = x_size
        self.y_size = y_size
        self.new_frame_bgr = cv.resize(from_frame_bgr[
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

        frame_hsv = cv.cvtColor(bgr2hsv, cv.COLOR_BGR2HSV)
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])

        # Use these attributes
        self.mask = cv.inRange(frame_hsv, lower, upper)
        self.dilated_mask = cv.dilate(self.mask, self.kernel, iterations=1)
        self.dilated_contours = self.dilated_mask.copy()
        self.frame_result = cv.bitwise_and(self.bgr2hsv, self.bgr2hsv, mask=self.mask)


class AutoUtilityFunctions:

    @staticmethod
    def PressKey(hexKeyCode):
        x = INPUT(type=INPUT_KEYBOARD,
                  ki=KEYBDINPUT(wVk=hexKeyCode))
        user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

    @staticmethod
    def ReleaseKey(hexKeyCode):
        x = INPUT(type=INPUT_KEYBOARD,
                  ki=KEYBDINPUT(wVk=hexKeyCode,
                                dwFlags=KEYEVENTF_KEYUP))
        user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

    @staticmethod
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
                        img_array[x][y] = cv.resize(img_array[x][y], (0, 0), None, scale, scale)
                    else:
                        img_array[x][y] = cv.resize(img_array[x][y],
                                                    (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                    None, scale, scale)
                    if len(img_array[x][y].shape) == 2:
                        img_array[x][y] = cv.cvtColor(img_array[x][y], cv.COLOR_GRAY2BGR)
            img_blank = np.zeros((height, width, 3), np.uint8)
            hor = [img_blank] * rows
            # hor_con = [img_blank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(img_array[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if img_array[x].shape[:2] == img_array[0].shape[:2]:
                    img_array[x] = cv.resize(img_array[x], (0, 0), None, scale, scale)
                else:
                    img_array[x] = cv.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                             scale)
                if len(img_array[x].shape) == 2:
                    img_array[x] = cv.cvtColor(img_array[x], cv.COLOR_GRAY2BGR)
            hor = np.hstack(img_array)
            ver = hor
        return ver

    @staticmethod
    def detect_tomato(src_frame, contour_frame, display_frame, area_threshold, corners_threshold, mon_w, mon_h):
        x = mon_w / 2
        y = mon_h / 2
        contours, hierarchy = cv.findContours(src_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > area_threshold:
                cv.drawContours(contour_frame, cnt, -1, (255, 0, 0), 3)
                perimeter = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True)
                # print(len(approx))
                x, y, w, h = cv.boundingRect(approx)  # Put a rectangle on the tank
                if len(approx) >= corners_threshold:
                    cv.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv.putText(
                        display_frame,
                        "Armor",
                        (x + (w // 2), y + (h // 2)),
                        cv.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)
                else:
                    pass
        return x, y

    @staticmethod
    def detect_tank(src_frame, contour_frame, display_frame, area_threshold, corners_threshold, mon_w, mon_h):
        x = mon_w / 2
        y = mon_h / 2
        w = 1
        h = 1
        detected = False
        biggest_area = area_threshold
        contours, hierarchy = cv.findContours(src_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > area_threshold or area > biggest_area:
                biggest_area = area
                cv.drawContours(contour_frame, cnt, -1, (255, 0, 0), 3)
                perimeter = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True)
                # print(len(approx))
                x, y, w, h = cv.boundingRect(approx)  # Put a rectangle on the tank
                if len(approx) >= corners_threshold:
                    cv.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(
                        display_frame,
                        "Enemy tank",
                        (x + (w // 2) - 10, y + (h // 2) - 1),
                        cv.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)
                    detected = True
                else:
                    pass
        # if x == mon_w / 2 and y == mon_h / 2:
        #     cv.putText(display_frame, "No enemy tanks detected", (100, 100), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),
        #                 2)
        # else:
        #     cv.putText(display_frame, "Tank spotted! Tracking...", (100, 100), cv.FONT_HERSHEY_COMPLEX, 0.7,
        #                 (0, 255, 0), 2)

        return x, y, w, h, detected

    @staticmethod
    def detect_object(src_frame, contour_frame, area_threshold, corners_threshold):  # Input image and define
        detected = False
        contours, hierarchy = cv.findContours(src_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > area_threshold:
                cv.drawContours(contour_frame, cnt, -1, (255, 0, 0), 3)
                perimeter = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True)
                # print(len(approx))
                if len(approx) >= corners_threshold:
                    detected = True
                else:
                    detected = False
        return detected

    @staticmethod
    def detect_player(src_frame, contour_frame, display_frame, area_min, corners_threshold, m_width, m_height):
        x = m_width / 2
        y = m_height / 2
        w = 100
        h = 100
        maxArea = area_min
        biggest = np.array([])
        contours, hierarchy = cv.findContours(src_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area_min < area:
                perimeter = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True)
                x, y, w, h = cv.boundingRect(approx)
                if area > maxArea and len(approx) >= corners_threshold:
                    biggest = approx
                    maxArea = area
        cv.drawContours(contour_frame, biggest, -1, (255, 0, 0), 3)
        cv.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 4)
        return x, y, w, h

    @staticmethod
    def debug():
        print('Debug')
