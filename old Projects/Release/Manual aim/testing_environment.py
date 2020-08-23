"""
TODO: 1. Auto start Blitz
TODO: 2. Automate garage menu
TODO: 3. Automate tank movement
TODO: 4. Automate gun aiming
TODO: 5. Automate shooting

We are using:
    - CropAndResize class -> For cropping and resizing GUIs
    - ContourDilation class -> For creating dilated contours of our visual data.
    - detect_object function -> For detecting objects with the help of dilated contours.

    - Track-bar sliders -> For data testing.

Remember to:
    - Add cv2.destroyAllWindows() -> when switching between GUIs


"""
# Configuration STARTS
import cv2
import numpy as np
import pyautogui
from mss import mss
from PIL import Image
import subprocess
import time
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def detect_object(src_frame, contour_frame, display_frame, area_threshold, corners_threshold, mon_w, mon_h):
    x = mon_w / 2
    y = mon_h / 2
    detected = False
    contours, hierarchy = cv2.findContours(src_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            cv2.drawContours(contour_frame, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            # print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            if len(approx) >= corners_threshold:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected = True
            else:
                detected = False
    return detected, x, y


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

# Configuration ENDS --- Define data STARTS
mon_width = 1920  # Screen capture setup
mon_height = 1080
full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}
sct_full_mon = mss()

kernel_x = np.ones((10, 10), np.uint8)  # Kernels


def in_battle():
    """
    Here lies automated:
        1.
        - Tank movement.
        - Gun aiming.
        - Shooting.
    """
    pass


def garage_menu():
    """
    Here lies automated:
        - Store container.
        - Auto-tank selector based on "Missions" data.
        - Auto-battle.
    """
    # Local data:
    look_garage = False  # Booleans
    look_in_store = False
    look_at_containers = True

    click_delay = 1

    cv2.destroyAllWindows()

    def empty(i):
        pass

    cv2.namedWindow('TrackBars')
    cv2.resizeWindow('TrackBars', 640, 240)
    cv2.createTrackbar('Hue Min', 'TrackBars', 0, 179, empty)
    cv2.createTrackbar('Hue Max', 'TrackBars', 10, 179, empty)
    cv2.createTrackbar('Sat Min', 'TrackBars', 20, 255, empty)
    cv2.createTrackbar('Sat Max', 'TrackBars', 255, 255, empty)
    cv2.createTrackbar('Val Min', 'TrackBars', 100, 255, empty)
    cv2.createTrackbar('Val Max', 'TrackBars', 255, 255, empty)


    while 1:
        def screen_setup():
            sct_full_mon.get_pixels(full_mon)

            # Full monitor frames
            frameRGB_mon = np.array(
                Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
            frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)
            # frameGray_mon = cv2.cvtColor(frameBGR_mon, cv2.COLOR_BGR2GRAY)

            return frameBGR_mon

        # h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')
        # h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')
        # sat_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')
        # sat_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')
        # v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')
        # v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')

        while look_garage:
            bgr_mon = screen_setup()

            # News (X) scanner


            newsX_offset_from_y = 15
            newsX_offset_to_y = 65
            newsX_offset_from_x = 15
            newsX_offset_to_x = 65
            bgr_newsX = CropAndResize(bgr_mon,
                                      newsX_offset_from_y,
                                      newsX_offset_to_y,
                                      newsX_offset_from_x,
                                      newsX_offset_to_x,
                                      200,
                                      200).new_frame_bgr
            modify_newsX = ContourDilation(bgr_newsX, 0, 179, 0, 255, 130, 255, kernel_x)
            detect_x, x_x, x_y = detect_object(modify_newsX.dilated_mask,
                                               modify_newsX.dilated_contours,
                                               bgr_newsX,
                                               50,
                                               11,
                                               mon_width,
                                               mon_height)
            cv2.imshow('News (X) scanner', bgr_newsX)  # info GUI
            cv2.imshow('News (X) scanner 2', modify_newsX.dilated_mask)  # info GUI

            if detect_x:
                # pyautogui.leftClick(((newsX_offset_from_x + newsX_offset_to_x) / 2),
                #                     ((newsX_offset_from_y + newsX_offset_to_y) / 2),
                #                     duration=click_delay)
                print("'News': Closed")

            # Store notification scanner
            store_offset_from_y = int(mon_height - mon_height)
            store_offset_to_y = int(mon_height * 0.0722)
            store_offset_from_x = int(mon_width * 0.95)
            store_offset_to_x = int(mon_width)
            bgr_notify = CropAndResize(bgr_mon,
                                       store_offset_from_y,
                                       store_offset_to_y,
                                       store_offset_from_x,
                                       store_offset_to_x,
                                       200,
                                       200).new_frame_bgr
            modify_notify = ContourDilation(bgr_notify, 0, 0, 255, 255, 255, 255, kernel_x)
            detect_notification, notify_x, notify_y = detect_object(modify_notify.dilated_mask,
                                                                    modify_newsX.dilated_contours,
                                                                    bgr_notify,
                                                                    0,
                                                                    8,
                                                                    mon_width,
                                                                    mon_height)
            # cv2.imshow('Notification scanner', bgr_notify)  # info GUI

            if detect_notification:
                pyautogui.leftClick(((store_offset_from_x + store_offset_to_x) / 2),
                                    ((store_offset_from_y + store_offset_to_y) / 2), duration=click_delay)
                print("'Store': Opened")
                look_garage = False  # Stop this while loop
                cv2.destroyAllWindows()
                look_in_store = True  # Call store sub-menu detector while-loop
                break

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        while look_in_store:
            bgr_mon = screen_setup()

            # Container scanner in sub menu list
            sub_offset_from_y = int(mon_height * 0.0778)
            sub_offset_to_y = int(mon_height * 0.1222)
            sub_offset_from_x = int(mon_width - mon_width)
            sub_offset_to_x = int(mon_width)
            bgr_submenu = bgr_mon[sub_offset_from_y: sub_offset_to_y, sub_offset_from_x: sub_offset_to_x]
            modify_submenu = ContourDilation(bgr_submenu, 0, 0, 255, 255, 255, 255, kernel_x)
            detect_container, icon_x, icon_y = detect_object(modify_submenu.dilated_mask,
                                                             modify_submenu.dilated_contours,
                                                             bgr_submenu,
                                                             0,
                                                             8,
                                                             mon_width,
                                                             mon_height)
            # Show GUIs
            cv2.imshow('Sub menu scanner', bgr_submenu)

            if detect_container:
                pyautogui.leftClick(icon_x - 50, ((sub_offset_from_y + sub_offset_to_y) / 2), duration=click_delay)
                print("'CONTAINERS': Opened")
                look_garage = False  # Stop this while loop
                look_in_store = False  # Stop this while loop
                cv2.destroyAllWindows()
                look_at_containers = True  # Call free-container detector while-loop
                pass

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        while look_at_containers:
            bgr_mon = screen_setup()

            h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')  # Get the track bar. Which window to change.
            h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')
            sat_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')
            sat_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')
            v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')
            v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')

            # "Common Container" scanner
            common_offset_from_y = int(mon_height * 0.2444)
            common_offset_to_y = int(mon_height)
            common_offset_from_x = int(mon_width - mon_width)
            common_offset_to_x = int(mon_width * 0.375)
            bgr_common = bgr_mon[common_offset_from_y: common_offset_to_y, common_offset_from_x: common_offset_to_x]
            modify_common = ContourDilation(bgr_common, h_min, h_max, sat_min, sat_max, v_min, v_max, kernel_x)
            detect_common, icon_x, icon_y = detect_object(modify_common.dilated_mask,
                                                          modify_common.dilated_contours,
                                                          bgr_common,
                                                          0,
                                                          4,
                                                          mon_width,
                                                          mon_height)
            cv2.imshow('Common container scanner', modify_common.dilated_mask)  # info GUI
            cv2.imshow('Common container scanner 2', bgr_common)  # info GUI


            # if detect_common:
            #     pyautogui.leftClick(icon_x - 50, ((common_offset_from_y + common_offset_to_y) / 2) + 50,
            #                         duration=click_delay)
            #     print("'Common Container': Opening")
            #     look_garage = False  # Stop this while loop
            #     look_in_store = False  # Stop this while loop
            #     cv2.destroyAllWindows()
            #     look_at_containers = True  # Call free-container detector while-loop
            #     pass

            # "Big Container" scanner
            big_offset_from_y = int(mon_height * 0.2444)
            big_offset_to_y = int(mon_height)
            big_offset_from_x = int(mon_width * 0.375)
            big_offset_to_x = int(mon_width * 0.625)
            bgr_big = bgr_mon[big_offset_from_y: big_offset_to_y, big_offset_from_x: big_offset_to_x]
            modify_big = ContourDilation(bgr_big, 0, 0, 255, 255, 255, 255, kernel_x)
            detect_big, icon_x, icon_y = detect_object(modify_big.dilated_mask,
                                                       modify_big.dilated_contours,
                                                       bgr_big,
                                                       0,
                                                       4,
                                                       mon_width,
                                                       mon_height)
            cv2.imshow('Big container scanner', bgr_big)  # info GUI

            # if detect_big:
            #     pyautogui.leftClick(icon_x - 50, ((big_offset_from_y + big_offset_to_y) / 2) + 50,
            #                         duration=click_delay)
            #     print("'Big Container': Opening")
            #     look_garage = False  # Stop this while loop
            #     look_in_store = False  # Stop this while loop
            #     cv2.destroyAllWindows()
            #     look_at_containers = True  # Call free-container detector while-loop
            #     pass

            # "Huge Container" scanner
            huge_offset_from_y = int(mon_height * 0.2444)
            huge_offset_to_y = int(mon_height)
            huge_offset_from_x = int(mon_width * 0.625)
            huge_offset_to_x = int(mon_width)
            bgr_huge = bgr_mon[huge_offset_from_y: huge_offset_to_y, huge_offset_from_x: huge_offset_to_x]
            modify_huge = ContourDilation(bgr_huge, 0, 0, 255, 255, 255, 255, kernel_x)
            detect_huge, icon_x, icon_y = detect_object(modify_huge.dilated_mask,
                                                        modify_huge.dilated_contours,
                                                        bgr_huge,
                                                        0,
                                                        8,
                                                        mon_width,
                                                        mon_height)
            cv2.imshow('Huge container scanner', bgr_huge)  # info GUI

            # if detect_huge:
            #     pyautogui.leftClick(icon_x - 50, ((huge_offset_from_y + huge_offset_to_y) / 2) + 50,
            #                         duration=click_delay)
            #     print("'Huge Container': Opening")
            #     look_garage = False  # Stop this while loop
            #     look_in_store = False  # Stop this while loop
            #     cv2.destroyAllWindows()
            #     look_at_containers = True  # Call free-container detector while-loop
            #     pass

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def auto_start():
    """
    Auto start the game using Python.
    """
    start_input = input("Start blitz? 0 || 1: ")
    if start_input == '1':
        print("Starting up blitz...")
        location = 'D:\SteamLibrary\steamapps\common\World of Tanks Blitz\wotblitz.exe'
        subprocess.Popen([location])
    else:
        # auto_start()
        pass
    time.sleep(2)
    print("NOTE: Whenever you stop this program, unless you press 'q', blitz closes automatically")

    # Call the automated garage menu:
    garage_menu()


auto_start()
