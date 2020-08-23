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

TODO: 1. Remove distance-based y_offset. Use index-based bobber instead.


"""

import numpy as np
import cv2
from mss import mss
from PIL import Image
import pyautogui
import time
import random
import pytesseract
from AutoUtility import AutoUtilityFunctions as At, ContourDilation, CropAndResize

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

# (standard) Define data
mon_width = 1920
mon_height = 1080

full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}
sct_full_mon = mss()

# Define data

kernel_x = np.ones((10, 10), np.uint8)
kernel_reticule = np.ones((10, 10), np.uint8)
kernel_reload = np.ones((5, 5), np.uint8)
kernel_message = np.ones((1, 1), np.uint8)
kernel_icon = np.ones((2, 2), np.uint8)

auto_aimed = False

# while loop config
index = 0
index_threshold = 4
DexA, DexB, DexC, DexD = 1, 2, 3, 4
DexA_2, DexB_2, DexC_2, DexD_2 = 0, 0, 0, 0

while 1:
    '''Frames'''
    print(index)
    sct_full_mon.get_pixels(full_mon)

    # Full monitor -> Set monitor frame
    frameRGB_mon = np.array(Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
    frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)
    frameGray_mon = cv2.cvtColor(frameBGR_mon, cv2.COLOR_BGR2GRAY)

    # Computer view-port -> Set computer's viewport
    CV_from_y = 100
    CV_from_x = 100
    frameBGR_CV_view = frameBGR_mon[CV_from_y:mon_height - 247,
                       CV_from_x:(mon_width - 100)]  # Crop view port out of full monitor
    modify_CV_view = ContourDilation(frameBGR_CV_view, 0, 0, 215, 240, 220, 245, kernel_x)  # Emphasize red targets

    # Auto aim message frame -> For scanning the message at the bottom
    frameBGR_message = CropAndResize(frameBGR_mon,
                                     int(mon_height * (900 / 1080)),
                                     int(mon_height * (1000 / 1080)),
                                     int(mon_width * (880 / 1920)),
                                     int(mon_width - (mon_width * (880 / 1920))),
                                     800, 400).new_frame_bgr
    modify_message_true = ContourDilation(frameBGR_message, 50, 80, 90, 255, 0, 255, kernel_message)
    modify_message_false = ContourDilation(frameBGR_message, 7, 14, 171, 255, 220, 255, kernel_message)

    # Mini-map view -> View the mini-map at the lower left corner
    map_offset_from_y = int(mon_height - mon_height * 0.2722222222)  # Crop from full monitor
    map_offset_to_y = int(mon_height)
    map_offset_from_x = int(mon_width - mon_width)
    map_offset_to_x = int(mon_width * 0.15)
    map_window_x, map_window_y = 700, 700

    # Track player icon on map stuff
    frameBGR_player_icon = CropAndResize(frameBGR_mon,
                                         map_offset_from_y,
                                         map_offset_to_y,
                                         map_offset_from_x,
                                         map_offset_to_x,
                                         map_window_x, map_window_y).new_frame_bgr
    modify_player_icon = ContourDilation(frameBGR_player_icon, 80, 90, 10, 140, 200, 255,
                                         kernel_icon)  # Emphasize player icon

    ''' Set values and decision making '''
    # Values:
    pi_x, pi_y, pi_w, pi_h = At.detect_player(modify_player_icon.dilated_mask,
                                              # Return X, Y, W, H, and ratio of player icon
                                              modify_player_icon.dilated_contours,
                                              frameBGR_player_icon,
                                              100, 4, mon_width, mon_height)
    d_x, d_y, d_w, d_h, detected_tank = At.detect_tank(modify_CV_view.dilated_mask, modify_CV_view.dilated_contours,
                                                       frameBGR_CV_view,
                                                       100, 4, mon_width,
                                                       mon_height)  # Return X and Y position of target
    message_true = pytesseract.image_to_string(
        modify_message_true.frame_result)  # Return what the message below says when auto-aiming
    message_false = pytesseract.image_to_string(
        modify_message_false.frame_result)  # Return what the message below says when auto-aiming

    # Draw line from (0, 0) to player icon location
    player_icon_x = int(pi_x + pi_w / 2)
    player_icon_y = int(pi_y + pi_h / 2)
    coords_area = player_icon_x * player_icon_y
    tolerance = 100
    area_min_threshold = coords_area - tolerance
    area_max_threshold = coords_area + tolerance

    offset_y = 150
    if index == int(index_threshold * 0.25):
        DexA_2 = coords_area
        offset_y = -75
    elif index == int(index_threshold * 0.5):
        DexB_2 = coords_area
        offset_y = -25
    elif index == int(index_threshold * 0.75):
        DexC_2 = coords_area
        offset_y = 25
    elif index == int(index_threshold * 1):
        DexD_2 = coords_area
        offset_y = 75
        # At.PressKey(R)
        # At.PressKey(E)
        # At.ReleaseKey(R)
        # At.ReleaseKey(E)

    # Track the position of the target if it exists.
    if detected_tank:
        print("Tracking tank...")
        pyautogui.moveTo(CV_from_x + d_x + d_w / 2, CV_from_y + d_y + offset_y)

        At.ReleaseKey(W)
        if index < index_threshold * 0.25:
            At.PressKey(S)
            random_decision = int(random.uniform(0, 2))
            if random_decision == 0:
                At.PressKey(A)
            else:
                At.PressKey(D)
        else:
            At.ReleaseKey(S)
            At.ReleaseKey(A)
            At.ReleaseKey(D)

        if message_true == "Auto-aim enabled" and not auto_aimed:
            auto_aimed = True
        if message_false == "Auto-aim disabled" and auto_aimed:
            auto_aimed = False

        if not auto_aimed:
            pyautogui.middleClick()
            At.PressKey(mouse_mid)
            if message_true == "Auto-aim enabled":
                auto_aimed = True
                if not auto_aimed:
                    auto_aimed = False
                    At.ReleaseKey(mouse_mid)
        elif auto_aimed:
            print("Fire!")
            pyautogui.leftClick()
    else:
        auto_aimed = False
        At.ReleaseKey(mouse_mid)

        # Look left and right from time to time
        if random_decision == 0:
            value = 1
        else:
            value = -1
        if index == 0:
            pyautogui.moveRel(500 * value, 0)
            random_decision = int(random.uniform(0, 2))
        elif index == int(index_threshold / 2):
            pyautogui.moveRel(500 * value * -1, 0)

        average_area = (DexA_2 + DexB_2 + DexC_2 + DexD_2) / 4
        print("The average area is: " + str(average_area))
        print("Min and max thresholds: " + str(area_min_threshold) + ' ' + str(area_max_threshold))
        # Handle movement
        if area_min_threshold < average_area < area_max_threshold:
            print("The tank had stopped... Reversing")

            At.ReleaseKey(W)
            At.PressKey(S)
            time.sleep(4)

            random_decision = int(random.uniform(0, 2))
            print(random_decision)
            if random_decision == 0:
                At.PressKey(A)
            else:
                At.PressKey(D)
            At.ReleaseKey(S)
            time.sleep(2)
            At.ReleaseKey(D)
            At.ReleaseKey(A)
            At.PressKey(W)
            time.sleep(3)
        elif not auto_aimed:
            # At.PressKey(T)
            At.PressKey(W)
        # At.ReleaseKey(T)

    index += 1
    pyautogui.moveRel(0, 30)  # Look down
    if index > index_threshold:  # Reset the index of the while-loop
        index = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
