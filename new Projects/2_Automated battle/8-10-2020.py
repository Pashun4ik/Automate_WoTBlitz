"""

Process:
monitor frames -> image processing -> detection algorithm -> decision algorithm

In-game:
    Data we're handling with:
        Mini-map:
        - Mini-map terrain outline
        - Mini-map player icon and orientation
        Near player's tank:
        - (F) Auto-aim messages
        - Player's green arrows orientation
        Enemy:
        - (needs work on) Enemy GUIs

    Decisions:
    1. Move forwards and avoid obstacles. Reverse if needed.
    2. If an enemy tank is detected, then aim the gun at it and try to enable auto-aim. Otherwise, keep gun straight

TODO: Optimize enemy detector
TODO: Control flow for detecting WHICH enemy tank to aim at
TODO: Obstacle collision avoidance
TODO: Keep gun straight and align with player tank body
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

# (standard) Define data
mon_width = 1920
mon_height = 1080
full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}
sct_full_mon = mss()

# Define auto data
kernel_enemy = np.ones((10, 10), np.uint8)
kernel_message = np.ones((1, 1), np.uint8)
kernel_icon = np.ones((2, 2), np.uint8)
auto_aimed = False

# Virtual keyboard data
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

# while loop config data
index = 0
index_threshold = 6
DexA, DexB, DexC, DexD = 0, 0, 0, 0

while 1:
    print(index)  # Debug

    ''' Frame grabbing '''
    sct_full_mon.get_pixels(full_mon)

    # Full monitor -> Set monitor frame
    frameRGB_mon = np.array(Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
    frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)
    frameGray_mon = cv2.cvtColor(frameBGR_mon, cv2.COLOR_BGR2GRAY)

    # Auto aim message frame -> For scanning the message at the bottom
    frameBGR_message = CropAndResize(frameBGR_mon,
                                     int(mon_height * (900 / 1080)),
                                     int(mon_height * (1000 / 1080)),
                                     int(mon_width * (880 / 1920)),
                                     int(mon_width - (mon_width * (880 / 1920))),
                                     800, 400).new_frame_bgr  # Crop bottom messages from full monitor

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

    ''' Image processing for emphasizing certain colors and targets '''
    # Emphasize red enemies
    em_enemy_gui = ContourDilation(frameBGR_mon, 0, 0, 215, 240, 220, 245, kernel_enemy)
    # Emphasize the messages that are being received
    em_message_true = ContourDilation(frameBGR_message, 50, 80, 90, 255, 0, 255, kernel_message)
    em_message_false = ContourDilation(frameBGR_message, 7, 14, 171, 255, 220, 255, kernel_message)
    # Emphasize player icon on the map
    em_player_icon = ContourDilation(frameBGR_player_icon, 80, 90, 10, 140, 200, 255, kernel_icon)

    ''' Detection algorithms '''
    # Below refers to the player icon of the mini-map and returns the X Y coordinates from the screen
    pi_x, pi_y, pi_w, pi_h = At.detect_player(em_player_icon.dilated_mask,
                                              em_player_icon.dilated_contours,
                                              frameBGR_player_icon,
                                              100, 4, mon_width, mon_height)
    # Below refers to enemy tanks and returns their X Y coordinates on the screen
    d_x, d_y, d_w, d_h, detected_tank = At.detect_tank(em_enemy_gui.dilated_mask, em_enemy_gui.dilated_contours,
                                                       frameBGR_mon,
                                                       100, 4, mon_width,
                                                       mon_height)
    # Message readers
    message_true = pytesseract.image_to_string(
        em_message_true.frame_result)  # Return what the message below says when auto-aiming
    message_false = pytesseract.image_to_string(
        em_message_false.frame_result)  # Return what the message below says when auto-aiming

    # Detect if whether the player has stopped moving
    player_icon_x = int(pi_x + pi_w / 2)
    player_icon_y = int(pi_y + pi_h / 2)
    coords_area = player_icon_x * player_icon_y
    tolerance = 100
    area_min_threshold = coords_area - tolerance
    area_max_threshold = coords_area + tolerance

    ''' Decision algorithms '''
    # For player navigation
    if index == int(index_threshold * 0.25):
        DexA = coords_area
    elif index == int(index_threshold * 0.5):
        DexB = coords_area
    elif index == int(index_threshold * 0.75):
        DexC = coords_area
    elif index == int(index_threshold * 1):
        DexD = coords_area

    # For enemy tank detection
    """
    When an enemy tank is detected:
        When there is no auto-aim yet:
            - Look towards enemy tank
            - Try to enable auto-aim
            - Move towards enemy position (go forwards, avoid obstacles)
        If auto-aimed:
            - Stop moving
            - Orientate player tank body towards enemy tank
            - Go in reverse
            - Fire gun
    """
    if detected_tank and d_x > map_offset_to_x and d_y < map_offset_from_y:
        print("Tracking tank...")  # Debug
        pyautogui.moveTo(d_x + d_w / 2, d_y)  # Aim at target

        # Decide to set auto_aimed boolean
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
        else:
            At.ReleaseKey(W)  # Stop moving forwards

            if index < index_threshold * 0.5:
                At.PressKey(S)  # Go reverse

                # Decide whether to randomly reverse left or right
                random_decision = int(random.uniform(0, 2))
                if random_decision == 0:
                    At.PressKey(A)
                else:
                    At.PressKey(D)
            else:
                At.ReleaseKey(S)
                At.ReleaseKey(A)
                At.ReleaseKey(D)

            print("Fire!")  # Debug
            pyautogui.leftClick()
    else:
        auto_aimed = False
        At.ReleaseKey(mouse_mid)

        # Handle movement
        average_area = (DexA + DexB + DexC + DexD) / 4
        # print("The average area is: " + str(average_area))  # Debug
        # print("Min and max thresholds: " + str(area_min_threshold) + ' ' + str(area_max_threshold)) # Debug
        if area_min_threshold < average_area < area_max_threshold:
            print("The tank had stopped... Reversing")  # Debug
            At.ReleaseKey(W)
            At.PressKey(S)
            time.sleep(4)
            # Decide random reverse turn
            random_decision = int(random.uniform(0, 2))
            # print(random_decision)  # Debug
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

    pyautogui.moveRel(0, 30)  # Better to look down than to look up
    index += 1
    if index > index_threshold:  # Reset the index of the while-loop
        index = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
