"""
Tools:
    - CropAndResize: For cropping certain parts of the monitor for scanning. Resize so that we can see better.
    - ContourDilation: For emphasizing particular parts in the crop.
    - read_counter(): For using pytesseract to read off of text-information (reload and distance for example).
    - detect_map(): For reading the contours of the map and returning the position of obstacles.
    - detect_tank(): For detecting the red parts of the CV port that belong to enemy tanks.
    - detect_arrows(): For detecting the red arrows. The program chooses the arrow with the biggest area and returns
    data.
    - PressKey(): For simulating virtual keyboard and virtual mouse inputs.
    - ReleaseKey(): For simulating virtual keyboard and virtual mouse inputs.


TODO: 1. Auto start Blitz ✓
TODO: 2. Automate garage menu
TODO: 3. Automate tank movement
TODO: 4. Automate gun aiming
TODO: 5. Automate shooting


1. Auto start Blitz: (pyautogui, CropAndResize, ContourDilation)
TODO: 1A. Return camera to game. ✓
TODO: 1B. Boot-up blitz. ✓
TODO: 1C. Scan the Blitz logo -> Return loading = False
TODO: 1D. Turn off auto_start() and turn on auto_menu() ✓

2. Automate garage menu: (CropAndResize, ContourDilation, pyautogui, ctypes)
TODO: 2A. Scan news (x) -> Exit pop-ups ✓
TODO: 2B. (garage menu) Scan container notification ✓
TODO: 2C. (store menu) Scan sub-menu headers ✓
TODO: 2D. (container menu) Scan container notification for each container. -> Return to garage after opening all.
TODO: 2E. Go to missions tab after scanning the containers.
TODO: 2F. (missions tab) Scroll down and scan the list. -> Return info for when the program starts choosing tanks.
TODO: 2G. At garage menu, click filter, and click whatever tier of tank that is needed by the mission.
TODO: 2H. Pick a tank and go to battle. -> Return chosen tank.
TODO: 2I. Scan map name. -> Return map-name
TODO: 2J. Turn off auto_menu() and turn on auto_in_game() after map loads
    Sent values:
    - Map name -> This is so that the program can strategize.
    - Tank name -> Changes tank behaviour depending on the tier (a slow-ass heavy ain't gonna run circles).

3. Automate in-game processes(map_name, tank_mode, travel_mode):
Contains (all working together, NOT seperately):
    - auto_move(map_name, tank_mode): -> (detect_map(), detect_arrows())
    while travel_mode:
    TODO: 3A_1 Decide way-points using map_name
    TODO: 3A_2. (strict) Scan mini-map for contours.

    -> Way-points: <-
    - Red spots placed on the mini-map by the user or computer, to tell the program to move the player to those
    coordinates.

    TODO: 3A_3. (strict) Orient the tank away from obstacles, and towards any way-points.
    while not travel_mode:
    TODO: 3A_4. Using tank_mode, orient the tank towards its chosen foe.
    TODO: 3A_5.


    - auto_aim and auto_shoot(): -> (CropAndResize, ContourDilation, read_counter(), detect_tank(), detect_arrows())
    while not sniper:
    TODO: 3B_0. detect_arrows() -> Move turret accordingly.
    TODO: 3B_1. (strict) Detect enemy tank's red GUI -> Return position of GUIs using an expandable 2D list.
    TODO: 3B_2. (strict) Scan distance (m) GUI counter -> Return value of meters and adjust Y value aim offset.
    TODO: 3B_3. Using X and Y values, point player's gun at screen coordinates.
        if target == "available!":
    TODO: 3B_4. sniper = True. travel_mode = False
        else:
    TODO: 3B_5. Switch targets if shooting cursor is OFF. Use 2D list from before. travel_mode = True. Re-loop
    while sniper:
    TODO: 3B_6. (strict) Detect armor -> Capture and return position of armor.
    TODO: 3B_7. (strict) Move player's gun away from position of armor.
        if not on_armor:
    TODO: 3B_8. Shoot the gun. Exit sniper-mode -> sniper = False
        else:
        -> Re-loop. Continue to look for non-armor spots.


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
escape = 0x1B

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


# Configuration ENDS --- Define data STARTS
mon_width = 1920  # Screen capture setup
mon_height = 1080
full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}
sct_full_mon = mss()

kernel_logo = np.ones((5, 5), np.uint8)  # Kernels
kernel_x = np.ones((10, 10), np.uint8)


def automate_blitz():  # Contains all of the game automation. Every process is dictated by while loops
    """
    Here lies automated:
        - Store container.
        - Auto-tank selector based on "Missions" data.
        - Auto-battle.

        - Tank movement.
        - Gun aiming.
        - Shooting.
    """
    # Local data:
    # Menu booleans
    _quit = False
    check_loading_logo = True
    look_garage = False
    look_in_store = False
    look_at_containers = False
    look_at_missions = False
    chosen_tank = "randomize"
    loading_battle = False
    in_battle = False

    click_delay = 1
    i = 0
    while not _quit:
        def screen_setup():
            sct_full_mon.get_pixels(full_mon)

            # Full monitor frames
            frameRGB_mon = np.array(
                Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
            frameBGR_mon = cv2.cvtColor(frameRGB_mon, cv2.COLOR_RGB2BGR)

            return frameBGR_mon

        def travel():

        # Garage menu loops
        while check_loading_logo:  # Check the loading logo
            bgr_mon = screen_setup()  # Setup monitor
            i += 1
            print(i)

            # Logo scanner
            logo_offsret_from_y = int(mon_height / 2 - 50)  # Offsets for cropping the loading logo
            logo_offset_to_y = int(mon_height / 2 + 50)
            logo_offset_from_x = int(mon_width / 2 - 50)
            logo_offset_to_x = int(mon_width / 2 + 50)
            bgr_logo = CropAndResize(bgr_mon,
                                     logo_offset_from_y,
                                     logo_offset_to_y,
                                     logo_offset_from_x,
                                     logo_offset_to_x,
                                     200,
                                     200).new_frame_bgr
            modify_logo = ContourDilation(bgr_logo, 0, 90, 110, 225, 0, 255, kernel_logo)  # Emphasize logo
            logo_x, logo_y, detect_logo = detect_object(modify_logo.mask,
                                                        modify_logo.dilated_contours,
                                                        bgr_logo,
                                                        50, 8, mon_width, mon_height)
            cv2.imshow('Logo scanner', bgr_mon)  # info GUI

        while look_garage:  # Garage menu script
            bgr_mon = screen_setup()  # Setup data

            # News (X) scanner
            newsX_offset_from_y = 15  # Offsets for cropping the (x)
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
            modify_newsX = ContourDilation(bgr_newsX, 0, 179, 0, 255, 130, 255, kernel_x)  # Emphasize (x)
            detect_x, x_x, x_y = detect_object(modify_newsX.dilated_mask,
                                               modify_newsX.dilated_contours,
                                               bgr_newsX,
                                               50,
                                               11,
                                               mon_width,
                                               mon_height)
            cv2.imshow('News (X) scanner', bgr_newsX)  # info GUI

            if detect_x:  # Click (x) and close news pop-up
                pyautogui.leftClick(((newsX_offset_from_x + newsX_offset_to_x) / 2),
                                    ((newsX_offset_from_y + newsX_offset_to_y) / 2),
                                    duration=click_delay)
                print("'News': Closed")

            # Store notification scanner
            store_offset_from_y = int(mon_height - mon_height)  # Crop the store icon.
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
            modify_notify = ContourDilation(bgr_notify, 0, 0, 255, 255, 255, 255, kernel_x)  # Emphasize notification
            detect_notification, notify_x, notify_y = detect_object(modify_notify.dilated_mask,
                                                                    modify_newsX.dilated_contours,
                                                                    bgr_notify,
                                                                    0,
                                                                    8,
                                                                    mon_width,
                                                                    mon_height)
            cv2.imshow('Notification scanner', bgr_notify)  # info GUI

            if detect_notification:  # Click store if the notification is detected
                pyautogui.leftClick(((store_offset_from_x + store_offset_to_x) / 2),
                                    ((store_offset_from_y + store_offset_to_y) / 2), duration=click_delay)
                print("Store: Opened")
                cv2.destroyAllWindows()
                look_garage = False  # Stop garage-menu script
                look_in_store = True  # Start store sub-menu script

            else:  # Otherwise, click the missions icon
                pyautogui.leftClick(mon_width * 0.025, mon_height * (14 / 45), duration=click_delay)
                print("Missions: Opened")
                cv2.destroyAllWindows()
                look_garage = False  # Stop garage-menu script
                look_at_missions = True  # Start missions scanner script

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                _quit = True

        while look_in_store:  # Store sub-menu script
            bgr_mon = screen_setup()  # Setup

            # Container scanner in sub menu list
            sub_offset_from_y = int(mon_height * 0.065)  # Crop the sub-menu headers
            sub_offset_to_y = int(mon_height * 0.1222)
            sub_offset_from_x = int(mon_width - mon_width)
            sub_offset_to_x = int(mon_width)
            bgr_submenu = bgr_mon[sub_offset_from_y: sub_offset_to_y, sub_offset_from_x: sub_offset_to_x]
            modify_submenu = ContourDilation(bgr_submenu, 0, 0, 255, 255, 255, 255, kernel_x)  # Emphasize notification
            detect_container, icon_x, icon_y = detect_object(modify_submenu.dilated_mask,
                                                             modify_submenu.dilated_contours,
                                                             bgr_submenu,
                                                             0,
                                                             8,
                                                             mon_width,
                                                             mon_height)
            cv2.imshow('Sub menu scanner', bgr_submenu)  # info GUI

            if detect_container:  # Click the "Containers" header if there's a notification
                pyautogui.leftClick(icon_x - 50, ((sub_offset_from_y + sub_offset_to_y) / 2), duration=click_delay)
                print("CONTAINERS: Opened")
                cv2.destroyAllWindows()
                look_in_store = False  # Stop this loop
                look_at_containers = True  # Call free-container detector loop
                pass
            else:  # Otherwise, exit out of the Store sub-menu.
                PressKey(escape)
                print("Store: Closed")
                cv2.destroyAllWindows()
                ReleaseKey(escape)
                look_in_store = False  # Stop this loop
                look_garage = True  # Recall garage-menu script

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                _quit = True

        while look_at_containers:  # Look for containers that are available for free
            bgr_mon = screen_setup()  # Setup data

            # "Common Container" scanner
            common_offset_from_y = int(mon_height * 0.2444)  # Crop the common container
            common_offset_to_y = int(mon_height)
            common_offset_from_x = int(mon_width - mon_width)
            common_offset_to_x = int(mon_width * 0.375)
            bgr_common = bgr_mon[common_offset_from_y: common_offset_to_y, common_offset_from_x: common_offset_to_x]
            modify_common = ContourDilation(bgr_common, 0, 10, 20, 255, 100, 255, kernel_x)  # Emphasize notification
            detect_common, icon_x, icon_y = detect_object(modify_common.dilated_mask,
                                                          modify_common.dilated_contours,
                                                          bgr_common,
                                                          0,
                                                          4,
                                                          mon_width,
                                                          mon_height)
            cv2.imshow('Common container scanner', modify_common.dilated_mask)  # info GUI

            if detect_common:  # Click the container and open it
                pyautogui.leftClick(icon_x - 50, ((common_offset_from_y + common_offset_to_y) / 2) + 50,
                                    duration=click_delay)
                print("Common Container: Opening")

            # "Big Container" scanner
            big_offset_from_y = int(mon_height * 0.2444)  # Crop the big container
            big_offset_to_y = int(mon_height)
            big_offset_from_x = int(mon_width * 0.375)
            big_offset_to_x = int(mon_width * 0.625)
            bgr_big = bgr_mon[big_offset_from_y: big_offset_to_y, big_offset_from_x: big_offset_to_x]
            modify_big = ContourDilation(bgr_big, 0, 10, 20, 255, 100, 255, kernel_x)  # Emphasize notification
            detect_big, icon_x, icon_y = detect_object(modify_big.dilated_mask,
                                                       modify_big.dilated_contours,
                                                       bgr_big,
                                                       0,
                                                       4,
                                                       mon_width,
                                                       mon_height)
            cv2.imshow('Big container scanner', bgr_big)  # info GUI

            if detect_big:  # Click the big container and open it
                pyautogui.leftClick(icon_x - 50, ((big_offset_from_y + big_offset_to_y) / 2) + 50,
                                    duration=click_delay)
                print("Big Container: Opening")

            # "Huge Container" scanner
            huge_offset_from_y = int(mon_height * 0.2444)  # Crop the huge container
            huge_offset_to_y = int(mon_height)
            huge_offset_from_x = int(mon_width * 0.625)
            huge_offset_to_x = int(mon_width)
            bgr_huge = bgr_mon[huge_offset_from_y: huge_offset_to_y, huge_offset_from_x: huge_offset_to_x]
            modify_huge = ContourDilation(bgr_huge, 0, 10, 20, 255, 100, 255, kernel_x)  # Emphasize notification
            detect_huge, icon_x, icon_y = detect_object(modify_huge.dilated_mask,
                                                        modify_huge.dilated_contours,
                                                        bgr_huge,
                                                        0,
                                                        4,
                                                        mon_width,
                                                        mon_height)
            cv2.imshow('Huge container scanner', bgr_huge)  # info GUI

            if detect_huge:  # Click the huge container and open it
                pyautogui.leftClick(icon_x - 50, ((huge_offset_from_y + huge_offset_to_y) / 2) + 50,
                                    duration=click_delay)
                print("Huge Container: Opening")

            if not detect_common and not detect_big and not detect_huge:  # If there are no free containers, then exit.
                PressKey(escape)
                print("CONTAINERS: Closed")
                ReleaseKey(escape)
                look_at_containers = False  # Stop this loop
                look_garage = True  # Recall garage-menu script

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                _quit = True

        while look_at_missions:  # Check for key words of tank-type inside the missions list.
            bgr_mon = screen_setup()

        while loading_battle:  # Checks the map name and returns it for in_battle.
            bgr_mon = screen_setup()


        while in_battle:  # Contains auto_move, auto_aim, and auto_shoot in one single loop.
            """
            This loop contains the automated process of playing the game because each of the 3 process work and
            communicate with each other. 
            - Receives map_name and tank_type
            - Has these scanners working together:
                - CV view-port -> For enemy tanks
                - Reload counter
                - Distance counter
                - Aiming reticule scanner
                - Red arrows scanner
                - Armor detector
                - Mini-map scanner:
                    - Map layout scanner.
                    - Player icon scanner.
            
            auto_move: (Mini-map scanner,)
            - Moves player to designated way-point on the chosen map. Receives 'fighting' boolean
            - Stops player when it is fighting.
            - Receives fighting.
            -> travel(), fight()
            
            auto_aim: (CV view-port, distance counter, red arrows scanner, aiming reticule scanner)
            - Uses CV view-port to scan enemy tanks.
            - Aims the player's gun at the enemy.
            - Receives
            - Return 
            
            auto_shoot: (Reload counter, armor detector, aiming reticule scanner)
            - Detects where the gun is pointing and decides whether the gun shoots or not, based on what it sees.
            - Returns on_armor, 
            """




        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            _quit = True


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
    pyautogui.moveTo(mon_width / 2, mon_height / 2, duration=1)
    print("NOTE: Whenever you stop this program, unless you press 'q', blitz closes automatically")

    # Call the automated garage menu:
    time.sleep(7)

    automate_blitz()


auto_start()
