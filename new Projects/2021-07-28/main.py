"""
Automate WoTBlitz is a computer vision program that plays World of Tanks Blitz.
It takes in screen video-input and outputs mouse and keyboard controls.

The main files executes core, which then runs program applications. These
applications execute implementations to fulfill their desired functions.

For now, it scans the left monitor of the computer.

TODO (✓):
    - Blitz Image segmentation
    - Finish command-line argument parser       <-- ACHTUNG
        - flags
    - Control window spawn location
"""

# Base dependencies
from mss import mss  # <- screen capture input
import cv2  # <- computer vision algorithms and image operations
import numpy as np  # <- for array computation
from PIL import Image  # <- for general image manipulation
import pyautogui  # <- for GUI automation & information
import argparse  # <- for parsing arguments

# Core utilities
from Applications import core

# Constants
MON_WIDTH = 1920
MON_HEIGHT = 1080
FULL_MON = {"top": 0, "left": 0, "width": MON_WIDTH, "height": MON_HEIGHT}
sct_full_mon = mss()

# Start
if __name__ == "__main__":

    # Parse and obtain command-line arguments
    arguments = ["--display", "--training", "--battle"]  # List of available flags

    parser = argparse.ArgumentParser(description="Automate WoTBlitz CLI")
    parser.add_argument(arguments[0], type=str, nargs='?', const='True', help="Display visual result (default: True)")
    parser.add_argument(arguments[1], type=str, help="Training mode")
    parser.add_argument(arguments[2], type=str, help="Battle mode (default)")
    args = parser.parse_args()
    print(args)
    print(args.display)
    print(type(args.display))


    # Initialize a result-display window
    window = cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    # Run program loop
    while True:

        # Obtain visual input -> format into numpy array with BGR channels
        screenshot = sct_full_mon.grab(FULL_MON)
        rgb = np.array(Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Pass visual input to core.py for handling processes
        automater = core.Automate(bgr, args)

        # Control-flow access debug
        if cv2.waitKey(1) == 27:
            break
        else:
            # Output visual result to user
            # cv2.imshow("Result", automater.output_frame)
            pass

    # When program loop is over, deconstruct window processes
    cv2.destroyAllWindows()
