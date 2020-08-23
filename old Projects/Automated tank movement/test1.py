"""
Keyboard stuff is very hard to do. Try doing some automated shooting.

    "w":
        X = 63%
        Y = 89%
    "a":
        X = 62%
        Y = 92%
    "s":
        X = 64%
        Y = 92%
    "d":
        X = 66%
        Y = 92%

"""

import pyautogui
import time

x = 1920
y = 1080


def hold_letter(letter, hold_time):
    start = time.time()
    while time.time() - start < hold_time:
        pyautogui.press(letter)
        print(letter)


def drive():
    while 1:
        pyautogui.press("ctrlleft")

        pyautogui.mouseDown((0.63 * x), (0.89 * y))



def return_to_game():
    pyautogui.click(x / 2, (y / 2) + 50)
    pyautogui.click(x / 2, (y / 2) + 50)
    pyautogui.keyDown('ctrlleft')

    hold_letter('ctrlleft', 2)


return_to_game()
drive()


