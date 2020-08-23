import pyautogui
import time

x = 1600
y = 900


def return_to_game():
    pyautogui.click(x / 2, (y / 2) + 50)
    pyautogui.click(x / 2, (y / 2) + 50)
    pyautogui.click(x / 2, (y / 2) + 100)


return_to_game()


def hold_W(hold_time):
    start = time.time()
    while time.time() - start < hold_time:
        pyautogui.keyDown("38")


def fire():
    while True:
        pyautogui.click(x / 2, (y / 2))
        pyautogui.click(x / 2, (y / 2))
        time.sleep(4.4)

hold_W(10)