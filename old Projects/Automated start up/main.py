"""
TODO: 1. Boot up Blitz ✓
TODO: 2. Switch off auto-start ✓

"""

import subprocess
import time


def auto_start():
    start_input = input("Start blitz? 0 || 1: ")
    if start_input == '1':
        print("Starting up blitz...")
        subprocess.Popen(['D:\SteamLibrary\steamapps\common\World of Tanks Blitz\wotblitz.exe'])
    else:
        pass
    time.sleep(2)
    print("NOTE: Whenever you stop this program, unless you press 'q', blitz closes automatically")


auto_start()




