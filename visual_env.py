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
import cv2 as cv
from mss import mss
from PIL import Image

# Define standard data
mon_width = 1920
mon_height = 1080
full_mon = {'top': 0, 'left': 0, 'width': mon_width, 'height': mon_height}
sct_full_mon = mss()

# Define OpenCV data
backsub = cv.createBackgroundSubtractorMOG2()

while 1:
    sct_full_mon.get_pixels(full_mon)

    # (standard) Full monitor frame
    frameRGB_mon = np.array(Image.frombytes('RGB', (sct_full_mon.width, sct_full_mon.height), sct_full_mon.image))
    frameBGR_mon = cv.cvtColor(frameRGB_mon, cv.COLOR_RGB2BGR)
    frameGray_mon = cv.cvtColor(frameBGR_mon, cv.COLOR_BGR2GRAY)

    # Background subtraction method
    fg_mask = backsub.apply(frameBGR_mon)
    cv.rectangle(frameBGR_mon, (10, 2), (100, 20), (255, 255, 255), -1)

    # cv.imshow('Screen', frameBGR_mon)
    cv.imshow('FG Mask', fg_mask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break
