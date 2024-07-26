import cv2
import numpy as np
import pyautogui as pag
from PIL import ImageGrab


while True:
    img = pag.screenshot()
    screen = np.array(img)
    cv2.imshow('hallo', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    key = cv2.waitKey(20)
    if key == 27:   # exit on ESC
        break
