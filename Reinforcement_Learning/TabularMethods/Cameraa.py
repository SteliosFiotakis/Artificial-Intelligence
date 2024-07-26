import cv2
import numpy as np
import pyautogui as pag
from PIL import Image



# Camera input
vc = cv2.VideoCapture(0)

if vc.isOpened():   # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


while True:
    # img = pag.screenshot()      # screen input
    # img = ImageGrab.grab()      # screen input

    # test 1 2
    # offset, size, data = Image.core.grabscreen_win32(False, False)
    # img = Image.frombytes("RGB", size, data, "raw", "BGR", (size[0] * 3 + 3) & -4, -1,)
    #
    # frame = np.array(img)       # screen input

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # camera input
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)          # camera input


    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    cv2.imshow('Sobel X', sobelx)
    cv2.imshow('Sobel Y', sobely)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)


    # Canny Edge Detection
    edges_original = cv2.Canny(image=frame, threshold1=100, threshold2=200)  # Canny Edge Detection
    edges_grayscale = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)  # Canny Edge Detection
    edges_blur = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection

    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection original', edges_original)
    cv2.imshow('Canny Edge Detection grayscale', edges_grayscale)
    cv2.imshow('Canny Edge Detection blur', edges_blur)

    cv2.imshow("preview", frame)  # camera input
    rval, frame = vc.read()   # camera input
    key = cv2.waitKey(20)
    if key == 27:   # exit on ESC
        break

vc.release()      # camera input
cv2.destroyAllWindows()
