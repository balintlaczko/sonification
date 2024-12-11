# %%
# imports
import os
import numpy as np
import cv2

# %%
# generate a random image
size = 256
img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

# show it using opencv
cv2.imshow('random image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# show it again, but get mouse click
mouse_is_down = False

def on_mouse_click(event, x, y, flags, param):
    global mouse_is_down
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'clicked at ({x}, {y})')
        mouse_is_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        print(f'released at ({x}, {y})')
        mouse_is_down = False
    elif event == cv2.EVENT_MOUSEMOVE and mouse_is_down:
        print(f'moving at ({x}, {y})')

cv2.startWindowThread()
cv2.imshow('random image', img)
cv2.setMouseCallback('random image', on_mouse_click)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

# %%