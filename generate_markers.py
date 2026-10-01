import cv2
import os

markerNum = 10  # number of markers for each size
pixelNum = 600

# Marker sizes to be generated
for n in range(4, 8):
    myArucoDict = cv2.aruco.custom_dictionary(markerNum, n, 0)

    if not os.path.isdir(f'./{n}x{n}'):
        os.mkdir(f'{n}x{n}')
    for i in range(0, markerNum):
        tag = cv2.aruco.drawMarker(myArucoDict, i, pixelNum)
        cv2.imwrite(f'./{n}x{n}/{i}.png', tag)




