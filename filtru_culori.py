import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
from numpy.random import uniform
import pdb

def find_color_values_using_trackbar(frame):

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 
    def nothing(x):
        pass

    cv.namedWindow("Trackbar") 
    
    cv.createTrackbar("LH", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LS", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LV", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("UH", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("US", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("UV", "Trackbar", 255, 255, nothing)
    
    
    while True:

        l_h = cv.getTrackbarPos("LH", "Trackbar")
        l_s = cv.getTrackbarPos("LS", "Trackbar")
        l_v = cv.getTrackbarPos("LV", "Trackbar")
        u_h = cv.getTrackbarPos("UH", "Trackbar")
        u_s = cv.getTrackbarPos("US", "Trackbar")
        u_v = cv.getTrackbarPos("UV", "Trackbar")


        l = np.array([l_h, l_s, l_v])
        u = np.array([u_h, u_s, u_v])
        mask_table_hsv = cv.inRange(frame_hsv, l, u)        

        res = cv.bitwise_and(frame, frame, mask=mask_table_hsv) 
        
        # Resize the images before displaying them
        display_frame = cv.resize(frame, (500, 500))  # Adjust the size as needed
        display_mask = cv.resize(mask_table_hsv, (500, 500))
        display_res = cv.resize(res, (500, 500))
           
        cv.imshow("Frame", display_frame)
        cv.imshow("Mask", display_mask)
        cv.imshow("Res", display_res)

        if cv.waitKey(25) & 0xFF == ord('q'):
                break
    cv.destroyAllWindows()
    
img2 = cv.imread("./evaluare/fake_test/1_18.jpg")
img = cv.imread("./antrenare/5_20.jpg")
find_color_values_using_trackbar(img)

low_yellow = (15, 105, 105)
high_yellow = (90, 255, 255)

# Resize and display the final images
display_img = cv.resize(img, (1500, 1500))  # Adjust the size as needed
mask_yellow_hsv = cv.inRange(cv.cvtColor(display_img, cv.COLOR_BGR2HSV), low_yellow, high_yellow)
display_mask_yellow_hsv = cv.resize(mask_yellow_hsv, (500, 500))

cv.imshow('img_initial', display_img)
cv.imshow('mask_yellow_hsv', display_mask_yellow_hsv)
cv.waitKey(0)
cv.destroyAllWindows()