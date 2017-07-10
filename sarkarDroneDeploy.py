import numpy as np
import cv2
from functools import cmp_to_key

im = cv2.imread('IMG_6719.JPG')
im = cv2.resize(im, (500,700)) #image is very large, so for testing purposes, resize to smaller scale
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) #finding contours involves finding white object from black background, so object to be found should be white and background should be black
#convert image to grayscale
ret,thresh = cv2.threshold(imgray,127,255,0) #for better accuracy, use binary images (apply threshold)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contours is a Python list of all the contours, each contour is a Numpy array of coordinates of boundary points of the object

def sortArea(x,y):
    return cv2.contourArea(y) - cv2.contourArea(x) #sort compare function that compares area of each contour
contours.sort(key=cmp_to_key(sortArea)) #sorts contours by area in descending order

maxContour = contours[1] #finds 2nd largest contour, aka the barcode, because largest contour is paper

#finds rotated rectangle, bounding rectangle is drawn with minimum area, so it considers the rotation as well
rect = cv2.minAreaRect(maxContour) #returns a Box2D structure which contains (top-left cornor,(width, height), angle of rotation)
angleofRotation = rect[2] #angle of rotation is third element of rect

(x,y),(MA,ma),angle = cv2.fitEllipse(maxContour) #fitEllipse finds the orientation, or the angle at which the object is directed,
# it also gives the (x,y) coordinates and the Major Axis (MA) and minor axis (ma) lengths

#8.8 cm by 8.8 cm is actual size (3.46 by 3.46 inches) of barcode
#distance of object from camera = actual size * focal length / image size
#focal length is equal to (perceived width * known distance (inches) object from camera) / known width at known distance (inches)

width = rect[1][0]
height = rect[1][1]
focalLength = (width * 24) / (3.46)

distanceFromCamera = (3.46  * focalLength) / (width) #distance between barcode and camera

#draw a bounding box around the image
box = cv2.boxPoints(rect) #finds the four corners of the rectangle
box = np.int0(box)
im = cv2.drawContours(im,[box],0,(0,0,255),2) #draws bounding rectangle

text = "Distance from camera is: " + str(distanceFromCamera) + "inches"
text2 = "Angle of rotation is " + str(angleofRotation) #creates readable text containing info (rotation angle and distance)
font = cv2.FONT_HERSHEY_SIMPLEX
im = cv2.putText(im,text,(0,30), font, 0.7,(100,0,255),1.7) 
im = cv2.putText(im,text2,(0,60), font, 0.7,(100,0,255),1.7) #adds text explanation to the image

cv2.imshow('image', im) #shows image
cv2.waitKey(0) #waits for a pressed key, delay is forever (0 is a special value)
cv2.destroyAllWindows() #distroys windows (in this case 'image')
