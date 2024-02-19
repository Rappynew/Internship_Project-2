import cv2
import numpy as np

#reading the image
image = cv2.imread('line.jpg')

#converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#applying Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

#finding contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#initializing variables for storing the thinnest point
thinnest_point = None
min_width = float('inf')

#iterating through the contours to find the thinnest point
for contour in contours:
    #getting the bounded rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    #calculating the width of the bounding rectangle
    width = w
    
    #checking if the current width is smaller than the minimum width found 
    if width < min_width:
        min_width = width
        #Calculating the thinnest point as the center of the bounding rectangle
        thinnest_point = (x + w // 2, y + h // 2)

#marking a dot at the thinnest point on the original image
result_image = image.copy()
cv2.circle(result_image, thinnest_point, 5, (0, 0, 255), -1)

#displaying the result
cv2.imshow('Thinnest Point', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
