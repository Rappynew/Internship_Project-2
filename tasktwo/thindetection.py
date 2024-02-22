import cv2
import numpy as np

#Reading both the images
image = cv2.imread('line.jpg')
focus_ruler = cv2.imread('Fruler.jpg')

#Resizing the focus ruler to match the width of the output image
height, width, _ = image.shape
focus_ruler_resized = cv2.resize(focus_ruler, (width, int(focus_ruler.shape[0] * width / focus_ruler.shape[1])))

#Converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Applying Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

#Detect lines using HoughLines
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

if lines is None or len(lines) == 0:
    raise ValueError("No lines detected.")

#Calculating the distance of each pixel on the line from the center of the image
centerimg = (image.shape[1] // 2, image.shape[0] // 2)

def distance_center(x, y):
    return np.sqrt((x - centerimg[0])**2 + (y - centerimg[1])**2)

#Finding the thinnest point on the line based on the distance from the center
thin_point = min(lines, key=lambda line: distance_center((line[0, 0] + line[0, 2]) // 2, (line[0, 1] + line[0, 3]) // 2))

#marking a small red dot at the thinnest point on the original image
result_image = image.copy()
x, y = (thin_point[0, 0] + thin_point[0, 2]) // 2, (thin_point[0, 1] + thin_point[0, 3]) // 2
cv2.circle(result_image, (x, y), 5, (0, 0, 255), -1)

# Draw an arrow from the thinnest point to the focusruler
arrow_color = (0, 255, 0)  
arrow_thickness = 2
arrow_tip_length = 108
cv2.arrowedLine(result_image, (x, y), (x, y - arrow_tip_length), arrow_color, arrow_thickness)

#Binding the focusruler above the result image
combined_image = np.vstack((focus_ruler_resized, result_image))

#Displaying the result
cv2.imshow('Thinnest Point', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
