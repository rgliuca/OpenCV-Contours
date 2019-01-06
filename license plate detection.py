import cv2
import numpy as np

# Read image
im_color=cv2.imread("license plate\\CA6.jpg")

im=cv2.imread("license plate\\CA6.jpg", cv2.IMREAD_GRAYSCALE)
#im=cv2.imread("images\\blob.jpg", cv2.IMREAD_GRAYSCALE)
#im=cv2.imread("images\\rubiks1.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("BW image", im)


#im = cv2.GaussianBlur(im, (5, 5), 0)
#im = cv2.bilateralFilter(im,11,100,100)
im = cv2.bilateralFilter(im, 9, 75, 75)
cv2.imshow("Bilateral image", im)


edged = cv2.Canny(im, 230, 255)
cv2.imshow("Edge Detection Image", edged)

#now let's dialate the lines
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(edged, kernel, iterations=1)

cv2.imshow("Dilated Image", dilated)

im2, cnts, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts= sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
#cv2.drawContours(im, cnts, -1, (0,255,0), 3)
#cv2.imshow("contours", im)


sq_cont=[]
contour_centers=[]
for contour in cnts:
    # calculate the contour perimeter
    peri = cv2.arcLength(contour, True)
    # approximate the contour with a polygon 0.1*peri is the max error allowed
    # between the approximation and the original contour
    approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    if len(approx)==4 and w>1.5*h: # and cv2.contourArea(contour)>200:
            sq_cont.append(contour)
            m = cv2.moments(contour)
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            contour_centers.append([cx,cy])
            
            
cv2.drawContours(im_color, sq_cont, -1, (0,255,0), 3)
cv2.imshow("Final Image", im_color)
